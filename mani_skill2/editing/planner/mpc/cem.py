import multiprocessing as mp
from typing import Optional

import gym
import numpy as np
import sapien.core as sapien
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from mani_skill2.editing.keyframe_editor import MSDuration, MSKeyFrame
from mani_skill2.editing.serialization import SerializedEnv
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.logging_utils import logger

from ..base import BasePlanner


def _ms2_env_fn(env_id: str, idx: int):
    def _env_fn():
        env: BaseEnv = gym.make(env_id, obs_mode="none", control_mode="pd_ee_delta_pos")
        env.reset(seed=idx)

        return env

    return _env_fn


class CEM(BasePlanner):
    """Cross Entropy Method Planner"""

    def __init__(
        self,
        population: int,
        elite: int,
        sample_env: SerializedEnv,
        horizon: int,
        cem_iter: int = 10,
        lr: float = 1.0,
        use_history: bool = True,
        num_wip_envs: int = mp.cpu_count(),
        record_dir: Optional[str] = None,
        seed: Optional[int] = None,
        engine: Optional[sapien.Engine] = None,
    ):
        self.population = population
        self.elite = elite
        self.horizon = horizon
        self.cem_iter = cem_iter
        self.lr = lr
        self.use_history = use_history
        self.num_wip_envs = num_wip_envs

        super().__init__(
            senv=sample_env, seed=seed, engine=engine, record_dir=record_dir
        )

    def reset(self, senv: SerializedEnv):
        super().reset(senv)

        if hasattr(self, "_wip_envs") and isinstance(self._wip_envs, SubprocVecEnv):
            self._wip_envs.close()
        self._wip_envs = self._duplicate_envs(senv, self.num_wip_envs)

        self.current_actions = None
        self.step = 0

        # Initial distribution
        # NOTE: we only support environments with single controller for now
        if self.action_space.is_bounded():
            self.lb, self.ub = self.action_space.low, self.action_space.high
        else:
            self.lb, self.ub = -np.inf, np.inf

        self.init_mean = ((self.lb + self.ub) * 0.5)[None, :].repeat(
            self.horizon, axis=0
        )
        self.init_std = ((self.action_space.high - self.action_space.low) * 0.5)[
            None, ...
        ].repeat(self.horizon, axis=0)

        self._odd_batch_warning = False

    def close(self):
        """Close the planner, along with all of its associated environments."""

        self._wip_envs.close()
        super().close()

    def _duplicate_envs(
        self,
        senv: SerializedEnv,
        num_envs: int,
        spawn: bool = True,
    ) -> SubprocVecEnv:
        """Generate VecEnv for internal evaluation use.

        Args:
            senv (SerializedEnv): the serialized environment
            num_scenes (int): the number of scenes to be duplicated
            spawn (bool, optional): whether to use spawn for vectorized environments. Defaults to True.

        Returns:
            SubprocVecEnv: the vectorized environment
        """

        env_fns = [_ms2_env_fn(senv.env_id, i) for i in range(num_envs)]

        env = SubprocVecEnv(env_fns, start_method="spawn" if spawn else "fork")
        env.reset()
        env.env_method("set_state", senv.state)

        return env

    def _eval(self, state: np.ndarray, samples: np.ndarray):
        """Evaluate the samples and return the reward of the samples.

        Args:
            state (np.ndarray): the state of the environment
            samples (np.ndarray): the samples to be evaluated, has the shape (num_samples, horizon, action_dim)

        Returns:
            np.ndarray: the reward of the samples, has the shape (num_samples, horizon)
        """

        reward_per_step = np.zeros(shape=(self.population, self.horizon))

        if self.num_wip_envs != self.population:
            num_iters = self.population // self.num_wip_envs
            if self.population % self.num_wip_envs != 0:
                num_iters += 1

                if not self._odd_batch_warning:
                    logger.warning(
                        "The population is not divisible by the number of envs."
                        " Last batch will be padded."
                    )
                    self._odd_batch_warning = True
        else:
            num_iters = self.num_wip_envs

        for i in range(num_iters):
            self._wip_envs.reset()
            self._wip_envs.env_method("set_state", state)

            if self._wip_envs.num_envs != self.population:
                _samples = samples[i * self.num_wip_envs : (i + 1) * self.num_wip_envs]
                _sample_size = _samples.shape[0]
                _samples = _samples.repeat(self.num_wip_envs, axis=0)[
                    : self.num_wip_envs
                ]
            else:
                _samples = samples
                _sample_size = self.population

            for j in range(self.horizon):
                _, reward, _, info = self._wip_envs.step(_samples[:, j])
                reward_per_step[
                    i * self.num_wip_envs : (i + 1) * self.num_wip_envs, j
                ] = reward[:_sample_size]

        return reward_per_step

    def plan(self, kf_1: MSKeyFrame, kf_2: MSKeyFrame, duration: MSDuration):
        """Plan a trajectory from keyframe to keyframe.

        Args:
            kf_1 (MSKeyFrame): the first keyframe
            kf_2 (MSKeyFrame): the second keyframe
        """

        if not self._wip_envs or not self._eval_env:
            raise RuntimeError("The planner has not been reset yet.")

        allowed_frames = kf_2.frame() - kf_1.frame()

        with logging_redirect_tqdm(loggers=[logger]):
            logger.info("Planning %i frames between keyframes.", allowed_frames)

            assert self.current_actions is None
            self.current_actions = np.zeros(
                shape=(allowed_frames, self.action_space.shape[0])
            )

            current_state = kf_1.serialized_env.state
            for i in trange(allowed_frames):
                self.step = i
                optimized_action = self._optimize(
                    current_state, self.init_mean, self.init_std
                )

                if not self.execute(optimized_action):
                    logger.error("Failed to execute the optimized action.")
                    break

                current_state = self._eval_env.get_state()
                self.current_actions[i] = optimized_action
            else:
                return False

        return True

    def _optimize(self, state: np.ndarray, init_mean: np.ndarray, init_std: np.ndarray):
        """Optimize action given current state.

        Args:
            state (np.ndarray): current state
            init_mean (np.ndarray): initial mean
            init_std (np.ndarray): initial std

        Returns:
            np.ndarray: best action
        """

        history_elites = None

        for _ in range(self.cem_iter):
            ld = init_mean - self.lb
            rd = self.ub - init_mean
            _std = np.minimum(np.abs(np.minimum(ld, rd) / 2), init_std)
            _mean = init_mean[None, ...].repeat(self.population, axis=0)
            _std = _std[None, ...].repeat(self.population, axis=0)
            samples = self._rng.normal(loc=_mean, scale=_std)

            reward_per_step = self._eval(state, samples)
            rewards = reward_per_step.mean(axis=-1)

            # all_infos = samples, rewards, reward_per_step
            # if history_elites is not None and self.use_history:
            #     all_infos = [
            #         np.concatenate([a, b], axis=0)
            #         for a, b in zip(all_infos, history_elites)
            #     ]
            #     samples, rewards, reward_per_step = all_infos

            valid_sign = rewards == rewards
            rewards = rewards[valid_sign]
            reward_per_step = reward_per_step[valid_sign]
            samples = samples[valid_sign]

            elite_idx = np.argsort(-rewards, axis=0)[: self.elite]
            # if self.use_history:
            #     history_elites = [a[elite_idx] for a in all_infos]

            elites = samples[elite_idx]
            elite_mean = elites.mean(axis=0)
            elite_var = elites.var(axis=0)

            _mean = _mean * (1 - self.lr) + elite_mean * self.lr
            _std = (_std**2 * (1 - self.lr) + elite_var * self.lr) ** 0.5

        assert len(elites) > 0
        return elites[0, 0]  # First action of the elite action sequence

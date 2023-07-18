from typing import Callable, Optional, Sequence

import gym
import numpy as np
import sapien.core as sapien

from mani_skill2.editing.keyframe_editor import MSDuration, MSKeyFrame
from mani_skill2.editing.serialization import SerializedEnv
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.logging_utils import logger

from ..base import BasePlanner


class CEMOptimizer:
    """Cross Entropy Method Optimizer

    Args:
        population (int): population size
        elite (int): elite size
        n_iter(int): number of iteration per optimization step
        lr (float): learning rate
        bounds (Sequence): bounds of the action space
        rng (np.random.Generator): random number generator
    """

    def __init__(
        self,
        population: int,
        elite: int,
        eval_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
        n_iter: int = 10,
        lr: float = 1.0,
        bounds: Optional[Sequence] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        logger.info(
            "Initializing CEM Optimizer: num_population=%s, "
            "num_elite=%s, num_iter=%s",
            population,
            elite,
            n_iter,
        )

        self.population = population
        self.elite = elite
        self.eval_fn = eval_fn
        self.n_iter = n_iter
        self.lr = lr
        self._rng = rng if rng is not None else np.random.default_rng()

        self.lb = np.float32(bounds[0]) if bounds[0] else -np.inf
        self.ub = np.float32(bounds[1]) if bounds[1] else np.inf

        self.mean = None
        self.std = None

    def reset(self, actions: np.ndarray):
        """Reset the mean and std of the distribution.

        Args:
            actions (np.ndarray): the actions to be optimized
        """

        self.mean = np.zeros_like(actions)
        self.std = np.ones_like(actions)

    def optimize(self, state: SerializedEnv):
        """Optimize the mean and std of the distribution.

        Args:
            state (SerializedEnv): the current state of the environment.
                The state will be directly passed to the evaluation function.

        Returns:
            np.ndarray: elite action
        """

        x_shape = (self.population,) + self.mean.shape

        for _ in range(self.n_iter):
            samples = self._rng.normal(loc=self.mean, scale=self.std, size=x_shape)

            reward = self.eval_fn(state, samples)
            elite_idx = np.argsort(reward)[-self.elite :]
            elite_samples = samples[elite_idx]

            self.mean = (
                self.mean * (1 - self.lr) + np.mean(elite_samples, axis=0) * self.lr
            )
            self.std = (
                self.std**2 * (1 - self.lr)
                + np.std(elite_samples, axis=0) ** 2 * self.lr
            ) ** 0.5

        return self.mean


class CEM(BasePlanner):
    """Cross Entropy Method Planner"""

    def __init__(
        self,
        population: int,
        elite: int,
        n_iter: int = 10,
        lr: float = 1.0,
        bounds: Optional[Sequence] = None,
        seed: Optional[int] = None,
        engine: Optional[sapien.Engine] = None,
    ):
        super().__init__(seed=seed, engine=engine)

        self.population = population
        self.elite = elite
        self.n_iter = n_iter
        self.lr = lr
        self.bounds = bounds

        self.reset()

    def reset(self):
        self._wip_envs = []

    def duplicate_envs(self, senv: SerializedEnv, num_scenes: int) -> Sequence[BaseEnv]:
        """Duplicate sapien.Scene for optimization purposes.

        Args:
            senv (SerializedEnv): the serialized environment
            num_scenes (int): the number of scenes to be duplicated

        Returns:
            Sequence[BaseEnv]: the duplicated scenes
        """

        self._wip_envs: list[BaseEnv] = [
            gym.make(
                senv.env_id, obs_mode="none", control_mode=senv.control_mode
            ).unwrapped
            for _ in range(num_scenes)
        ]

        for env in self._wip_envs:
            env.reset()
            senv.dump_state_into(env._scene)

    def eval(self, senv: SerializedEnv, samples: np.ndarray):
        """Evaluate the samples and return the reward of the samples.

        Args:
            senv (SerializedEnv): the serialized environment
            samples (np.ndarray): the samples to be evaluated

        Returns:
            np.ndarray: the reward of the samples
        """

        rewards = np.zeros(samples.shape[0])
        for i, env in enumerate(self._wip_envs):
            senv.dump_state_into(env._scene)
            _, rewards[i], _, _ = env.step(samples[i])
            senv.dump_state_into(env._scene)  # reset the env back to its starting state

        return rewards

    def execute(self, state: SerializedEnv, action: np.ndarray) -> SerializedEnv:
        """Execute an action on the duplicated scenes.

        Args:
            action (np.ndarray): the action to be executed
        """

        if not self._wip_envs:
            raise RuntimeError(
                "CEM planner requires duplicated scenes for optimization."
            )

        state.dump_state_into(self._wip_envs[0]._scene)
        self._wip_envs[0].step(action)
        state.update_state_from(self._wip_envs[0]._scene)

        return state

    def plan(
        self, kf_1: MSKeyFrame, kf_2: MSKeyFrame, duration: MSDuration
    ) -> SerializedEnv:
        """Plan a trajectory from keyframe to keyframe.

        Args:
            kf_1 (MSKeyFrame): the first keyframe
            kf_2 (MSKeyFrame): the second keyframe
        """

        if not self._wip_envs:
            raise RuntimeError(
                "CEM planner requires duplicated scenes for optimization."
            )

        frame_1, frame_2 = kf_1.frame(), kf_2.frame()
        allowable_frames = frame_2 - frame_1

        action_space = kf_1.serialized_env.action_space
        if action_space != kf_2.serialized_env.action_space:
            raise ValueError("The action space of the two keyframes are not the same.")

        optimizer = CEMOptimizer(
            population=self.population,
            elite=self.elite,
            eval_fn=self.eval,
            n_iter=self.n_iter,
            lr=self.lr,
            bounds=self.bounds,
            rng=self._rng,
        )

        state = kf_1.serialized_env
        for i in allowable_frames:
            optimal_action = optimizer.optimize(state)
            state = self.execute(state, optimal_action)

        return state

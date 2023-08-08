import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import sapien.core as sapien

from mani_skill2.editing.keyframe_editor import MSDuration, MSKeyFrame
from mani_skill2.editing.serialization import SerializedEnv
from mani_skill2.utils.logging_utils import logger
from mani_skill2.utils.wrappers import RecordEpisode


class BasePlanner:
    """Base class for all planners.

    Args:
        senv (SerializedEnv): the sample environment
        engine (sapien.Engine): the simulation engine, if None, a new engine will be created.
        seed (int): the random seed. If None, no seed will be set.
        record_dir (str): the directory to save the recordings. If None, no recordings will be saved.
    """

    def __init__(
        self,
        senv: SerializedEnv,
        engine: Optional[sapien.Engine] = None,
        seed: Optional[int] = None,
        record_dir: Union[str, os.PathLike, None] = None,
    ):
        self._engine = engine if engine is not None else sapien.Engine()
        self._rng = np.random.default_rng(seed)
        self.record_dir = record_dir

        self.reset(senv)

    def plan(self, kf_1: MSKeyFrame, kf_2: MSKeyFrame, duration: MSDuration, **kwargs):
        raise NotImplementedError

    def reset(self, senv: SerializedEnv):
        """Reset the planner based on a sample environment.

        Args:
            senv (SerializedEnv): the sample environment
        """

        self.closed = False

        self._seed = int(self._rng.integers(0, 2**32 - 1))
        self._eval_env = gym.make(
            senv.env_id,
            obs_mode=senv.obs_mode,
            control_mode=senv.control_mode,
            render_mode="rgb_array",
        )
        if self.record_dir is not None:
            self._eval_env = RecordEpisode(
                self._eval_env, self.record_dir, save_on_reset=False
            )
        self._eval_env.reset(seed=self._seed)
        self._eval_env.set_state(senv.state)
        self.cumulative_eval_reward = 0

        assert isinstance(senv.action_space, gym.Space)
        self.action_space = senv.action_space

        logger.info(
            "Resetted %s planner with evaluation environment seeded at %i",
            self.__class__.__name__,
            self._seed,
        )

    def close(self):
        """Close and clean-up the planner."""

        self._eval_env.flush_trajectory()
        self._eval_env.flush_video()
        self._eval_env.close()
        self.closed = True

    def execute(self, action: np.ndarray):
        """Execute the actions in the environment.

        Args:
            actions (np.ndarray): the action to be executed in the evaluation environment.
                It has the shape (action_dim, ).

        Returns:
            bool: whether the execution is successful
        """

        assert self._eval_env
        _, reward, terminated, truncated, info = self._eval_env.step(action)

        self.cumulative_eval_reward += reward
        logger.info(
            "Step %i: step_reward=%f, cum_reward=%f, terminated=%s, truncated=%s, env_info=%s",
            self.step,
            reward,
            self.cumulative_eval_reward,
            terminated,
            truncated,
            info,
        )

        return True

import pickle

import sapien.core as sapien
from sapien.utils.viewer.serialization import SerializedScene as _SerializedScene

from mani_skill2.envs.sapien_env import BaseEnv


class SerializedEnv(_SerializedScene):
    """A ManiSkill2-compatible extension of SAPIEN's SerializedScene."""

    def __init__(self, env: BaseEnv):
        self._sim_freq = env.sim_freq
        self._control_freq = env.control_freq
        self._agent_cfg = env._agent_cfg
        self._camera_cfg = env._camera_cfg
        self._render_camera_cfg = env._render_camera_cfg

        self._obs = env.get_obs()
        self._env_info = env.get_info()
        self._env_reward = env.get_reward()

        super().__init__(env._scene)

    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def agent_cfg(self):
        return self._agent_cfg

    @property
    def camera_cfg(self):
        return self._camera_cfg

    @property
    def render_camera_cfg(self):
        return self._render_camera_cfg

    @property
    def obs(self):
        return self._obs

    @property
    def env_info(self):
        return self._env_info

    @property
    def env_reward(self):
        return self._env_reward

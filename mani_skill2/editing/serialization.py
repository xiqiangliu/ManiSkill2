import sapien.core as sapien
from sapien.utils.viewer.serialization import SerializedScene as _SerializedScene

from mani_skill2.envs.sapien_env import BaseEnv


class SerializedEnv(_SerializedScene):
    """A ManiSkill2-compatible extension of SAPIEN's SerializedScene."""

    def __init__(self, env: BaseEnv):
        self._sim_freq = env.sim_freq
        self._control_freq = env.control_freq
        self._agent_cfg = env._agent_cfg
        self._camera_cfgs = env._camera_cfgs
        self._render_camera_cfgs = env._render_camera_cfgs

        self._obs = env.get_obs()
        self._extra_info = env.get_info(obs=self.obs)

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
        return self._camera_cfgs

    @property
    def render_camera_cfg(self):
        return self._render_camera_cfgs

    @property
    def obs(self):
        return self._obs

    @property
    def extra_info(self):
        return self._extra_info

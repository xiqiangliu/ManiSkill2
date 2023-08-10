import numpy as np
import sapien.core as sapien
from sapien.utils.viewer.serialization import SerializedScene as _SerializedScene

from mani_skill2.envs.sapien_env import BaseEnv


class SerializedEnv(_SerializedScene):
    """A ManiSkill2-compatible extension of SAPIEN's SerializedScene."""

    def __init__(self, env: BaseEnv):
        self._env_id: str = env.spec.id
        self._sim_freq: float = env.sim_freq
        self._control_freq: float = env.control_freq
        # self._action_space: Dict = env.agent.action_space  # NOTE: this not being used!
        self._control_mode: str = env.control_mode
        self._obs_mode: str = env.obs_mode
        self._state: np.ndarray = env.get_state()
        self._obs = env.get_obs()
        self._extra_info = env.get_info(obs=self.obs)

        super().__init__(env.unwrapped._scene)

    @property
    def env_id(self):
        return self._env_id

    @property
    def control_mode(self):
        return self._control_mode

    # @property
    # def action_space(self):
    #     return self._action_space

    @property
    def obs_mode(self):
        return self._obs_mode

    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def obs(self):
        return self._obs

    @property
    def extra_info(self):
        return self._extra_info

    @property
    def state(self):
        return self._state

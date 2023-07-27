import gym

import mani_skill2.envs
from mani_skill2.editing.serialization import SerializedEnv
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.editing.planner.mpc import CEM


class KF:
    def __init__(self, senv, frame: int = 0) -> None:
        self.serialized_env = senv
        self._frame = frame

    def frame(self) -> int:
        return self._frame


# For debug purposes
if __name__ == "__main__":
    env: BaseEnv = gym.make(
        "LiftCube-v0", obs_mode="none", control_mode="pd_ee_delta_pos"
    )
    env.reset()
    senv = SerializedEnv(env)
    env.close()

    planner = CEM(
        population=200,
        elite=20,
        sample_env=senv,
        horizon=30,
        cem_iter=4,
        num_wip_envs=8,
        seed=1234,
        lr=0.9,
        record_dir=f"recordings/{senv.env_id}",
        # engine=env.unwrapped._engine,
    )

    kf_1 = KF(senv, 0)
    kf_2 = KF(senv, 100)

    try:
        planner.plan(kf_1, kf_2, None)
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        planner.close()

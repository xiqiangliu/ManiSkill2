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
        "PickCube-v0", obs_mode="none", control_mode="pd_joint_delta_pos"
    )
    env.reset()
    senv = SerializedEnv(env)
    env.close()

    planner = CEM(
        population=200,
        elite=20,
        sample_env=senv,
        horizon=20,
        cem_iter=5,
        num_wip_envs=10,
        seed=1234,
        lr=0.9,
        record_dir=f"recordings/{senv.env_id}",
        # engine=env.unwrapped._engine,
    )

    kf_1 = KF(senv, 0)
    kf_2 = KF(senv, 200)

    try:
        planner.plan(kf_1, kf_2, None)
    except KeyboardInterrupt:
        planner.close()
    finally:
        planner.close()

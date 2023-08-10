import argparse
from pathlib import Path

import gymnasium as gym

import mani_skill2.envs
from mani_skill2.editing.planner.mpc import CEM
from mani_skill2.editing.serialization import SerializedEnv
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.editing.keyframe_editor import MSDuration


class KF:
    def __init__(self, senv, frame: int = 0) -> None:
        self.serialized_env = senv
        self._frame = frame

    def frame(self) -> int:
        return self._frame


# For debug purposes
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-r", "--record-dir", type=Path, default=Path("recordings"))

    args = parser.parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    env.reset()
    senv = SerializedEnv(env)
    env.close()

    planner = CEM(
        population=200,
        elite=20,
        horizon=20,
        cem_iter=5,
        num_wip_envs=10,
        seed=1234,
        lr=0.9,
        record_dir=args.record_dir / args.env_id,
    )

    duration = MSDuration(KF(senv, 0), KF(senv, 200))
    planner.reset(senv)

    try:
        planner.plan(duration)
    finally:
        planner.close()

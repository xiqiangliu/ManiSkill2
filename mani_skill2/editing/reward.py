SUPPORTED_REWARD_TEMPLATES = {
    "custom": "\n".join(
        [
            "import numpy as np",
            "import sapien.core as sapien\n",
            "from mani_skill2.envs.sapien_env import BaseEnv\n",
            "class Reward:",
            "    def __init__(self, env: BaseEnv, scene: sapien.Scene):",
            "        self.env = env",
            "        self.scene = scene\n",
            "    def compute(self):",
            "        return 0",
        ]
    ),
    "l2_distance": "\n".join(
        [
            "import numpy as np",
            "import sapien.core as sapien\n",
            "from mani_skill2.envs.sapien_env import BaseEnv\n",
            "class Reward:",
            "    def __init__(self, env: BaseEnv, scene: sapien.Scene):",
            "        self.env = env",
            "        self.scene = scene",
            '        self.target_actor = self.scene.find_actor_by_id("ID_OF_TARGET_ACTOR")\n',
            "    def compute(self):",
            "        return np.linalg.norm(self.env.tcp.pose.p - self.target_actor.pose.p)",
        ]
    ),
}

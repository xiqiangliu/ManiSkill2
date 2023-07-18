from typing import Optional

import numpy as np
import sapien.core as sapien

from mani_skill2.editing.keyframe_editor import MSDuration, MSKeyFrame


class BasePlanner:
    """Base class for all planners.

    Args:
        engine (sapien.Engine): the simulation engine, if None, a new engine will be created.
    """

    def __init__(
        self, engine: Optional[sapien.Engine] = None, seed: Optional[int] = None
    ):
        self._engine = engine if engine is not None else sapien.Engine()
        self._rng = np.random.default_rng(seed)

    def plan(self, kf_1: MSKeyFrame, kf_2: MSKeyFrame, duration: MSDuration, **kwargs):
        raise NotImplementedError

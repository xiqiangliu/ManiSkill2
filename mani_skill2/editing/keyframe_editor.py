import os
import shutil
import tempfile
from collections.abc import Iterable

import sapien.core as sapien
from sapien.core import renderer as R
from sapien.utils.viewer.plugin import Plugin

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.logging_utils import logger

from .serialization import SerializedEnv


class MSKeyFrame(R.UIKeyframe):
    """ManiSkill2-compatible KeyFrame Implementation"""

    def __init__(self, serialized_env: SerializedEnv, frame: int = 0):
        super().__init__()
        self.serialized_env = serialized_env
        self._frame = frame

    def frame(self):
        return self._frame

    def set_frame(self, frame: int):
        self._frame = frame

    def __repr__(self):
        return f"Keyframe(scene, frame={self.frame()})"


class MSDuration(R.UIDuration):
    """ManiSkill2-compatible Duration Implementation"""

    DEFAULT_DEFINITION = "\n".join(
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
    )

    def __init__(
        self,
        keyframe0: MSKeyFrame,
        keyframe1: MSKeyFrame,
        name: str = "",
        definition: str = DEFAULT_DEFINITION,
    ):
        super().__init__()
        self._keyframe0 = keyframe0
        self._keyframe1 = keyframe1
        self._name = name
        self.definition = definition

    def keyframe0(self):
        return self._keyframe0

    def keyframe1(self):
        return self._keyframe1

    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name


def serialize_keyframes(
    keyframes: Iterable[MSKeyFrame], durations: Iterable[MSDuration]
):
    """Serialize MSKeyFrames and MSDurations into a state that can be saved to disk

    Args:
        keyframes: A list of MSKeyFrames
        durations: A list of MSDurations

    Returns:
        state: A state that can be saved to disk
    """

    s_keyframes = [(f.serialized_env, f.frame()) for f in keyframes]
    f2i = dict((f, i) for i, f in enumerate(keyframes))
    s_durations = [
        (f2i[d._keyframe0], f2i[d._keyframe1], d._name, d.definition) for d in durations
    ]

    return s_keyframes, s_durations


def deserialize_keyframes(
    state: Iterable[
        Iterable[Iterable[SerializedEnv, int]], Iterable[Iterable[int, int, str, str]]
    ]
):
    """Deserialize a state into MSKeyFrames and MSDurations

    Args:
        state: A state that was previously serialized by `serialize_keyframes`.

    Returns:
        keyframes: A list of MSKeyFrames
        durations: A list of MSDurations
    """

    s_keyframes, s_durations = state

    keyframes = [MSKeyFrame(*f) for f in s_keyframes]
    durations = [
        MSDuration(keyframes[k1_idx], keyframes[k2_idx], name, definition)
        for k1_idx, k2_idx, name, definition in s_durations
    ]
    return keyframes, durations


class MSKeyframeWindow(Plugin):
    """A keyframe editor plugin in SAPIEN for ManiSkill2"""

    SUPPORTED_PLANNERS = ("CEM",)

    ui_window: R.UIWindow

    popup_duration: R.UIPopup
    show_popup_duration: bool

    popup_no_editor: R.UIPopup
    show_popup_no_editor: bool

    popup_planner_cfg: R.UIPopup
    show_popup_planner_cfg: bool

    keyframe_editor: R.UIKeyframeEditor
    key_frame_envs: list[MSKeyFrame]
    edited_duration: MSDuration
    _editor_file_name: str

    def __init__(self):
        self.reset()

    @property
    def scene(self):
        return self.viewer.scene

    @property
    def env(self):
        if self._env is None:
            raise RuntimeError("Attmpting to access env before it is set")
        return self._env

    @env.setter
    def env(self, env: BaseEnv):
        self._env = env

    @property
    def keyframes(self):
        return self.keyframe_editor.get_keyframes()

    def reset(self):
        self.ui_window = None
        self.popup_duration = None
        self.show_popup_duration = False

        self.popup_no_editor = None
        self.show_popup_no_editor = False

        self.popup_planner_cfg = None
        self.show_popup_planner_cfg = False
        self._backend = None
        self._backend_cfg = None

        self.keyframe_editor = None
        self.key_frame_envs = []
        self.edited_duration = None
        self._editor_file_name = None

        self.current_frame = 0
        self.total_frames = 32
        self._env = None

    def close(self):
        self.reset()

    def close_popup_duration(self):
        self.show_popup_duration = False
        if self._editor_file_name:
            try:
                os.remove(self._editor_file_name)
            except FileNotFoundError:
                pass

        self._editor_file_name = None

    def open_popup_duration(self):
        self.show_popup_duration = True
        self.popup_duration.get_children()[1].Value(self.edited_duration.name())
        self.popup_duration.get_children()[2].Value(self.edited_duration.definition)

    def duration_name_change(self, text):
        self.edited_duration.set_name(text.value)

    def duration_definition_change(self, text):
        self.edited_duration.definition = text.value

    def confirm_popup_duration(self, _):
        self.edited_duration.set_name(self.popup_duration.get_children()[1].value)
        self.edited_duration.definition = self.popup_duration.get_children()[2].value
        self.close_popup_duration()

    def cancel_popup_duration(self, _):
        self.close_popup_duration()

    def open_in_editor(self, _):
        if shutil.which("gedit") is None:
            # Cannot have more than one popup at a time
            self.show_popup_duration = False
            self.show_popup_no_editor = True
            return

        if self._editor_file_name is None:
            editor_file = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix=self.edited_duration.name(),
                suffix=".py",
                delete=False,
            )
            editor_file.write(self.edited_duration.definition)
            editor_file.close()
            self._editor_file_name = editor_file.name

        if os.system(f"gedit '{self._editor_file_name}' & disown") != 0:
            # Cannot have more than one popup at a time
            self.show_popup_duration = False
            self.show_popup_no_editor = True

    def close_popup_no_editor(self, _):
        # need to revert popup due to one popup restriction
        self.show_popup_no_editor = False
        self.show_popup_duration = True

    def open_popup_planner_cfg(self, _):
        self.show_popup_planner_cfg = True
        self.update_planner_cfg_display(_)

    def confirm_popup_planner_cfg(self, _):
        self.close_popup_duration()

    def close_popup_planner_cfg(self, _):
        self.show_popup_planner_cfg = False

    def update_planner_cfg_display(self, _):
        self._backend = self.popup_planner_cfg.get_children()[0].get_children()[1].value

        section: R.UISection = self.popup_planner_cfg.get_children()[1]
        section.remove_children()

        # NOTE: Pre-filled with default values
        if self._backend == "CEM":
            from .planner.mpc import CEMConfig

            if not isinstance(self._backend_cfg, CEMConfig):
                self._backend_cfg = CEMConfig()

            section.append(R.UIDisplayText().Text("CEM Configuration"))
            section.append(
                R.UISliderInt()
                .Label("Iterations")
                .Bind(self._backend_cfg, "cem_iter")
                .Min(1)
                .Max(10),
                R.UIInputInt()
                .Label("Population")
                .Bind(self._backend_cfg, "population"),
                R.UIInputInt().Label("Elite").Bind(self._backend_cfg, "elite"),
                R.UIInputInt().Label("Horizon").Bind(self._backend_cfg, "horizon"),
                R.UISliderFloat()
                .Label("Learning Rate")
                .Bind(self._backend_cfg, "lr")
                .Min(0.0)
                .Max(2.0),
                R.UIInputInt()
                .Label("# of Rollout Environments")
                .Bind(self._backend_cfg, "num_wip_envs"),
                R.UIInputInt()
                .Label("Seed (-1 for No Seed)")
                .Bind(self._backend_cfg, "seed_ui"),
            )
        else:
            logger.error("Unsupported planner backend: %s", self._backend_cfg["type"])

    def notify_window_focus_change(self, focused):
        if focused and self.edited_duration and self._editor_file_name:
            with open(self._editor_file_name, "r") as f:
                self.edited_duration.definition = f.read()
                self.popup_duration.get_children()[2].Value(
                    self.edited_duration.definition
                )

    def build(self):
        if self.scene is None:
            self.ui_window = None
            return

        if self.ui_window:
            return

        self.popup_duration = (
            R.UIPopup()
            .Label("Edit Duration")
            .EscCallback(self.close_popup_duration)
            .append(
                R.UISameLine().append(
                    R.UIButton().Label("Confirm").Callback(self.confirm_popup_duration),
                    R.UIButton().Label("Cancel").Callback(self.cancel_popup_duration),
                    R.UIButton().Label("Open in Editor").Callback(self.open_in_editor),
                ),
                R.UIInputText().Label("Name").Callback(self.duration_name_change),
                R.UIInputTextMultiline()
                .Label("Definition")
                .Callback(self.duration_definition_change),
            )
        )

        self.popup_no_editor = (
            R.UIPopup()
            .Label("Unsupported Operation")
            .append(
                R.UIDisplayText().Text(
                    "Cannot open gedit. Please edit everything in build-in editor instead."
                ),
                R.UIButton().Label("OK").Callback(self.close_popup_no_editor),
            )
        )

        self.popup_planner_cfg = (
            R.UIPopup()
            .Label("Planner Configuration")
            .append(
                R.UISection()
                .Label("Planner Selection")
                .append(
                    R.UIDisplayText().Text(
                        "NOTE: Changing planner type will discard all settings with it."
                    ),
                    R.UIOptions()
                    .Label("Type")
                    .Style("select")
                    .Id("backend")
                    .Items(self.SUPPORTED_PLANNERS)
                    .Callback(self.update_planner_cfg_display),
                ),
                R.UISection().Label("Planner Configuration"),
                R.UISameLine().append(
                    R.UIButton()
                    .Label("Confirm")
                    .Callback(self.confirm_popup_planner_cfg),
                    R.UIButton().Label("Cancel").Callback(self.close_popup_planner_cfg),
                ),
            )
        )

        self.keyframe_editor = (
            R.UIKeyframeEditor(self.viewer.window.get_content_scale())
            .BindCurrentFrame(self, "current_frame")
            .BindTotalFrames(self, "total_frames")
            .AddKeyframeCallback(self.add_keyframe)
            .MoveKeyframeCallback(self.move_keyframe)
            .AddDurationCallback(self.add_duration)
            .DoubleClickKeyframeCallback(self.load_keyframe)
            .DoubleClickDurationCallback(self.edit_duration)
            .append(
                R.UIButton().Label("Export").Callback(self.editor_export),
                R.UIButton().Label("Import").Callback(self.editor_import),
                R.UIButton().Label("Plan").Callback(self.plan_traj),
                R.UIButton()
                .Label("Planner Configuration")
                .Callback(self.open_popup_planner_cfg),
            )
        )

        self.ui_window = (
            R.UIWindow().Label("Key Frame Editor").append(self.keyframe_editor)
        )

    def add_keyframe(self, frame: int):
        frame = MSKeyFrame(SerializedEnv(self.env), frame)
        self.keyframe_editor.add_keyframe(frame)
        print(self.keyframes)

    def add_duration(self, frame0: MSKeyFrame, frame1: MSKeyFrame):
        duration = MSDuration(frame0, frame1, "New Reward")
        self.keyframe_editor.add_duration(duration)

    def move_keyframe(self, frame: MSKeyFrame, time: MSKeyFrame):
        frame.set_frame(time)

    def load_keyframe(self, frame: MSKeyFrame):
        s_env = frame.serialized_env
        s_env.dump_state_into(self.scene)

    def edit_duration(self, duration):
        self.edited_duration = duration
        self.open_popup_duration()

    def get_ui_windows(self):
        self.build()
        windows = []
        if self.ui_window:
            windows.append(self.ui_window)
        if self.show_popup_duration:
            windows.append(self.popup_duration)
        if self.show_popup_no_editor:
            windows.append(self.popup_no_editor)
        if self.show_popup_planner_cfg:
            windows.append(self.popup_planner_cfg)
        return windows

    def get_editor_state(self):
        keyframes = self.keyframe_editor.get_keyframes()
        durations = self.keyframe_editor.get_durations()
        return (
            self.total_frames,
            self.current_frame,
            serialize_keyframes(keyframes, durations),
        )

    def set_editor_state(self, state):
        self.total_frames, self.current_frame, s = state
        keyframes, durations = deserialize_keyframes(s)
        self.keyframe_editor.set_state(keyframes, durations)

    def editor_export(self, _):
        self._state = self.get_editor_state()

    def editor_import(self, _):
        self.set_editor_state(self._state)

    def plan_traj(self, _):
        """Plan a trajectory using the serialized keyframes and durations in the editor"""
        logger.info("Planning trajectory...")

        # avoid circular import
        from ..editing.planner.mpc.cem import CEM

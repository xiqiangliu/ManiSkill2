import os
import tempfile
from collections.abc import Iterable

import sapien.core as sapien
from sapien.core import renderer as R
from sapien.utils.viewer.plugin import Plugin

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.logging_utils import logger

from ..editing.planner.mpc.cem import CEM
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

    ui_window: R.UIWindow
    popup: R.UIPopup
    show_popup: bool
    keyframe_editor: R.UIKeyframeEditor
    key_frame_envs: list[MSKeyFrame]
    edited_duration: MSDuration
    _editor_file_name: str

    def __init__(self):
        self.reset()
        self._env = None

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
        self.popup = None
        self.show_popup = False
        self.keyframe_editor = None
        self.key_frame_envs = []
        self.edited_duration = None
        self._editor_file_name = None
        self.optim_kwargs = None

        self.current_frame = 0
        self.total_frames = 32

    def close(self):
        self.reset()

    def close_popup(self):
        self.show_popup = False
        if self._editor_file_name:
            try:
                os.remove(self._editor_file_name)
            except FileNotFoundError:
                pass

        self._editor_file_name = None

    def open_popup(self):
        self.show_popup = True
        self.popup.get_children()[1].Value(self.edited_duration.name())
        self.popup.get_children()[2].Value(self.edited_duration.definition)
        self.popup.get_children()[3].Value(self.optim_kwargs)

    def duration_name_change(self, text):
        self.edited_duration.set_name(text.value)

    def duration_definition_change(self, text):
        self.edited_duration.definition = text.value

    def optim_kwargs_change(self, text):
        self.optim_kwargs = text.value

    def confirm_popup(self, _):
        self.edited_duration.set_name(self.popup.get_children()[1].value)
        self.edited_duration.definition = self.popup.get_children()[2].value
        self.optim_kwargs = self.popup.get_children()[3].value
        self.close_popup()

    def cancel_popup(self, _):
        self.close_popup()

    def open_in_editor(self, _):
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

        if os.uname().sysname == "Linux":
            os.system("gedit '{}' & disown".format(self._editor_file_name))
        elif os.uname().sysname == "Windows":
            print(f"Please open and edit the file {self._editor_file_name} manually")

    def notify_window_focus_change(self, focused):
        if focused and self.edited_duration and self._editor_file_name:
            with open(self._editor_file_name, "r") as f:
                self.edited_duration.definition = f.read()
                self.popup.get_children()[2].Value(self.edited_duration.definition)

    def build(self):
        if self.scene is None:
            self.ui_window = None
            return

        if self.ui_window:
            return

        self.popup = (
            R.UIPopup()
            .Label("Edit Duration")
            .EscCallback(self.close_popup)
            .append(
                R.UISameLine().append(
                    R.UIButton().Label("Confirm").Callback(self.confirm_popup),
                    R.UIButton().Label("Cancel").Callback(self.cancel_popup),
                    R.UIButton().Label("Open in Editor").Callback(self.open_in_editor),
                ),
                R.UIInputText().Label("Name").Callback(self.duration_name_change),
                R.UIInputTextMultiline()
                .Label("Definition")
                .Callback(self.duration_definition_change),
                R.UIInputTextMultiline()
                .Label("Optimizer Kwargs")
                .Callback(self.optim_kwargs_change),
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
        self.open_popup()

    def get_ui_windows(self):
        self.build()
        windows = []
        if self.ui_window:
            windows.append(self.ui_window)
        if self.show_popup:
            windows.append(self.popup)
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

        _, _, (serialize_keyframes, serialized_durations) = self.get_editor_state()
        self.optim_kwargs["env"] = self.env
        optimizer = CEM(**self.optim_kwargs)

        for kf1_id, kf2_id, name, duration in serialized_durations:
            logger.info("Planning trajectory for duration %s", name)
            kf1: MSKeyFrame = serialize_keyframes[kf1_id]
            kf2: MSKeyFrame = serialize_keyframes[kf2_id]

            optimizer.optimize(kf1, kf2)

"""Data model for keyframe animation: snapshots, keyframes, timeline."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtGui import QQuaternion, QVector3D

from .interpolation import easing, interpolate_snapshot


INTERPOLATION_MODES = ["linear", "ease_in_out", "ease_in", "ease_out", "constant"]


@dataclass
class CameraSnapshot:
    """Gimbal-lock-free snapshot of the full animatable viewport state."""

    position: QVector3D
    target: QVector3D
    up: QVector3D
    scale: float
    perspective: bool
    model_rotation: QQuaternion
    point_size: float = 2.0  # viewport point size

    # ------------------------------------------------------------------
    # Serialization helpers (QVector3D / QQuaternion → plain floats)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "position": [self.position.x(), self.position.y(), self.position.z()],
            "target": [self.target.x(), self.target.y(), self.target.z()],
            "up": [self.up.x(), self.up.y(), self.up.z()],
            "scale": self.scale,
            "perspective": self.perspective,
            "model_rotation": [
                self.model_rotation.scalar(),
                self.model_rotation.x(),
                self.model_rotation.y(),
                self.model_rotation.z(),
            ],
            "point_size": self.point_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraSnapshot":
        p = d["position"]
        t = d["target"]
        u = d["up"]
        r = d["model_rotation"]
        return cls(
            position=QVector3D(p[0], p[1], p[2]),
            target=QVector3D(t[0], t[1], t[2]),
            up=QVector3D(u[0], u[1], u[2]),
            scale=float(d["scale"]),
            perspective=bool(d["perspective"]),
            model_rotation=QQuaternion(r[0], r[1], r[2], r[3]),
            point_size=float(d.get("point_size", 2.0)),
        )


@dataclass
class Keyframe:
    """A single keyframe on the animation timeline."""

    time: float  # seconds
    camera: CameraSnapshot
    data_frame: Optional[int] = None  # None = hold current XYZ frame
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "camera": self.camera.to_dict(),
            "data_frame": self.data_frame,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(
            time=float(d["time"]),
            camera=CameraSnapshot.from_dict(d["camera"]),
            data_frame=d.get("data_frame"),
            label=d.get("label", ""),
        )


class AnimationTimeline:
    """Ordered list of keyframes with per-segment interpolation profiles."""

    def __init__(self) -> None:
        self.keyframes: list[Keyframe] = []
        self.interpolation: list[str] = []  # len == len(keyframes) - 1
        self.fps: int = 24
        self.duration: float = 10.0

    # ------------------------------------------------------------------
    # Keyframe management
    # ------------------------------------------------------------------

    def add_keyframe(self, kf: Keyframe) -> int:
        """Insert keyframe sorted by time; returns its index."""
        times = [k.time for k in self.keyframes]
        idx = bisect.bisect_right(times, kf.time)
        self.keyframes.insert(idx, kf)
        # Insert a segment entry between this keyframe and the next
        if len(self.keyframes) > 1:
            insert_at = max(0, idx - 1)
            self.interpolation.insert(insert_at, "ease_in_out")
        return idx

    def remove_keyframe(self, index: int) -> None:
        if index < 0 or index >= len(self.keyframes):
            return
        self.keyframes.pop(index)
        if self.interpolation:
            seg_idx = min(index, len(self.interpolation) - 1)
            self.interpolation.pop(seg_idx)

    def move_keyframe(self, index: int, new_time: float) -> None:
        """Move a keyframe to new_time, keeping the list sorted."""
        if index < 0 or index >= len(self.keyframes):
            return
        kf = self.keyframes[index]
        kf.time = new_time
        # Re-sort by removing and re-inserting
        self.keyframes.pop(index)
        if self.interpolation and index < len(self.interpolation):
            self.interpolation.pop(index)
        elif self.interpolation and index > 0:
            self.interpolation.pop(index - 1)
        self.add_keyframe(kf)

    def set_interpolation(self, segment_index: int, mode: str) -> None:
        if 0 <= segment_index < len(self.interpolation):
            self.interpolation[segment_index] = mode

    # ------------------------------------------------------------------
    # Interpolation query
    # ------------------------------------------------------------------

    def get_state_at_time(self, t: float) -> tuple[CameraSnapshot, Optional[int]]:
        """Return interpolated (CameraSnapshot, data_frame) at time t."""
        if not self.keyframes:
            raise ValueError("Timeline has no keyframes")

        # Clamp to timeline bounds
        t = max(self.keyframes[0].time, min(t, self.keyframes[-1].time))

        # Find bracketing keyframes
        times = [k.time for k in self.keyframes]
        idx = bisect.bisect_right(times, t)

        if idx == 0:
            kf = self.keyframes[0]
            return kf.camera, kf.data_frame
        if idx >= len(self.keyframes):
            kf = self.keyframes[-1]
            return kf.camera, kf.data_frame

        kf_a = self.keyframes[idx - 1]
        kf_b = self.keyframes[idx]

        span = kf_b.time - kf_a.time
        if span < 1e-9:
            return kf_b.camera, kf_b.data_frame

        u = (t - kf_a.time) / span
        seg_idx = idx - 1
        mode = self.interpolation[seg_idx] if seg_idx < len(self.interpolation) else "linear"

        eu = easing(u, mode)
        snapshot = interpolate_snapshot(kf_a.camera, kf_b.camera, eu)

        # Interpolate data_frame: if both are set, round-lerp; else use first non-None
        if kf_a.data_frame is not None and kf_b.data_frame is not None:
            data_frame = round(kf_a.data_frame + (kf_b.data_frame - kf_a.data_frame) * eu)
        elif kf_a.data_frame is not None:
            data_frame = kf_a.data_frame
        else:
            data_frame = kf_b.data_frame

        return snapshot, data_frame

    def total_frames(self) -> int:
        return max(1, round(self.duration * self.fps))

    def time_at_frame(self, frame_index: int) -> float:
        return frame_index / self.fps

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "duration": self.duration,
            "interpolation": list(self.interpolation),
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnimationTimeline":
        tl = cls()
        tl.fps = int(d.get("fps", 24))
        tl.duration = float(d.get("duration", 10.0))
        tl.interpolation = list(d.get("interpolation", []))
        tl.keyframes = [Keyframe.from_dict(kfd) for kfd in d.get("keyframes", [])]
        return tl

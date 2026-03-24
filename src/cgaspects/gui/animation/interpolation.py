"""Easing functions and camera snapshot interpolation."""

from __future__ import annotations

from PySide6.QtGui import QQuaternion, QVector3D


def easing(t: float, mode: str) -> float:
    """Apply easing to normalised time t in [0, 1]. Returns remapped t."""
    t = max(0.0, min(1.0, t))
    match mode:
        case "linear":
            return t
        case "ease_in_out":
            return 3 * t**2 - 2 * t**3
        case "ease_in":
            return t**2
        case "ease_out":
            return 1 - (1 - t) ** 2
        case "constant":
            return 0.0
        case _:
            return t


def _lerp_vec3(a: QVector3D, b: QVector3D, t: float) -> QVector3D:
    return a + (b - a) * t


def interpolate_snapshot(a, b, t: float):
    """Interpolate between two CameraSnapshots at normalised time t.

    Uses linear interpolation for position, target, and scale;
    nlerp for the up direction; slerp for model_rotation.
    """
    from .keyframe import CameraSnapshot

    position = _lerp_vec3(a.position, b.position, t)
    target = _lerp_vec3(a.target, b.target, t)
    scale = a.scale + (b.scale - a.scale) * t

    # Normalised linear interpolation for up vector (good enough, no gimbal lock)
    up_lerped = _lerp_vec3(a.up, b.up, t)
    length = up_lerped.length()
    up = up_lerped / length if length > 1e-9 else QVector3D(0, 1, 0)

    # Spherical linear interpolation for object rotation
    model_rotation = QQuaternion.slerp(a.model_rotation, b.model_rotation, t)

    # Perspective: use the value of whichever keyframe we are closer to
    perspective = a.perspective if t < 0.5 else b.perspective

    point_size = a.point_size + (b.point_size - a.point_size) * t

    return CameraSnapshot(
        position=position,
        target=target,
        up=up,
        scale=scale,
        perspective=perspective,
        model_rotation=model_rotation,
        point_size=point_size,
    )

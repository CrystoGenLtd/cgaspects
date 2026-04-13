"""Easing functions and camera snapshot interpolation."""

from __future__ import annotations

import dataclasses

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


def _lerp_tuple3(a: tuple, b: tuple, t: float) -> tuple:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _interpolate_planes(planes_a: list, planes_b: list, t: float) -> list:
    """Match planes by index and interpolate origins for slice-enabled planes."""
    n = max(len(planes_a), len(planes_b))
    result = []
    for i in range(n):
        if i >= len(planes_a):
            result.append(planes_b[i])
        elif i >= len(planes_b):
            result.append(planes_a[i])
        else:
            pa, pb = planes_a[i], planes_b[i]
            # Interpolate origin when slicing is active in both keyframes
            if pa.slice_enabled and pb.slice_enabled:
                origin = _lerp_tuple3(pa.origin, pb.origin, t)
            else:
                origin = pa.origin if t < 0.5 else pb.origin
            base = pa if t < 0.5 else pb
            result.append(dataclasses.replace(base, origin=origin))
    return result


def _interpolate_directions(dirs_a: list, dirs_b: list, t: float) -> list:
    """Match directions by index and interpolate origins."""
    n = max(len(dirs_a), len(dirs_b))
    result = []
    for i in range(n):
        if i >= len(dirs_a):
            result.append(dirs_b[i])
        elif i >= len(dirs_b):
            result.append(dirs_a[i])
        else:
            da, db = dirs_a[i], dirs_b[i]
            origin = _lerp_tuple3(da.origin, db.origin, t)
            base = da if t < 0.5 else db
            result.append(dataclasses.replace(base, origin=origin))
    return result


def interpolate_snapshot(a, b, t: float):
    """Interpolate between two CameraSnapshots at normalised time t.

    Camera: lerp position/target/scale, nlerp up, slerp model_rotation.
    Style / color_by / colormap: constant, switch at t=0.5.
    single_color: lerp RGBA.
    Planes: match by index; interpolate origin when slice_enabled in both.
    Directions: match by index; interpolate origin.
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

    # Style-level settings: constant, switch at midpoint
    style = a.style if t < 0.5 else b.style
    color_by = a.color_by if t < 0.5 else b.color_by
    colormap = a.colormap if t < 0.5 else b.colormap

    # Single colour: lerp RGBA
    sc_a, sc_b = a.single_color, b.single_color
    single_color = (
        sc_a[0] + (sc_b[0] - sc_a[0]) * t,
        sc_a[1] + (sc_b[1] - sc_a[1]) * t,
        sc_a[2] + (sc_b[2] - sc_a[2]) * t,
        sc_a[3] + (sc_b[3] - sc_a[3]) * t,
    )

    planes = _interpolate_planes(a.planes, b.planes, t)
    directions = _interpolate_directions(a.directions, b.directions, t)

    return CameraSnapshot(
        position=position,
        target=target,
        up=up,
        scale=scale,
        perspective=perspective,
        model_rotation=model_rotation,
        point_size=point_size,
        style=style,
        color_by=color_by,
        colormap=colormap,
        single_color=single_color,
        planes=planes,
        directions=directions,
    )

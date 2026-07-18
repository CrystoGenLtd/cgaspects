"""Offline ray-tracing export for the 3D viewport (POV-Ray and Tachyon).

The GL viewport draws spheres as GPU impostors — fast, but with no true
shadows, ambient occlusion or reflections. This module snapshots the current
scene (spheres, bonds, camera, light and material) into a backend-neutral
:class:`RTScene`, then writes it as a POV-Ray ``.pov`` or a Tachyon ``.dat``
scene file and (optionally) shells out to the renderer to produce a PNG.

Geometry is baked into world space: every centre is transformed by the widget's
model matrix (camera zoom scale · object rotation) and every radius scaled by
``camera.scale``, so the exported scene matches exactly what the viewport shows.

The default material maps the live :class:`RenderSettings` one-to-one; the
optional :class:`PhotorealOptions` layers on radiosity/ambient-occlusion, soft
shadows, reflection and focal blur for publication-quality stills.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PySide6.QtGui import QMatrix4x4, QVector3D

logger = logging.getLogger("CA:Raytrace")


# --------------------------------------------------------------------------- #
# Backend-neutral scene description
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RTMaterial:
    """Surface finish, mapped from the live GL RenderSettings."""

    ambient: float = 0.30
    diffuse: float = 0.70
    specular: float = 0.0
    shininess: float = 16.0
    specular_tint: float = 0.0   # 0 = white highlight (plastic), 1 = tinted (metal)
    brightness: float = 1.0


@dataclasses.dataclass
class PhotorealOptions:
    """Extra quality knobs applied on top of the base material (Photoreal mode)."""

    ambient_occlusion: bool = True
    ao_samples: int = 32
    soft_shadows: bool = True
    shadow_softness: float = 1.0      # area-light size, world units
    reflection: float = 0.0           # 0 = matte, up to ~0.4 for a wet look
    focal_blur: bool = False
    aperture: float = 0.05            # larger = shallower depth of field
    antialiasing: int = 9             # POV +A quality / Tachyon AA samples


@dataclasses.dataclass
class RTCamera:
    position: tuple
    target: tuple
    up: tuple
    fov_deg: float                    # vertical field of view
    perspective: bool
    ortho_size: float
    aspect: float


@dataclasses.dataclass
class RTLight:
    direction: tuple                  # world-space direction from scene toward light
    color: tuple = (1.0, 1.0, 1.0)


@dataclasses.dataclass
class RTScene:
    spheres: np.ndarray               # (N, 7): x y z r g b radius
    cylinders: np.ndarray             # (M, 10): x0 y0 z0 x1 y1 z1 r g b radius
    camera: RTCamera
    light: RTLight
    background: tuple
    width: int
    height: int
    material: RTMaterial
    photoreal: PhotorealOptions | None = None   # None => "Match GL"

    @property
    def aspect(self) -> float:
        return self.camera.aspect

    @property
    def center(self) -> np.ndarray:
        """Scene centroid, for placing the light and focal point."""
        if len(self.spheres):
            return self.spheres[:, :3].mean(axis=0)
        if len(self.cylinders):
            mids = 0.5 * (self.cylinders[:, 0:3] + self.cylinders[:, 3:6])
            return mids.mean(axis=0)
        return np.array(self.camera.target, dtype=np.float64)

    @property
    def radius_extent(self) -> float:
        """Distance from the centre to the farthest sphere surface."""
        c = self.center
        if len(self.spheres):
            d = np.linalg.norm(self.spheres[:, :3] - c, axis=1) + self.spheres[:, 6]
            return float(d.max())
        return max(float(np.linalg.norm(np.array(self.camera.position) - c)), 1.0)


# --------------------------------------------------------------------------- #
# Scene extraction from the live widget
# --------------------------------------------------------------------------- #
def _reshape(raw, cols):
    if raw is None:
        return np.zeros((0, cols), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1, cols)
    return arr


def build_scene_from_widget(widget, width: int, height: int,
                            photoreal: PhotorealOptions | None = None) -> RTScene:
    """Snapshot the widget's current geometry, camera and material into an RTScene."""
    cam = widget.camera
    model = cam.modelMatrix()               # scale · object-rotation
    scale = float(cam.scale)
    ps = float(widget.point_size)

    def bake_centers(pos_xyz: np.ndarray) -> np.ndarray:
        out = np.empty_like(pos_xyz)
        for i, (x, y, z) in enumerate(pos_xyz):
            v = model.map(QVector3D(float(x), float(y), float(z)))
            out[i] = (v.x(), v.y(), v.z())
        return out

    spheres = np.zeros((0, 7), dtype=np.float32)
    cylinders = np.zeros((0, 10), dtype=np.float32)

    if getattr(widget, "is_atom_view", False) and widget.atom_renderer is not None:
        atoms = _reshape(getattr(widget.atom_renderer, "_raw_points", None), 8)
        if len(atoms):
            centers = bake_centers(atoms[:, :3])
            radii = atoms[:, 7] * (ps / 6.0) * scale
            spheres = np.column_stack([centers, atoms[:, 3:6], radii]).astype(np.float32)
        bonds = _reshape(getattr(widget.bond_renderer, "instances", None), 10) \
            if getattr(widget, "bond_renderer", None) is not None else np.zeros((0, 10), np.float32)
        if len(bonds):
            starts = bake_centers(bonds[:, 0:3])
            ends = bake_centers(bonds[:, 3:6])
            radii = bonds[:, 9] * (ps / 6.0) * scale
            cylinders = np.column_stack(
                [starts, ends, bonds[:, 6:9], radii]).astype(np.float32)

        # Optional extra cylinder set (e.g. the unit cell viewer's energy-scaled
        # interaction tubes) — absent on widgets that don't define it.
        extra_renderer = getattr(widget, "conn_tube_renderer", None)
        if extra_renderer is not None:
            tubes = _reshape(getattr(extra_renderer, "instances", None), 10)
            if len(tubes):
                starts = bake_centers(tubes[:, 0:3])
                ends = bake_centers(tubes[:, 3:6])
                radii = tubes[:, 9] * (ps / 6.0) * scale
                tube_cyl = np.column_stack(
                    [starts, ends, tubes[:, 6:9], radii]).astype(np.float32)
                cylinders = (
                    np.vstack([cylinders, tube_cyl]) if len(cylinders) else tube_cyl
                )
    else:
        raw = getattr(widget.sphere_renderer, "_raw_points", None) \
            if getattr(widget, "sphere_renderer", None) is not None else None
        pts = _reshape(raw, 7)
        if not len(pts) and getattr(widget, "point_cloud_renderer", None) is not None:
            pts = _reshape(getattr(widget.point_cloud_renderer, "points", None), 7)
        if len(pts):
            centers = bake_centers(pts[:, :3])
            radii = np.full(len(pts), ps * 0.2 * scale, dtype=np.float32)
            spheres = np.column_stack([centers, pts[:, 3:6], radii]).astype(np.float32)

    rs = widget.render_settings
    material = RTMaterial(
        ambient=rs.ambient, diffuse=rs.diffuse, specular=rs.specular,
        shininess=rs.shininess, specular_tint=rs.specular_tint,
        brightness=rs.brightness,
    )

    # The GL light is a view-space headlight; rotate it into world space so the
    # highlights land where the viewport shows them.
    view = cam.viewMatrix()
    world_light = view.inverted()[0].mapVector(rs.light_dir())
    world_light.normalize()

    bg = widget.backgroundColor
    camera = RTCamera(
        position=(cam.position.x(), cam.position.y(), cam.position.z()),
        target=(cam.target.x(), cam.target.y(), cam.target.z()),
        up=(cam.up.x(), cam.up.y(), cam.up.z()),
        fov_deg=float(cam.fieldOfView),
        perspective=bool(cam.perspectiveProjection),
        ortho_size=float(cam.orthoSize),
        aspect=(width / height) if height else 1.0,
    )
    return RTScene(
        spheres=spheres,
        cylinders=cylinders,
        camera=camera,
        light=RTLight(direction=(world_light.x(), world_light.y(), world_light.z())),
        background=(bg.redF(), bg.greenF(), bg.blueF()),
        width=int(width),
        height=int(height),
        material=material,
        photoreal=photoreal,
    )


# --------------------------------------------------------------------------- #
# POV-Ray writer
# --------------------------------------------------------------------------- #
def _v(vec) -> str:
    return f"<{vec[0]:.6g}, {vec[1]:.6g}, {vec[2]:.6g}>"


def write_povray(scene: RTScene, path: str | Path) -> Path:
    """Serialise an RTScene to a POV-Ray 3.7 ``.pov`` file. Returns the path."""
    path = Path(path)
    m = scene.material
    pr = scene.photoreal
    half_v = math.tan(math.radians(scene.camera.fov_deg) / 2)
    hfov = math.degrees(2 * math.atan(scene.aspect * half_v))
    center = scene.center
    extent = scene.radius_extent

    lines: list[str] = ["#version 3.7;", ""]

    if pr and pr.ambient_occlusion:
        lines += [
            "global_settings {",
            "  assumed_gamma 1.0",
            "  radiosity {",
            f"    count {max(pr.ao_samples, 20)}",
            "    nearest_count 10  error_bound 0.4  recursion_limit 2",
            "    brightness 1.0",
            "  }",
            "}",
        ]
    else:
        lines += ["global_settings { assumed_gamma 1.0 }"]

    lines += ["", f"background {{ color rgb {_v(scene.background)} }}", ""]

    # Camera — negative "right" gives OpenGL-style right-handedness; POV
    # re-perpendicularises the frame from look_at + sky.
    cam = scene.camera
    lines += ["camera {"]
    if cam.perspective:
        lines += [
            "  perspective",
            f"  location {_v(cam.position)}",
            f"  sky {_v(cam.up)}",
            f"  right <{-scene.aspect:.6g}, 0, 0>",
            f"  look_at {_v(cam.target)}",
            f"  angle {hfov:.6g}",
        ]
    else:
        half_h = cam.ortho_size
        half_w = cam.ortho_size * scene.aspect
        lines += [
            "  orthographic",
            f"  location {_v(cam.position)}",
            f"  sky {_v(cam.up)}",
            f"  right <{-2 * half_w:.6g}, 0, 0>",
            f"  up <0, {2 * half_h:.6g}, 0>",
            f"  look_at {_v(cam.target)}",
        ]
    if pr and pr.focal_blur and cam.perspective:
        focal = np.linalg.norm(np.array(cam.position) - center)
        lines += [
            f"  aperture {pr.aperture * extent:.6g}",
            f"  focal_point {_v(center)}",
            f"  blur_samples {max(pr.antialiasing * 3, 19)}",
            f"  // focal distance ~= {focal:.3g}",
        ]
    lines += ["}", ""]

    # Light — placed far along the world light direction from the scene centre.
    light_pos = center + np.array(scene.light.direction) * extent * 6.0
    if pr and pr.soft_shadows:
        a = pr.shadow_softness * extent * 0.15
        lines += [
            "light_source {",
            f"  {_v(light_pos)} color rgb {_v([m.brightness]*3)}",
            f"  area_light <{a:.4g},0,0>, <0,{a:.4g},0>, 5, 5 adaptive 1 jitter",
            "}",
            "",
        ]
    else:
        lines += [
            "light_source {",
            f"  {_v(light_pos)} color rgb {_v([m.brightness]*3)}",
            "}",
            "",
        ]
    # A soft fill so shadowed sides aren't pure black (mirrors the GL ambient).
    fill_pos = center - np.array(scene.light.direction) * extent * 6.0
    lines += [
        "light_source {",
        f"  {_v(fill_pos)} color rgb {_v([m.ambient * m.brightness]*3)} shadowless",
        "}",
        "",
    ]

    finish = (f"finish {{ ambient {m.ambient:.4g} diffuse {m.diffuse:.4g}"
              f" phong {m.specular:.4g} phong_size {max(m.shininess, 1):.4g}")
    if m.specular_tint > 0:
        # Tints the specular highlight toward the surface colour (metal look).
        finish += f" metallic {m.specular_tint:.4g}"
    if pr and pr.reflection > 0:
        refl = f"reflection {{ {pr.reflection:.4g}"
        if m.specular_tint > 0:
            refl += " metallic"
        refl += " }"
        finish += " " + refl
    finish += " }"

    # Geometry. Group into unions so POV parses large scenes efficiently.
    if len(scene.spheres):
        lines.append("union {")
        for x, y, z, r, g, b, rad in scene.spheres:
            lines.append(
                f"  sphere {{ <{x:.5g},{y:.5g},{z:.5g}>, {rad:.5g}"
                f" pigment {{ rgb <{r:.4g},{g:.4g},{b:.4g}> }} {finish} }}")
        lines += ["}", ""]

    if len(scene.cylinders):
        lines.append("union {")
        for x0, y0, z0, x1, y1, z1, r, g, b, rad in scene.cylinders:
            if abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6 and abs(z0 - z1) < 1e-6:
                continue
            lines.append(
                f"  cylinder {{ <{x0:.5g},{y0:.5g},{z0:.5g}>,"
                f" <{x1:.5g},{y1:.5g},{z1:.5g}>, {rad:.5g}"
                f" pigment {{ rgb <{r:.4g},{g:.4g},{b:.4g}> }} {finish} }}")
        lines += ["}", ""]

    path.write_text("\n".join(lines))
    logger.info("Wrote POV-Ray scene: %s (%d spheres, %d cylinders)",
                path, len(scene.spheres), len(scene.cylinders))
    return path


# --------------------------------------------------------------------------- #
# Tachyon writer
# --------------------------------------------------------------------------- #
def _tex(m: RTMaterial, pr: PhotorealOptions | None, rgb) -> list[str]:
    r, g, b = rgb
    refl = pr.reflection if pr else 0.0
    # A tinted highlight reads as metal; Tachyon exposes this as the phong type.
    phong_type = "Metal" if m.specular_tint > 0.5 else "Plastic"
    return [
        f"  Texture Ambient {m.ambient:.4g} Diffuse {m.diffuse:.4g}"
        f" Specular {refl:.4g} Opacity 1",
        f"    Phong {phong_type} {m.specular:.4g} Phong_size {max(m.shininess, 1):.4g}"
        f" Color {r:.4g} {g:.4g} {b:.4g} TexFunc 0",
    ]


def write_tachyon(scene: RTScene, path: str | Path) -> Path:
    """Serialise an RTScene to a Tachyon ``.dat`` scene file. Returns the path."""
    path = Path(path)
    m = scene.material
    pr = scene.photoreal
    cam = scene.camera
    center = scene.center
    extent = scene.radius_extent

    viewdir = np.array(cam.target) - np.array(cam.position)
    n = np.linalg.norm(viewdir)
    viewdir = viewdir / n if n else np.array([0, 0, -1.0])
    # Tachyon Zoom sets the field of view: half-height = 1/zoom at unit distance.
    zoom = 1.0 / max(math.tan(math.radians(cam.fov_deg) / 2), 1e-4)

    lines = [
        "Begin_Scene",
        f"Resolution {scene.width} {scene.height}",
        "Shader_Mode Medium",
        "  Trans_VMD",
        "  Fog_VMD",
    ]
    if pr and pr.ambient_occlusion:
        lines += [
            "  Ambient_Occlusion",
            f"    Ambient_Color {m.ambient:.4g} {m.ambient:.4g} {m.ambient:.4g}",
            f"    Rescale_Direct 1.0 Samples {pr.ao_samples}",
        ]
    lines += [
        "End_Shader_Mode",
        "Camera",
        "  Projection Perspective" if cam.perspective else "  Projection Orthographic",
        f"  Zoom {zoom:.6g}",
        f"  Aspectratio {scene.aspect:.6g}",
        f"  Antialiasing {pr.antialiasing if pr else 4}",
        "  Raydepth 8",
        f"  Center {cam.position[0]:.6g} {cam.position[1]:.6g} {cam.position[2]:.6g}",
        f"  Viewdir {viewdir[0]:.6g} {viewdir[1]:.6g} {viewdir[2]:.6g}",
        f"  Updir {cam.up[0]:.6g} {cam.up[1]:.6g} {cam.up[2]:.6g}",
        "End_Camera",
    ]

    light_pos = center + np.array(scene.light.direction) * extent * 6.0
    lb = m.brightness
    lines += [
        f"Directional_Light Direction {-scene.light.direction[0]:.6g}"
        f" {-scene.light.direction[1]:.6g} {-scene.light.direction[2]:.6g}"
        f" Color {lb:.4g} {lb:.4g} {lb:.4g}",
        f"Light Center {light_pos[0]:.6g} {light_pos[1]:.6g} {light_pos[2]:.6g}"
        f" Rad {max(extent * 0.02, 0.1):.4g} Color {lb:.4g} {lb:.4g} {lb:.4g}",
        f"Background {scene.background[0]:.4g} {scene.background[1]:.4g}"
        f" {scene.background[2]:.4g}",
    ]

    for x, y, z, r, g, b, rad in scene.spheres:
        lines.append(f"Sphere Center {x:.5g} {y:.5g} {z:.5g} Rad {rad:.5g}")
        lines += _tex(m, pr, (r, g, b))
    for x0, y0, z0, x1, y1, z1, r, g, b, rad in scene.cylinders:
        if abs(x0 - x1) < 1e-6 and abs(y0 - y1) < 1e-6 and abs(z0 - z1) < 1e-6:
            continue
        lines.append(
            f"FCylinder Base {x0:.5g} {y0:.5g} {z0:.5g}"
            f" Apex {x1:.5g} {y1:.5g} {z1:.5g} Rad {rad:.5g}")
        lines += _tex(m, pr, (r, g, b))

    lines.append("End_Scene")
    path.write_text("\n".join(lines))
    logger.info("Wrote Tachyon scene: %s (%d spheres, %d cylinders)",
                path, len(scene.spheres), len(scene.cylinders))
    return path


# --------------------------------------------------------------------------- #
# Renderer discovery + invocation
# --------------------------------------------------------------------------- #
# name -> list of candidate executables on PATH
_BACKENDS = {
    "povray": ["povray", "megapov"],
    "tachyon": ["tachyon"],
}

SCENE_SUFFIX = {"povray": ".pov", "tachyon": ".dat"}


def find_renderer(backend: str) -> str | None:
    """Absolute path to the backend executable, or None if not on PATH."""
    for exe in _BACKENDS.get(backend, []):
        found = shutil.which(exe)
        if found:
            return found
    return None


def write_scene(scene: RTScene, backend: str, path: str | Path) -> Path:
    if backend == "povray":
        return write_povray(scene, path)
    if backend == "tachyon":
        return write_tachyon(scene, path)
    raise ValueError(f"Unknown ray-trace backend: {backend!r}")


def render_scene_file(backend: str, scene_path: str | Path, image_path: str | Path,
                      width: int, height: int, *, antialiasing: int = 4,
                      timeout: float = 600.0,
                      extra_args: list[str] | None = None) -> Path:
    """Run an already-written scene file through the backend to produce a PNG.

    ``extra_args`` are appended verbatim to the renderer command line (e.g.
    ``["+WT4"]`` to cap POV-Ray's thread count when rendering frames in
    parallel). Raises RuntimeError if the backend is missing or exits non-zero.
    Used both by :func:`render_scene` and by the headless batch CLI
    (``cgaspects-render``).
    """
    exe = find_renderer(backend)
    if exe is None:
        raise RuntimeError(
            f"{backend} was not found on PATH. Install it, or use 'Export scene "
            f"file only' and render elsewhere.")
    scene_path = Path(scene_path)
    image_path = Path(image_path)

    if backend == "povray":
        cmd = [exe, f"+I{scene_path}", f"+O{image_path}",
               f"+W{width}", f"+H{height}",
               "+FN", "+A", f"+Q{min(max(antialiasing, 1), 11)}", "-D", "-P"]
    elif backend == "tachyon":
        cmd = [exe, str(scene_path), "-o", str(image_path),
               "-res", str(width), str(height), "-format", "PNG"]
    else:
        raise ValueError(f"Unknown ray-trace backend: {backend!r}")
    if extra_args:
        cmd += list(extra_args)

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    if not image_path.exists():
        raise RuntimeError(f"{backend} reported success but produced no image.")
    return image_path


def render_scene(scene: RTScene, backend: str, scene_path: str | Path,
                 image_path: str | Path, timeout: float = 600.0) -> Path:
    """Write the scene file and run the backend to produce ``image_path`` (PNG).

    Raises RuntimeError if the backend is missing or exits non-zero.
    """
    scene_path = write_scene(scene, backend, scene_path)
    aa = scene.photoreal.antialiasing if scene.photoreal else 4
    return render_scene_file(backend, scene_path, image_path,
                             scene.width, scene.height,
                             antialiasing=aa, timeout=timeout)

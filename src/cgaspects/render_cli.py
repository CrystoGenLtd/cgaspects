"""Headless batch renderer: XYZ movie + animation.json → ray-traced frames.

Rebuilds each animation frame's scene directly from the saved timeline
(`animation.json`) and the CrystoGen ``.XYZ`` file — no GUI, no OpenGL, no
display. Geometry, colours, slicing and camera baking replicate the viewport's
centroid (sphere) path, so the offline frames match the in-app preview.

Typical use on a cluster::

    # inspect the timeline first
    cgaspects-render movie.XYZ animation.json --info

    # render everything on one node, 8 frames at a time
    cgaspects-render movie.XYZ animation.json -o frames/ --jobs 8

    # SLURM array: each task renders one frame
    cgaspects-render movie.XYZ animation.json -o frames/ \\
        --frames $SLURM_ARRAY_TASK_ID

    # only write .pov scene files, render elsewhere
    cgaspects-render movie.XYZ animation.json -o frames/ --export-only

    # stitch to video (needs ffmpeg on PATH)
    cgaspects-render movie.XYZ animation.json -o frames/ --video out.mp4

Notes
-----
- POV-Ray and Tachyon are CPU ray tracers; parallelism comes from rendering
  many frames concurrently (``--jobs``), not from GPUs.
- Only the centroid/sphere representation is rendered. Plane quads and
  direction arrows are not drawn, but slice planes (point filtering) from the
  animation's keyframes are applied, including animated slice origins.
- Atom view, convex hull and legend filters are not supported headlessly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logger = logging.getLogger("CGA:RenderCLI")

# Mirrors OpenGLWidget.columnLabelToIndex for the centroid path.
_COLUMN_INDEX = {
    "Atom/Molecule Type": 0,
    "Atom/Molecule Number": 1,
    "Layer": 2,
    "Single Colour": -1,
    "Site Number": 6,
    "Particle Energy": 7,
}

# Mirrors OpenGLWidget.availableColormaps (GUI name → matplotlib name).
_COLORMAPS = {
    "Viridis": "viridis",
    "Cividis": "cividis",
    "Plasma": "plasma",
    "Inferno": "inferno",
    "Magma": "magma",
    "Twilight": "twilight",
    "HSV": "hsv",
}


def _cmap(name: str):
    from matplotlib import colormaps
    return colormaps[_COLORMAPS.get(name, "viridis")]


# --------------------------------------------------------------------------- #
# Snapshot → scene reconstruction (Qt-free numpy re-implementation of the
# widget's centroid path: updatePointCloudVertices + build_scene_from_widget)
# --------------------------------------------------------------------------- #
def _quat_matrix(q) -> np.ndarray:
    """3×3 rotation matrix from a QQuaternion (matches QMatrix4x4.rotate)."""
    w, x, y, z = q.scalar(), q.x(), q.y(), q.z()
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _compute_colors(vd, snapshot) -> np.ndarray:
    """(N, 3) float32 colours replicating the viewport's centroid colouring."""
    col_idx = _COLUMN_INDEX.get(snapshot.color_by, 2)
    if col_idx < -1:
        # Atom-view-only modes (Atom, Coordination Shell, …) fall back to Layer,
        # exactly like the widget's guard.
        col_idx = 2

    if col_idx == -1:
        rgb = np.asarray(snapshot.single_color[:3], dtype=np.float32)
        return vd.colors_uniform(rgb)

    n = vd.n_centroids
    min_val = max_val = None
    if col_idx == 0:
        values = vd.mol_types.astype(np.float32)
    elif col_idx == 1:
        values = (vd.mol_numbers.astype(np.float32)
                  if vd.mol_numbers is not None
                  else np.arange(n, dtype=np.float32))
    elif col_idx == 2:
        if vd.layers is not None:
            values = vd.layers.astype(np.float32)
            valid = values[values < 99]
            min_val = 1.0
            max_val = float(int(np.nanmax(valid))) if valid.size else 1.0
        else:
            values = np.zeros(n, dtype=np.float32)
            min_val = max_val = 0.0
    elif col_idx == 6:
        values = (vd.site_numbers.astype(np.float32)
                  if vd.site_numbers is not None
                  else np.zeros(n, dtype=np.float32))
    elif col_idx == 7:
        values = (vd.energies.astype(np.float32)
                  if vd.energies is not None
                  else np.zeros(n, dtype=np.float32))
    else:
        values = np.arange(n, dtype=np.float32)

    if min_val is None:
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
    return vd.colors_by_array(values, _cmap(snapshot.colormap), min_val, max_val)


def _slice_mask(centroids: np.ndarray, planes: list) -> np.ndarray | None:
    """Keep-mask replicating _active_slice_planes + _slice_centroid_mask.

    Fractional (Miller-index) plane normals need crystallography to resolve;
    headlessly they are used as-is with a warning, matching the widget when no
    crystallography is loaded.
    """
    active = []
    for plane in planes:
        if not plane.slice_enabled:
            continue
        normal = np.array(plane.normal, dtype=np.float64)
        if plane.fractional:
            logger.warning(
                "Slice plane has fractional (hkl) normal %s — no crystallography "
                "available headlessly, treating it as a Cartesian normal.",
                tuple(plane.normal))
        n_len = np.linalg.norm(normal)
        if n_len < 1e-9:
            continue
        active.append((normal / n_len, np.array(plane.origin, dtype=np.float64),
                       plane.slice_two_sided, plane.slice_thickness))
    if not active:
        return None
    pts = centroids.astype(np.float64)
    mask = np.ones(len(pts), dtype=bool)
    for normal, origin, two_sided, thickness in active:
        d = (pts - origin) @ normal
        if two_sided:
            mask &= np.abs(d) <= thickness / 2.0
        else:
            mask &= d >= -thickness
    return mask


def _world_light(snapshot, render_settings) -> tuple:
    """View-space headlight → world space (mirrors build_scene_from_widget)."""
    pos = np.array([snapshot.position.x(), snapshot.position.y(), snapshot.position.z()])
    tgt = np.array([snapshot.target.x(), snapshot.target.y(), snapshot.target.z()])
    up = np.array([snapshot.up.x(), snapshot.up.y(), snapshot.up.z()])

    f = tgt - pos
    f = f / max(np.linalg.norm(f), 1e-12)
    s = np.cross(f, up / max(np.linalg.norm(up), 1e-12))
    s = s / max(np.linalg.norm(s), 1e-12)
    u = np.cross(s, f)

    ld = render_settings.light_dir()
    world = s * ld.x() + u * ld.y() - f * ld.z()
    world = world / max(np.linalg.norm(world), 1e-12)
    return tuple(world)


def build_scene_from_snapshot(
    xyz: np.ndarray,
    snapshot,
    width: int,
    height: int,
    *,
    render_settings=None,
    photoreal=None,
    background: tuple = (1.0, 1.0, 1.0),
    fov_deg: float = 45.0,
    ortho_size: float = 10.0,
):
    """Build an RTScene from a raw (N, 7+) XYZ array and a CameraSnapshot.

    Headless twin of ``build_scene_from_widget``: same colour resolution, slice
    filtering, model-matrix baking (scale · rotation) and light placement, but
    driven entirely by the serialized snapshot instead of a live GL widget.
    """
    from .gui.visualisation import raytrace_export as rt
    from .gui.visualisation.shading import RenderSettings
    from .gui.visualisation.visual_data import VisualData

    rs = render_settings or RenderSettings()

    if snapshot.style not in ("Spheres", "Points"):
        logger.warning("Style %r is not supported headlessly — rendering spheres.",
                       snapshot.style)

    vd = VisualData.from_xyz(np.asarray(xyz))
    colors = _compute_colors(vd, snapshot)
    points = vd.centroids.astype(np.float32)

    mask = _slice_mask(points, snapshot.planes)
    if mask is not None:
        points = points[mask]
        colors = colors[mask]

    # Bake model matrix (M = S · R → p' = s · (R @ p)) and viewport point size,
    # matching the widget path: radius = point_size * 0.2 * scale.
    scale = float(snapshot.scale)
    rot = _quat_matrix(snapshot.model_rotation)
    centers = (points @ rot.T) * scale
    radii = np.full(len(points), float(snapshot.point_size) * 0.2 * scale,
                    dtype=np.float32)
    spheres = np.column_stack([centers, colors, radii]).astype(np.float32)

    camera = rt.RTCamera(
        position=(snapshot.position.x(), snapshot.position.y(), snapshot.position.z()),
        target=(snapshot.target.x(), snapshot.target.y(), snapshot.target.z()),
        up=(snapshot.up.x(), snapshot.up.y(), snapshot.up.z()),
        fov_deg=fov_deg,
        perspective=bool(snapshot.perspective),
        ortho_size=ortho_size,
        aspect=(width / height) if height else 1.0,
    )
    material = rt.RTMaterial(
        ambient=rs.ambient, diffuse=rs.diffuse, specular=rs.specular,
        shininess=rs.shininess, specular_tint=rs.specular_tint,
        brightness=rs.brightness,
    )
    return rt.RTScene(
        spheres=spheres,
        cylinders=np.zeros((0, 10), dtype=np.float32),
        camera=camera,
        light=rt.RTLight(direction=_world_light(snapshot, rs)),
        background=background,
        width=int(width),
        height=int(height),
        material=material,
        photoreal=photoreal,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_frames(spec: str, total: int) -> list[int]:
    """'12' → [12]; '10:40' → [10..40] (inclusive, clamped to the timeline)."""
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total - 1
    else:
        start = end = int(spec)
    start = max(0, start)
    end = min(total - 1, end)
    if start > end:
        raise SystemExit(f"--frames {spec!r}: empty range (timeline has {total} frames)")
    return list(range(start, end + 1))


def _parse_color(spec: str) -> tuple:
    named = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0),
             "grey": (0.5, 0.5, 0.5), "gray": (0.5, 0.5, 0.5)}
    if spec.lower() in named:
        return named[spec.lower()]
    parts = [float(p) for p in spec.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--background {spec!r}: use a name or 'r,g,b' in 0-1")
    return tuple(parts)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cgaspects-render",
        description="Headless batch renderer: XYZ + animation.json → "
                    "POV-Ray/Tachyon frames (and optionally a video).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("xyz", type=Path, help="CrystoGen .XYZ file (single- or multi-frame)")
    p.add_argument("animation", type=Path, help="animation.json saved from the GUI timeline")
    p.add_argument("-o", "--output", type=Path, default=Path("render_frames"),
                   help="output directory for scene files and PNGs")
    p.add_argument("--info", action="store_true",
                   help="print timeline info (frame count, fps, duration) and exit")

    p.add_argument("--backend", choices=("povray", "tachyon"), default="povray")
    p.add_argument("-W", "--width", type=int, default=1920)
    p.add_argument("-H", "--height", type=int, default=1080)
    p.add_argument("--frames", metavar="N[:M]", default=None,
                   help="render only this frame or inclusive range "
                        "(e.g. a SLURM array task id); default: all frames")
    p.add_argument("--export-only", action="store_true",
                   help="write scene files only; do not invoke the renderer")
    p.add_argument("--overwrite", action="store_true",
                   help="re-render frames whose PNG already exists")

    p.add_argument("-j", "--jobs", type=int, default=1,
                   help="frames to render concurrently")
    p.add_argument("--threads", type=int, default=0,
                   help="renderer threads per frame; 0 = cpu_count // jobs")
    p.add_argument("--timeout", type=float, default=600.0,
                   help="per-frame renderer timeout in seconds")

    p.add_argument("--material", default=None,
                   help="material preset (Matte, Metallic, Glossy, …); "
                        "default: viewport defaults")
    p.add_argument("--background", default="white",
                   help="background colour: name or 'r,g,b' in 0-1")
    p.add_argument("--fov", type=float, default=45.0,
                   help="vertical field of view (deg), matches the viewport default")
    p.add_argument("--ortho-size", type=float, default=10.0,
                   help="orthographic half-height, matches the viewport default")
    p.add_argument("--coord-scale", type=float, default=1.0,
                   help="multiply XYZ coordinates (the GUI applies the a-axis "
                        "length when crystallography is loaded)")
    p.add_argument("--default-frame", type=int, default=0,
                   help="XYZ movie frame used when a keyframe holds no data frame")

    p.add_argument("--photoreal", action="store_true",
                   help="enable radiosity/AO, soft shadows (publication quality)")
    p.add_argument("--ao-samples", type=int, default=32)
    p.add_argument("--reflection", type=float, default=0.0,
                   help="0 = matte, up to ~0.4 for a wet look (photoreal only)")
    p.add_argument("--aa", type=int, default=9, help="antialiasing quality")

    p.add_argument("--video", type=Path, default=None,
                   help="stitch rendered frames into this file with ffmpeg")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _render_one(backend: str, scene_path: Path, image_path: Path,
                width: int, height: int, aa: int, timeout: float,
                extra_args: list[str]) -> Path:
    from .gui.visualisation import raytrace_export as rt
    return rt.render_scene_file(backend, scene_path, image_path, width, height,
                                antialiasing=aa, timeout=timeout,
                                extra_args=extra_args)


def _pick_h264_encoder(ffmpeg: str) -> list[str]:
    """Encoder args for the first available H.264 encoder.

    Conda-forge/defaults ffmpeg builds are often GPL-free (no libx264), so probe
    what this build actually has. libopenh264 has no CRF mode — give it an
    explicit bitrate instead.
    """
    proc = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                          capture_output=True, text=True)
    encoders = proc.stdout
    if "libx264" in encoders:
        return ["-c:v", "libx264"]
    if "libopenh264" in encoders:
        return ["-c:v", "libopenh264", "-b:v", "12M"]
    if "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-b:v", "12M"]
    raise SystemExit(
        "This ffmpeg build has no H.264 encoder (libx264/libopenh264). "
        "Install one (e.g. 'conda install -c conda-forge ffmpeg' or a distro "
        "ffmpeg with libx264) and re-run with --video.")


def _stitch_video(out_dir: Path, video: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    pattern = str(out_dir / "frame_%05d.png")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH. Frames are rendered; stitch "
                         "them with:\n  ffmpeg -y -framerate " + str(fps) +
                         f" -i {pattern} -c:v libx264 -pix_fmt yuv420p {video}")
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", pattern,
           *_pick_h264_encoder(ffmpeg), "-pix_fmt", "yuv420p", str(video)]
    logger.info("Stitching video: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed (exit {proc.returncode}):\n"
                         f"{proc.stderr[-2000:]}")
    print(f"Video written: {video}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Imports deferred past --help so argparse stays snappy; keyframe/raytrace
    # modules use PySide6 value types only — no display or QApplication needed.
    from .fileio.xyz_file import read_XYZ
    from .gui.animation.keyframe import AnimationTimeline
    from .gui.visualisation import raytrace_export as rt
    from .gui.visualisation.shading import MATERIAL_PRESETS, RenderSettings

    with open(args.animation) as f:
        timeline = AnimationTimeline.from_dict(json.load(f))
    if not timeline.keyframes:
        raise SystemExit(f"{args.animation}: timeline has no keyframes")

    total = timeline.total_frames()
    if args.info:
        kf_frames = sorted({kf.data_frame for kf in timeline.keyframes
                            if kf.data_frame is not None})
        print(f"Keyframes:     {len(timeline.keyframes)}")
        print(f"Duration:      {timeline.duration:.2f} s @ {timeline.fps} fps")
        print(f"Total frames:  {total}  (0:{total - 1})")
        print(f"Data frames:   {kf_frames if kf_frames else 'none (static XYZ)'}")
        return 0

    frame_ids = _parse_frames(args.frames, total) if args.frames else list(range(total))

    logger.info("Reading XYZ: %s", args.xyz)
    xyz, xyz_movie = read_XYZ(args.xyz)
    if xyz is None or not isinstance(xyz, np.ndarray):
        raise SystemExit(f"{args.xyz}: not a readable XYZ point file")
    movie = xyz_movie if xyz_movie else {0: xyz}
    n_movie = len(movie)
    logger.info("XYZ loaded: %d movie frame(s), %d points in frame 0",
                n_movie, len(movie[0]))

    if args.coord_scale != 1.0:
        movie = {i: arr.copy() for i, arr in movie.items()}
        for arr in movie.values():
            arr[:, 3:6] *= args.coord_scale

    rs = RenderSettings()
    if args.material:
        if args.material not in MATERIAL_PRESETS:
            raise SystemExit(f"Unknown material {args.material!r}. "
                             f"Choose from: {', '.join(MATERIAL_PRESETS)}")
        rs = rs.with_preset(args.material)

    photoreal = None
    if args.photoreal:
        photoreal = rt.PhotorealOptions(
            ao_samples=args.ao_samples,
            reflection=args.reflection,
            antialiasing=args.aa,
        )

    background = _parse_color(args.background)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = rt.SCENE_SUFFIX[args.backend]

    # ---------------------------------------------------------------- export
    scene_paths: dict[int, Path] = {}
    for i in frame_ids:
        t = timeline.time_at_frame(i)
        snapshot, data_frame = timeline.get_state_at_time(t)
        movie_idx = args.default_frame if data_frame is None else data_frame
        movie_idx = max(0, min(movie_idx, n_movie - 1))
        scene = build_scene_from_snapshot(
            movie[movie_idx], snapshot, args.width, args.height,
            render_settings=rs, photoreal=photoreal, background=background,
            fov_deg=args.fov, ortho_size=args.ortho_size,
        )
        if not len(scene.spheres):
            logger.warning("Frame %d: no visible points (slice planes removed "
                           "everything?) — skipping", i)
            continue
        scene_path = out_dir / f"frame_{i:05d}{suffix}"
        rt.write_scene(scene, args.backend, scene_path)
        scene_paths[i] = scene_path

    print(f"Wrote {len(scene_paths)} scene file(s) to {out_dir}/")
    if args.export_only:
        print(f"Render elsewhere with e.g.:\n"
              f"  ls {out_dir}/*{suffix} | parallel -j 8 povray +I{{}} +O{{.}}.png "
              f"+W{args.width} +H{args.height} +FN +A -D -P +WT8")
        return 0

    # ---------------------------------------------------------------- render
    if rt.find_renderer(args.backend) is None:
        raise SystemExit(f"{args.backend} not found on PATH. Install it or use "
                         f"--export-only and render elsewhere.")

    jobs = max(1, args.jobs)
    threads = args.threads or max(1, (os.cpu_count() or 1) // jobs)
    if args.backend == "povray":
        extra = [f"+WT{threads}"]
    else:
        extra = ["-numthreads", str(threads)]
    logger.info("Rendering %d frame(s): %d concurrent, %d threads each",
                len(scene_paths), jobs, threads)

    failed: list[int] = []
    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for i, scene_path in scene_paths.items():
            image_path = out_dir / f"frame_{i:05d}.png"
            if image_path.exists() and not args.overwrite:
                logger.info("frame %05d: exists, skipping (use --overwrite)", i)
                done += 1
                continue
            fut = pool.submit(_render_one, args.backend, scene_path, image_path,
                              args.width, args.height, args.aa, args.timeout, extra)
            futures[fut] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                fut.result()
                done += 1
                print(f"frame {i:05d} done ({done}/{len(scene_paths)})")
            except Exception as exc:
                failed.append(i)
                logger.error("frame %05d FAILED: %s", i, exc)

    if failed:
        raise SystemExit(f"{len(failed)} frame(s) failed: {sorted(failed)}")
    print(f"Rendered {done} frame(s) to {out_dir}/")

    if args.video:
        _stitch_video(out_dir, args.video, timeline.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())

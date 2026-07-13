"""Shared shading model for the sphere/atom/bond renderers.

Holds the user-adjustable material, lighting and ambient-occlusion settings
(:class:`RenderSettings`), the GLSL lighting function injected into each
renderer's fragment shader (:data:`LIGHTING_GLSL`), and the CPU-side
neighbour-density ambient-occlusion estimate (:func:`compute_occlusion`).
"""

import dataclasses
import logging
import math
import time

import numpy as np
from PySide6.QtGui import QVector3D

logger = logging.getLogger("CA:Shading")

# GLSL Blinn-Phong lighting shared by the impostor-sphere and bond shaders.
# All uniforms are optional at draw time; callers pass them via setUniforms.
LIGHTING_GLSL = """
uniform vec3  u_lightDir;       // view-space direction toward the light
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform float u_specularTint;   // 0 = white highlight, 1 = tinted by base colour
uniform int   u_toonLevels;     // 0 = smooth shading, >0 = banded (cel) shading
uniform float u_aoStrength;
uniform float u_brightness;     // overall exposure gain applied to the final colour

vec3 shadeSurface(vec3 albedo, vec3 normal, vec3 viewDir, float occlusion) {
    vec3 L = normalize(u_lightDir);
    float lambert = max(dot(normal, L), 0.0);
    if (u_toonLevels > 0)
        lambert = floor(lambert * float(u_toonLevels) + 0.5) / float(u_toonLevels);
    float ao = 1.0 - u_aoStrength * occlusion;
    vec3 color = min(u_ambient + u_diffuse * lambert, 1.0) * ao * albedo;
    if (u_specular > 0.0) {
        vec3 h = normalize(L + viewDir);
        float spec = pow(max(dot(normal, h), 0.0), u_shininess);
        if (u_toonLevels > 0) spec = spec > 0.5 ? 1.0 : 0.0;
        color += u_specular * spec * ao * mix(vec3(1.0), albedo, u_specularTint);
    }
    return color * u_brightness;
}
"""

# Preset values applied to the matching RenderSettings fields. "Matte" matches
# the historical fixed-function look (ambient 0.3 + diffuse 0.7, no specular).
MATERIAL_PRESETS: dict[str, dict] = {
    "Matte": dict(ambient=0.30, diffuse=0.70, specular=0.0, shininess=16.0,
                  specular_tint=0.0, toon_levels=0),
    "Plastic": dict(ambient=0.28, diffuse=0.65, specular=0.35, shininess=32.0,
                    specular_tint=0.0, toon_levels=0),
    "Glossy": dict(ambient=0.24, diffuse=0.58, specular=0.70, shininess=90.0,
                   specular_tint=0.15, toon_levels=0),
    "Metallic": dict(ambient=0.20, diffuse=0.45, specular=0.90, shininess=64.0,
                     specular_tint=0.85, toon_levels=0),
    "Toon": dict(ambient=0.40, diffuse=0.75, specular=0.25, shininess=16.0,
                 specular_tint=0.0, toon_levels=3),
}


@dataclasses.dataclass
class RenderSettings:
    """Material, lighting and ambient-occlusion settings for the 3D viewport."""

    material: str = "Matte"
    ambient: float = 0.30
    diffuse: float = 0.70
    specular: float = 0.0
    shininess: float = 16.0
    specular_tint: float = 0.0
    toon_levels: int = 0
    # Light direction in view space (a headlight that follows the camera),
    # expressed as degrees. Defaults match the old fixed light (0.2, 0.5, 1.0).
    light_azimuth: float = 11.0
    light_elevation: float = 26.0
    # Overall exposure gain on the final shaded colour (1.0 = unchanged).
    brightness: float = 1.0
    ao_enabled: bool = False
    ao_strength: float = 0.55

    def light_dir(self) -> QVector3D:
        az = math.radians(self.light_azimuth)
        el = math.radians(self.light_elevation)
        return QVector3D(
            math.cos(el) * math.sin(az),
            math.sin(el),
            math.cos(el) * math.cos(az),
        )

    def shader_uniforms(self, perspective: bool = True) -> dict:
        """Uniform dict consumed by the sphere/atom/bond shaders."""
        return {
            "u_lightDir": self.light_dir(),
            "u_ambient": float(self.ambient),
            "u_diffuse": float(self.diffuse),
            "u_specular": float(self.specular),
            "u_shininess": float(max(self.shininess, 1.0)),
            "u_specularTint": float(self.specular_tint),
            "u_toonLevels": int(self.toon_levels),
            "u_aoStrength": float(self.ao_strength) if self.ao_enabled else 0.0,
            "u_brightness": float(self.brightness),
            "u_perspective": 1 if perspective else 0,
        }

    def with_preset(self, name: str) -> "RenderSettings":
        """A copy with the named material preset's fields applied."""
        preset = MATERIAL_PRESETS.get(name)
        if preset is None:
            return dataclasses.replace(self, material=name)
        return dataclasses.replace(self, material=name, **preset)


# Above this many spheres the KD-tree neighbour query becomes too slow for an
# interactive toggle, so AO silently degrades to "no occlusion".
AO_MAX_POINTS = 2_000_000


def compute_occlusion(positions: np.ndarray) -> np.ndarray:
    """Per-sphere ambient-occlusion factor in [0, 1] from local neighbour density.

    Buried particles (dense neighbourhoods) approach 1, exposed surface
    particles approach 0. The neighbourhood radius adapts to the point set:
    2.5x the median nearest-neighbour spacing.
    """
    n = len(positions)
    occ = np.zeros(n, dtype=np.float32)
    if n < 16:
        return occ
    if n > AO_MAX_POINTS:
        logger.warning(
            "Ambient occlusion skipped: %d points exceeds the %d-point limit",
            n, AO_MAX_POINTS,
        )
        return occ

    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    pts = np.ascontiguousarray(positions[:, :3], dtype=np.float64)
    tree = cKDTree(pts)

    # Median nearest-neighbour spacing from a bounded sample
    if n > 20_000:
        sample = pts[np.random.default_rng(0).choice(n, 20_000, replace=False)]
    else:
        sample = pts
    nn_dist, _ = tree.query(sample, k=2, workers=-1)
    spacing = float(np.median(nn_dist[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return occ

    counts = np.asarray(
        tree.query_ball_point(pts, 2.5 * spacing, return_length=True, workers=-1),
        dtype=np.float32,
    )
    # Normalise against a dense interior neighbourhood (90th percentile) so
    # fully buried particles read ~1 and surface particles read well below.
    interior = float(np.percentile(counts, 90))
    if interior <= 1.0:
        return occ
    occ = np.clip((counts - 1.0) / (interior - 1.0), 0.0, 1.0).astype(np.float32)
    logger.debug(
        "Computed AO for %d points in %.2f s (spacing %.3f)",
        n, time.perf_counter() - t0, spacing,
    )
    return occ

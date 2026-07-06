"""Centroid sphere renderer drawn as GPU impostors (see impostor_spheres.py).

Instance buffer layout (7 floats per sphere, as supplied by callers):
    position  [3] float32 – model-space position
    color     [3] float32 – RGB in [0, 1]
    selected  [1] float32 – 0.0 or 1.0

All spheres share one radius derived from the Point Size setting
(u_pointSize * 0.2, matching the historical icosphere scale).
"""

from .impostor_spheres import ImpostorSphereRenderer


class SphereRenderer(ImpostorSphereRenderer):
    INSTANCE_ATTRS = (("position", 3), ("color", 3), ("selected", 1))
    RADIUS_EXPR = "u_pointSize * 0.2"

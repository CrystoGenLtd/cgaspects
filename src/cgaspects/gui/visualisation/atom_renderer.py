"""Atom renderer: GPU-impostor spheres with per-atom VdW radius and CPK colour.

Instance buffer layout (8 floats per atom, as supplied by callers):
    position  [3] float32 – Cartesian world-space position
    color     [3] float32 – RGB in [0, 1]
    selected  [1] float32 – 0.0 or 1.0
    radius    [1] float32 – VdW radius in Angstroms (world-space)
"""

from .impostor_spheres import ImpostorSphereRenderer


class AtomRenderer(ImpostorSphereRenderer):
    INSTANCE_ATTRS = (("position", 3), ("color", 3), ("selected", 1), ("atomRadius", 1))
    # u_pointSize/6.0 gives 1x VdW radius at the default point size of 6
    RADIUS_EXPR = "(u_pointSize / 6.0) * atomRadius"

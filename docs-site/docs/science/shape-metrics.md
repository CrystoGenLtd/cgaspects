# Shape Metrics

This page explains the quantitative shape metrics computed by CGAspects.

!!! warning "Length units are scaled by the unit-cell *a* length"

    The coordinates in a CrystoGen `.XYZ` file are in Ångströms, but they
    are **scaled by the unit-cell *a* lattice parameter** — i.e. they are
    expressed in units of *a*, not raw Å. CGAspects performs **no** rescaling by default:
    every length-valued metric below (L/M/S dimensions, surface area, volume,
    SA:Vol) is reported in whatever units the input file uses. The "Å"-based
    units in the table are therefore only literally correct when the *a* length
    is 1 Å; otherwise multiply by *a* (or *a*², *a*³, 1/*a* as appropriate) to
    recover physical Ångström values. Dimensionless quantities (aspect ratios,
    shape class) are unaffected.

---

## Principal Component Analysis (PCA)

CGAspects uses **Principal Component Analysis (PCA)** to find the three principal dimensions of a crystal point cloud. PCA is performed via Singular Value Decomposition (SVD) of the centered coordinate matrix.

The three principal components define orthogonal directions:
- **PC1** (first principal component) — the direction of greatest extent (long axis, L)
- **PC2** — the direction of second greatest extent (middle axis, M)
- **PC3** — the direction of smallest extent (short axis, S)

The magnitudes of PC1, PC2, PC3 give the dimensions L, M, S used for Zingg classification.

### Why PCA Instead of Bounding Box?

A simple axis-aligned bounding box would give the X, Y, Z extents, which depend on crystal orientation. PCA is orientation-independent: it finds the intrinsic dimensions of the crystal regardless of how it is oriented in space.

---

## Convex Hull

The **convex hull** is the smallest convex shape that contains all points in the crystal. CGAspects uses the convex hull (via `scipy.spatial.ConvexHull`) to compute:

### Surface Area

The sum of the areas of all triangular faces on the convex hull, in Å².

### Volume

The volume enclosed by the convex hull surface, in Å³.

### SA:Vol Ratio

The surface area to volume ratio, in Å⁻¹:

```
SA:Vol = Surface Area / Volume
```

This ratio is important in dissolution: a higher SA:Vol means more surface exposed per unit of crystal mass, leading to faster dissolution. Needles and plates have higher SA:Vol ratios than blocks of the same volume.

---

## Aspect Ratios

From the PCA dimensions:

| Ratio | Formula | Name |
|-------|---------|------|
| Primary (S:M) | S / M | Short / Middle |
| Secondary (M:L) | M / L | Middle / Long |

Both ratios range from 0 to 1. See [Zingg Classification](zingg-classification.md) for how these ratios map to crystal shape classes.

---

## Crystal Extent Along Directions

**Crystallographic Direction Analysis (CDA)** describes the extent of the crystal along selected crystallographic directions.

!!! note "CDA is currently read from simulation output, not computed"

    At present CGAspects does **not** measure CDA itself. The per-direction
    values used in [Aspect Ratio Analysis](../analysis/aspect-ratios.md) are
    **read from the CrystoGen `simulation_parameters.txt` output**. The
    in-code projection described below is planned but not yet implemented.

Once computed in-code, the extent along a direction vector **d** (unit vector) will be:

```
extent = max(points · d) - min(points · d)
```

where the dot product projects each point onto the direction. This gives the crystal's "width" in that specific crystallographic direction, regardless of orientation.

---

## Shape Classification

CGAspects assigns a morphological class to each crystal frame using the Zingg threshold (2/3):

| Class | S:M | M:L | Description |
|-------|-----|-----|-------------|
| Block | ≥ 2/3 | ≥ 2/3 | Equant, roughly cubic |
| Plate | ≥ 2/3 | < 2/3 | Disk-like, one thin dimension |
| Needle | < 2/3 | ≥ 2/3 | Rod-like, one elongated dimension |
| Lath | < 2/3 | < 2/3 | Elongated and flat |

---

## Summary of Computed Metrics

| Metric | Method | Unit |
|--------|--------|------|
| S, M, L dimensions | PCA / SVD | Å* |
| S:M aspect ratio | PCA | dimensionless |
| M:L aspect ratio | PCA | dimensionless |
| Surface area | Convex hull | Å²* |
| Volume | Convex hull | Å³* |
| SA:Vol ratio | Convex hull | Å⁻¹* |
| Shape class | Zingg threshold | — |
| Direction extent | Read from `simulation_parameters.txt` (in-code projection planned) | Å* |

\* Scaled by the unit-cell *a* length — see the warning at the top of this page.

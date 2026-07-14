# Crystallography Primer

This page introduces the crystallographic concepts used in CGAspects. No prior knowledge of crystallography is assumed.

---

## Unit Cell and Lattice Parameters

A crystal is built by repeating a small structural unit — the **unit cell** — in three dimensions. The unit cell is a parallelepiped defined by six parameters:

| Parameter | Symbol | Describes |
|-----------|--------|-----------|
| Cell lengths | a, b, c | The edge lengths of the unit cell (Å) |
| Cell angles | α, β, γ | The angles between cell edges (°) |

- **α** is the angle between the **b** and **c** edges
- **β** is the angle between the **a** and **c** edges
- **γ** is the angle between the **a** and **b** edges

For a cubic crystal: a = b = c, α = β = γ = 90°. For a triclinic crystal, all six parameters are independent.

You set lattice parameters in CGAspects via **Crystallography → Set Lattice Parameters**.

---

## Miller Indices (h k l)

Miller indices are a notation for describing crystal planes. A plane with Miller indices (h k l) intersects the a axis at 1/h, the b axis at 1/k, and the c axis at 1/l (or is parallel to an axis if the corresponding index is 0).

**Common examples**:

| Indices | Plane Description |
|---------|------------------|
| (1 0 0) | Perpendicular to the a axis |
| (0 1 0) | Perpendicular to the b axis |
| (0 0 1) | Perpendicular to the c axis |
| (1 1 0) | Cuts a and b axes equally, parallel to c |
| (1 1 1) | Cuts all three axes equally |

By convention, negative indices are written with a bar: (1 0 -1) is often written as (10$\bar{1}$).

In CGAspects, you enter Miller indices in the [Planes dialog](../features/planes.md). The program converts them to Cartesian normal vectors using the reciprocal lattice.

---

## Direction Indices [u v w]

Crystallographic directions are written as [u v w] in square brackets. The vector [u v w] points along u × **a** + v × **b** + w × **c** in fractional coordinates.

**Common examples**:

| Indices | Direction |
|---------|----------|
| [1 0 0] | Along the a axis |
| [0 1 0] | Along the b axis |
| [0 0 1] | Along the c axis |
| [1 1 0] | Diagonal in the ab plane |
| [1 1 1] | Body diagonal |

In CGAspects, you enter direction indices in the [Directions dialog](../features/directions.md). The program converts them to Cartesian vectors using the direct lattice metric tensor.

---

## Reciprocal Lattice

The **reciprocal lattice** is used to compute the Cartesian normal to a plane from its Miller indices. For a plane with Miller indices (h k l), the normal vector in Cartesian space is:

```
n = h·a* + k·b* + l·c*
```

where **a***, **b***, **c*** are the reciprocal lattice vectors derived from the unit cell parameters.

This conversion is done automatically in CGAspects when you enter Miller indices for a plane.

---

## Fractional vs Cartesian Coordinates

**Cartesian coordinates** (X, Y, Z) are expressed in Å along orthogonal axes. This is the coordinate system of the XYZ files.

**Fractional coordinates** (in terms of a, b, c) express positions as fractions of the unit cell edges. A point at (0.5, 0.5, 0.5) in fractional coordinates is at the centre of the unit cell.

CGAspects normally works in Cartesian coordinates, but can display axes in fractional (a/b/c) mode using `Shift+A`.

---

## Crystal Faces and Morphology

The external shape of a crystal is determined by which crystal planes (faces) grow slowest — these are the most stable faces that persist as the crystal grows. Fast-growing faces grow away and disappear; slow-growing faces dominate the final morphology.

In CrystoGen, each face is characterized by its Miller indices and its relative growth rate. CGAspects visualises these faces as planes and computes aspect ratios to quantify the resulting shape.

# Lattice Parameters

Open the Lattice Parameters dialog with **Crystallography → Set Lattice Parameters**.

Lattice parameters define the crystal's unit cell geometry. They are required to:

- Use Miller index (h k l) input for **planes**
- Use fractional [u v w] input for **directions**
- Display crystallographic a/b/c axes instead of Cartesian X/Y/Z

---

## Parameters

| Parameter | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| a | a | Å | Unit cell length along a |
| b | b | Å | Unit cell length along b |
| c | c | Å | Unit cell length along c |
| Alpha | α | ° | Angle between b and c |
| Beta | β | ° | Angle between a and c |
| Gamma | γ | ° | Angle between a and b |

For a cubic crystal: a = b = c, α = β = γ = 90°.

---

## Input Methods

### Manual Entry

Enter each parameter directly into the number fields. Click **OK** or **Apply** to save.

**Validation**: Cell lengths must be positive; angles must be between 0° and 180°.

### From CIF File

Click **Load CIF** and select a `.cif` (Crystallographic Information File). The lattice parameters are extracted automatically from the `_cell_length_*` and `_cell_angle_*` fields.

### From CrystoGen Structure File

Click **Load Structure File** and select a CrystoGen output structure file. The lattice parameters are extracted from the file header.

---

## Effect on the Application

Once set, lattice parameters affect:

- **Planes dialog**: Miller (h k l) mode becomes available
- **Directions dialog**: Fractional [u v w] mode becomes available
- **Axes**: The a/b/c fractional axes are computed and can be displayed (`Shift+A`)
- **Coordinate display**: Miller indices are shown alongside Cartesian normals

---

## Auto-Loading

If a structure file is present in the loaded data folder, CGAspects automatically reads the lattice parameters on import. You can still override them manually in this dialog.

---

## Coordinate Transformations

CGAspects converts between coordinate systems using standard crystallographic transformations:

- **Miller indices (h k l) → Cartesian normal**: via the reciprocal lattice metric tensor
- **Fractional indices [u v w] → Cartesian vector**: via the direct lattice metric tensor

These conversions use the lattice parameters you provide.

# File Formats

## Input Formats

### XYZ Files

The primary input format. Each file represents one crystal structure frame (time step).

**File extension**: `.xyz` (or `.txt` with XYZ content)

**Structure**:
```
<N>                         ← number of points
<comment line>              ← metadata or blank
<type> <id> <layer> <x> <y> <z> [<site> <energy>]
<type> <id> <layer> <x> <y> <z> [<site> <energy>]
...                         ← N rows of point data
```

**Columns**:

| Column | Name | Description |
|--------|------|-------------|
| 1 | Type | Molecule/atom type identifier |
| 2 | Number | Atom number |
| 3 | Layer | Growth layer index |
| 4 | X | X coordinate (Å) |
| 5 | Y | Y coordinate (Å) |
| 6 | Z | Z coordinate (Å) |
| 7 | Site | Site number (optional) |
| 8 | Energy | Particle energy (optional) |

**Multi-frame trajectories**: Load an entire folder — each XYZ file in the folder becomes one frame. Files are sorted by filename to determine frame order.

---

### CIF Files

Crystallographic Information Files are used to import lattice parameters.

**File extension**: `.cif`

CGAspects reads:
- `_cell_length_a`, `_cell_length_b`, `_cell_length_c`
- `_cell_angle_alpha`, `_cell_angle_beta`, `_cell_angle_gamma`

---

### CrystoGen Structure File

The simulation structure file produced by CrystoGen. Contains lattice parameters and optionally crystallographic direction information.

Auto-loaded if present in the data folder.

---

### Size File (`size.csv`)

Contains crystal dimensions along each crystallographic direction at each time step. Used for [Growth Rate Analysis](analysis/growth-rates.md).

**Format**: CSV with columns for time and each direction's size measurement.

---

### Simulation Parameters File

A text file (`.txt`) produced by CrystoGen containing:
- List of crystallographic directions used in the simulation
- Supersaturation levels
- Other simulation metadata

Auto-detected in the data folder.

---

### Crystallization Event Files

Binary or text files recording when each lattice site crystallized during the simulation. Used for [Site Analysis](analysis/site-analysis.md).

---

### Population / Occupation Files

Record the occupancy fraction of each site over simulation time. Used for site analysis.

---

### Count Files

Record the number of crystallization/dissolution events per site. Used for site analysis.

---

### CSV Files (for Plotting)

Any CSV file with a header row and numerical data can be loaded for plotting:

```
column1,column2,column3
1.0,2.5,block
2.0,3.1,needle
...
```

Load via **File → Import CSV for Plotting** or **File → Import CSV from Clipboard**.

---

## Output Formats

### CSV Analysis Results

All analysis outputs are saved as CSV files in `aspects_output/` inside the data folder:

| File | Analysis | Contents |
|------|---------|---------|
| `aspect_ratios.csv` | Aspect Ratio | S, M, L dimensions, ratios, shape, SA, Vol per frame |
| `cda_results.csv` | Aspect Ratio (CDA) | Crystal extent along each direction per frame |
| `growth_rates.csv` | Growth Rate | Size vs time per direction |
| `site_analysis.csv` | Site Analysis | All site properties |
| `interaction_data.csv` | Site Analysis | Interaction frequencies per site |

---

### Plot Images

Plots are saved to `aspects_output/CGPlots/`:

| Format | Extension | Description |
|--------|-----------|-------------|
| PNG | `.png` | Raster image, 300 DPI by default |
| PDF | `.pdf` | Vector, scalable, publication-ready |
| EPS | `.eps` | Vector, for LaTeX inclusion |
| SVG | `.svg` | Vector, web-friendly |

---

### XYZ Export

**File → Export XYZ** (`Ctrl+Shift+E`) saves the current point cloud to an XYZ file, including any deletions made in the viewport.

---

### 3D Mesh Export

The viewport geometry can be exported as a 3D mesh:

| Format | Extension | Notes |
|--------|-----------|-------|
| OBJ | `.obj` | Wavefront, includes normals |
| STL | `.stl` | Binary, suitable for 3D printing |
| PLY | `.ply` | Stanford PLY |
| GLB | `.glb` | glTF binary, for web/game engines |

---

### Rendered Image

**File → Render** (`Ctrl+R`) saves the current 3D viewport as a PNG image at 1×, 2×, or 4× resolution.

---

### Log File

The application log is written to a platform-specific location. Access it via **View → Open Log File**.

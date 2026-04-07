# Installation

## Requirements

- **Python** 3.10 or newer
- **Operating System**: macOS, Windows, or Linux
- **GPU**: OpenGL 3.3 compatible (required for 3D visualisation)

### Python Dependencies

CGAspects depends on the following packages (installed automatically):

| Package | Purpose |
|---------|---------|
| PySide6 | Qt6 GUI framework |
| NumPy | Array operations and linear algebra |
| SciPy | Convex hull, spatial analysis, signal processing |
| Pandas | Data frames for analysis results |
| Matplotlib | Scientific plotting |
| Seaborn | Statistical data visualisation |
| scikit-learn | Machine learning and clustering |
| Trimesh | 3D mesh I/O and geometry |
| PyOpenGL | OpenGL bindings |
| natsort | Natural sorting of file and directory names |

---

## Install from PyPI

```bash
pip install cgaspects
```

---

## Install from Source

```bash
git clone https://github.com/CrystalGrowerOrg/cgaspects.git
cd cgaspects
pip install -e .
```

---

## Virtual Environment (Recommended)

Using a virtual environment avoids dependency conflicts:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

pip install cgaspects
```

---

## Launching the Application

After installation, run:

```bash
cgaspects
```

The main window will open. No further configuration is required to start visualising XYZ files.

---

## Verifying the Installation

If the application opens and you can see the 3D viewport, the installation is working correctly.

If you see OpenGL errors, ensure your graphics drivers are up to date and support OpenGL 3.3.

---

## Updating

```bash
pip install --upgrade cgaspects
```

---

## Troubleshooting

**`cgaspects: command not found`**
Make sure the Python `bin` (or `Scripts`) directory is on your `PATH`, or run via:
```bash
python -m cgaspects
```

**OpenGL errors on startup**
Update your GPU drivers. On Linux, you may need to install `libgl1-mesa-glx`.

**Application crashes loading XYZ files**
Ensure your XYZ files follow the [expected format](file-formats.md#xyz-files). Check the log file via **View → Open Log File** for details.

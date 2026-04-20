# CGAspects

**CGAspects** is a scientific GUI application for analysing and visualising crystal growth simulation data produced by [CrystoGen](https://crystogen.org) (formerly CrystalGrower). It provides interactive 3D visualisation of crystal point clouds, quantitative shape analysis, growth rate analysis, and site crystallization analysis.

---

## What CGAspects Does

- **Visualises** crystal structures from XYZ point cloud files in an interactive 3D viewport
- **Analyses** crystal shape using aspect ratio metrics and Zingg classification
- **Tracks** crystal growth rates over time from time-series simulation data
- **Characterises** crystallization events at individual lattice sites
- **Plots** analysis results with customisable scatter plots, histograms, and trend lines
- **Exports** results as CSV data, publication-quality plots, and 3D mesh files

---

## Quick Start

1. [Install CGAspects](installation.md)
2. Launch with `cgaspects` from the terminal
3. Follow the [Basic Workflow Tutorial](tutorials/basic-workflow.md)

---

## Key Features at a Glance

| Feature | Description |
|---------|-------------|
| 3D Viewport | Interactive OpenGL crystal visualisation with orbit camera |
| Crystallographic Planes | Add Miller-index or Cartesian planes; slice point cloud |
| Crystallographic Directions | Visualise direction vectors as lines, arrows, or cylinders |
| Aspect Ratio Analysis | Zingg shape classification and CDA from simulation data |
| Growth Rate Analysis | Track crystal size over time with smoothing and trend fitting |
| Site Analysis | Analyse crystallization events, site energies, and interaction patterns |
| Plotting Dialog | Configurable scatter plots, histograms, heatmaps |
| Movie Playback | Animate crystal growth trajectories frame by frame |

---

## System Requirements

- Python 3.10 or newer
- macOS, Windows, or Linux
- OpenGL 3.3-compatible GPU

---

## Status

CGAspects is currently in **alpha** (v0.9.8). Expect active development and occasional breaking changes.

---

## Source Code & Issues

- Repository: [github.com/crystogenltd/cgaspects](https://github.com/crystogenltd/cgaspects)
- Bug reports: [GitHub Issues](https://github.com/crystogenltd/cgaspects/issues)

# Changelog

All notable changes to this project since the new XYZ parser (commit 7990fa4).

## [Unreleased]

### Added

#### Site Analysis Features (2025-12-19 to 2025-12-23)
- **Site analysis and site parser** (1c7ad0d, 2025-12-19)
  - New site analysis module for analyzing crystallization sites
  - Site parser for reading population and events data
  - Test coverage for site parser functionality
  - Test resources: `test_crystallisation_events.csv` and `test_populations.csv`

- **Site highlighting and coloring** (37b8f73, ca13ded, 2025-12-20 to 2025-12-21)
  - Site highlighting dialog for interactive site selection
  - Site coloring capabilities in visualization
  - Improved handling of population and events data

- **Site analysis plot integration** (ef5c2ff, c17a20f, 2025-12-22)
  - Connected site analysis plots to visualizer
  - Support for loading events and population data from different files
  - New time series widget for displaying site analysis data

#### Visualization Features

- **Heatmap support** (1640739, 2025-12-09)
  - Added heatmap visualization capabilities
  - Enhanced aspect ratio dataframes for heatmap rendering

- **3D mesh export** (d271bb4, 2025-12-10)
  - Export functionality for 3D mesh data

- **Fractional axes conversion** (ffd5c14, 2025-12-19)
  - Lattice dialog for crystallographic parameters
  - Conversion between Cartesian and fractional coordinates
  - New crystallography utilities module

- **Axes with arrows and settings** (c3f7a5d, 2025-12-22)
  - Axes settings dialog for customizing axis display
  - Arrow indicators on axes
  - Enhanced axes renderer with improved visual options

- **Data filtering and customization** (0517f35, 2025-12-20)
  - Data filter dialog for selective data visualization
  - Label customization dialog for plot labels
  - Colorbar label improvements
  - Custom label support
  - Data filtering feature documentation (`DATA_FILTER_FEATURE.md`)

#### UI/UX Improvements

- **Growth/dissolution relabeling** (c206344, 2025-12-23)
  - Improved labeling for growth and dissolution processes
  - Option to hide bulk sites in visualization

- **Log file management** (db4db76, 2025-12-22)
  - Log files now stored in `.crystogen` folder
  - Text file viewer widget for viewing logs
  - Standalone growth rate plotter utility

### Changed

- **Visualization settings reorganization** (db4db76, 2025-12-22)
  - Improved visualization settings widget
  - Better organization of settings dialogs
  - Enhanced OpenGL rendering settings

- **Enhanced error handling** (1fcaac2, 2025-12-18)
  - UTF-8 error skips in file parsing
  - Improved error handling in aspect ratio and growth rate dataframes

- **CI/CD improvements** (1fcaac2, 2025-12-18)
  - Updated GitHub Actions workflow for package building

### Removed

- **chmpy dependency** (1dd4142, 2025-12-22)
  - Removed chmpy (spherical harmonics) dependency
  - Deleted `sph.py` module
  - Updated shape analysis to work without chmpy
  - Updated tests to reflect removal

### Fixed

- **Site highlighting** (ca13ded, 2025-12-21)
  - Fixed site highlighting functionality
  - Improved site selection and display

- **Site coloring** (37b8f73, 2025-12-20)
  - Initial attempt at site coloring (required fixes in later commits)

### Technical Details

#### Modified Files Summary
- GUI dialogs: Multiple new dialogs added for settings, filtering, and customization
- Visualization: Enhanced OpenGL rendering, axes rendering, and display options
- Analysis: New site analysis module and improved data handling
- File I/O: Better logging system and data file management
- Testing: New tests for site parser and updated shape analysis tests

#### Dependencies
- Removed: chmpy
- Updated: pyproject.toml dependency configuration

---

**Date Range**: 2025-12-09 to 2025-12-23
**Commits**: 15 commits since 7990fa4
**Contributors**: AJen01

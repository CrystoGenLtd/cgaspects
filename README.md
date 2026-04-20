# CGAspects

CGAspects is a CrystoGen (formerly CrystalGrower) data analysis tool equipped with a PySide6 GUI. 

## Features

- **Data Analysis and Visualization**: It is designed for the analysis and visualization of crystal growth data, including aspect ratios (Zingg analysis), surface area to volume ratios, growth rates and more.
- **PySide6 GUI**: A user-friendly graphical user interface for interaction with the application.

## Installation

Ensure you have Python >=3.10 installed on your system.

### Via PyPI (recommended)

```bash
pip install cgaspects
```

### From source

1. Clone the CGAspects repository (shallow clone with depth 1):

    ```bash
    git clone --depth=1 https://github.com/crystogenltd/cgaspects.git
    cd cgaspects
    ```

2. Install CGAspects via pip:

    ```bash
    pip install .
    ```

## Usage

After installation, you can run CGAspects using the command:

```bash
cgaspects
```

## Building and Packaging for macOS

CGAspects can be packaged as a standalone application for macOS using PyInstaller. 

To build and package the application:

1. Install pyinstaller in your python environment:
   
    ```bash
    pip install pyinstaller
    ```
2. Navigate to the project directory and run the provided script:

    ```bash
    ./scripts/mac_os_bundle.sh
    ```

3. After the build completes, the script will run an ad hoc code signing to sign the application.
   This will create a `.app` package in the `dist` directory, which can be distributed and run on macOS systems.

## Dependencies

    matplotlib>=3.10.8
    natsort>=8.4.0
    numpy>=2.4.0
    pandas>=2.3.3
    PyOpenGL>=3.1.10
    PySide6>=6.10.1
    scikit-learn>=1.5.0
    scipy>=1.16.3
    seaborn>=0.13.2
    trimesh>=4.8.3

## Development Status

CGAspects is currently in Beta stage.

## Authors

- **Alvin J. Walisinghe** - [jennerwalisinghe@gmail.com](mailto:jennerwalisinghe@gmail.com)
- **Peter R. Spackman** - [peterspackman@fastmail.com](mailto:peterspackman@fastmail.com)
- **Nathan de Bruyn**

## Contributions

Contributions are welcome! Please read the [contribution guidelines](https://github.com/crystogenltd/cgaspects/blob/main/CONTRIBUTING.md) before submitting a pull request.

## Links

- [Homepage](https://github.com/crystogenltd/cgaspects)
- [Documentation](https://crystogenltd.github.io/cgaspects/)
- [Repository](https://github.com/crystogenltd/cgaspects)
- [Submit Issues](https://github.com/crystogenltd/cgaspects/issues)

## Contact

For support or inquiries, please email the authors or create an issue in the project's issue tracker.

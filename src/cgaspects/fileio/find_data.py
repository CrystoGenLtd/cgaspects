import logging
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..gui.utils.crystallography import Cell
from ..utils.data_structures import file_info_tuple

logger = logging.getLogger("CA:FileIO")


def file_empty(x):
    return x.stat().st_size == 0


def filter_xyz_files(crystal_xyz_list):
    class FileSelectionDialog(QDialog):
        def __init__(self, unique_suffixes):
            super().__init__()
            self.setWindowTitle("Select XYZ File Types")
            self.selected_suffixes = []

            layout = QVBoxLayout()

            # Create a QListWidget for selection
            self.list_widget = QListWidget()
            for suffix in unique_suffixes:
                self.list_widget.addItem(suffix)
            self.list_widget.setSelectionMode(QListWidget.MultiSelection)

            layout.addWidget(self.list_widget)

            # Add OK and Cancel buttons
            self.ok_button = QPushButton("OK")
            self.ok_button.clicked.connect(self.accept_selection)
            layout.addWidget(self.ok_button)

            self.setLayout(layout)

        def accept_selection(self):
            selected_items = self.list_widget.selectedItems()
            self.selected_suffixes = [item.text() for item in selected_items]
            self.accept()

    # Get unique suffixes from file names
    suffixes = {Path(f).stem.split("_")[-1] for f in crystal_xyz_list}
    suffixes = natsorted(suffixes)

    if len(suffixes) == 1:
        return [
            f
            for f in crystal_xyz_list
            if Path(f).stem.split("_")[-1] in suffixes and not Path(f).name.startswith("._")
        ]

    if len(suffixes) < 1:
        raise NameError("Expected NAME_SUFFIX.XYZ, but no suffixes were identified.")

    # Show the dialog for user selection
    app = QApplication.instance() or QApplication([])
    dialog = FileSelectionDialog(suffixes)
    if dialog.exec() == QDialog.Accepted:
        selected_suffixes = dialog.selected_suffixes
        if not selected_suffixes:
            QMessageBox.warning(
                None,
                "No Selection",
                "No file types were selected. Returning all files.",
            )
        else:
            # Filter files based on selected suffixes
            return [
                f
                for f in crystal_xyz_list
                if Path(f).stem.split("_")[-1] in selected_suffixes
                and not Path(f).name.startswith("._")
            ]

    return crystal_xyz_list


# Incorporate the filter step into the main function
def locate_xyz_files(xyz_folderpath):
    if xyz_folderpath is None:
        logger.debug("Folder selection was canceled or no folder was selected.")
        return None

    xyz_folderpath = Path(xyz_folderpath)

    crystal_xyz_list = []
    docking_xyz_list = []

    try:
        if xyz_folderpath.is_dir():
            for filepath in Path(xyz_folderpath).rglob("*.XYZ"):
                if file_empty(filepath):
                    logger.info("Ignore empty XYZ file: %s", filepath)
                elif filepath.stem.endswith("_docking"):
                    docking_xyz_list.append(filepath)
                else:
                    crystal_xyz_list.append(filepath)

            if not crystal_xyz_list:
                raise FileNotFoundError("No .XYZ files found in the selected directory.")
        else:
            raise NotADirectoryError(f"{xyz_folderpath} is not a valid directory.")

        # Natural sort
        crystal_xyz_list = natsorted(crystal_xyz_list)
        docking_xyz_list = natsorted(docking_xyz_list)

        # Allow user to filter files
        crystal_xyz_list = filter_xyz_files(crystal_xyz_list)

    except (FileNotFoundError, NotADirectoryError, NameError) as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(
            f"{e}\nPlease make sure the folder you have selected "
            "contains valid .XYZ files from the simulation(s)."
        )
        msg.setWindowTitle("Error! No valid XYZ files detected.")
        msg.exec()
        return None

    logger.debug("%s .XYZ Files Found", len(crystal_xyz_list))
    return crystal_xyz_list, docking_xyz_list


def create_aspects_folder(path):
    """Creates crystalaspects folder"""
    time_string = time.strftime("%Y%m%d-%H%M%S")
    savefolder = Path(path) / "crystalaspects" / time_string
    savefolder.mkdir(parents=True, exist_ok=True)

    return savefolder


def find_info(path):
    """The method returns the crystallographic directions,
    supersaturation, and the size file paths from a CG simulation folder."""
    logger.debug("find_info called at: %s", path)
    path = Path(path)

    supersats, directions = [], []
    folders = []
    exclude_suffixes = {"XYZ_files", "CrystalAspects", "CrystalMaps", "crystalaspects"}

    # Initialize list containers for files found within directories
    size_files = []
    found_simparam_files = False
    summary_file = None
    growth_mod = None
    structure_file = None
    crystallisation_files = []
    population_files = []
    count_files = []

    # Check the root directory for the summary file
    summary_file = next(path.glob("*summary.csv"), None)

    # Process directories and files within them
    _folders = natsorted(path.iterdir())
    for entry in _folders:
        if not entry.is_dir() and entry.name not in exclude_suffixes:
            continue
        folders.append(entry)
        # Search for size and parameter files within the approved folders
        for file in entry.glob("*"):
            if file.name.endswith("size.csv") and file.stat().st_size > 0:
                size_files.append(file)
            elif file.name.endswith("_crystallisation_events.csv") and file.stat().st_size > 0:
                crystallisation_files.append(file)
            elif file.name.endswith("_populations.csv") and file.stat().st_size > 0:
                population_files.append(file)
            elif file.name.endswith("simulation_parameters.txt"):
                found_simparam_files = True
                with open(file, "r", encoding="utf-8", errors="replace") as file:
                    lines = file.readlines()
                growth_mod, structure_file = process_simulation_parameters(
                    lines, supersats, directions, growth_mod, structure_file, path
                )
            elif file.name.endswith("count.txt"):
                count_files.append(file)

    # Process parameter files if present
    if not found_simparam_files:
        logger.warning("No simulation parameter files found.")

    # Log if size files are not found
    if not size_files:
        logger.warning("No size files found.")

    if not found_simparam_files and size_files:
        QMessageBox.warning(
            None,
            "Missing Data",
            "No simulation parameter and size files found in the directory.\n"
            "Please make sure you've selected a valid CrystalGrower output directory.",
        )

    return file_info_tuple(
        supersats,
        size_files,
        directions,
        growth_mod,
        folders,
        summary_file,
        crystallisation_files,
        population_files,
        count_files,
        structure_file,
    )


def process_simulation_parameters(
    lines: list,
    supersats: list,
    directions: list,
    growth_mod: str | None,
    structure_file: Path | None = None,
    basedir: Path | None = None,
):
    get_facets = True
    for line in lines:
        if line.startswith("Starting delta mu value (kcal/mol):"):
            supersat = float(line.split()[-1])
            supersats.append(supersat)
        if line.startswith("normal, ordered or growth modifier"):
            growth_mod = line.endswith("growth_modifier\n")
        if line.startswith("File containing TOPOS input?:") and structure_file is None:
            path = line.split(":", 1)[1].strip()
            if path and path != "N/A":
                path = Path(path)

                if path.exists():
                    structure_file = path
                else:
                    cwd_path = basedir / path.name
                    if cwd_path.exists():
                        structure_file = cwd_path
        if line.startswith("Size of crystal") and get_facets:
            frame = lines.index(line) + 1
            for n in range(frame, len(lines)):
                line = lines[n]
                if line in [" \n", "\n"]:
                    get_facets = False
                    break
                facet = line.split("      ")[0]
                if len(facet) <= 8 and facet not in directions:
                    directions.append(facet)
    return growth_mod, structure_file


def find_growth_directions(csv):
    """Returns the lenghts from a size_file"""
    lt_df = pd.read_csv(csv, encoding="utf-8", encoding_errors="replace")
    columns = lt_df.columns
    directions = []
    for col in columns:
        # Direction columns start with space or minus sign (Miller indices)
        if col.startswith((" ", "-")):
            directions.append(col)

    return directions


def summary_compare(summary_csv, aspect_csv=False, aspect_df=""):
    summary_df = pd.read_csv(summary_csv, encoding="utf-8", encoding_errors="replace")

    if aspect_csv:
        aspect_df = pd.read_csv(aspect_csv, encoding="utf-8", encoding_errors="replace")

    summary_cols = summary_df.columns
    aspect_cols = aspect_df.columns

    # This allows backcompatibility with
    # an older version of CrystalGrower

    search = str(summary_df.iloc[0, 0])
    search = search.split("_")
    start_num = int(search[-1])
    search_string = "_".join(search[:-1]) if len(search) > 1 else None

    int_cols = summary_cols[1:]
    summary_df = summary_df.set_index(summary_cols[0])
    summary_df.index = summary_df.index.astype(str)
    compare_array = np.empty((0, len(aspect_cols) + len(int_cols)))

    for _, row in aspect_df.iterrows():
        sim_num = None
        try:
            sim_num = int(row["Simulation Number"]) - 1 + start_num
        except TypeError:
            sim_num = row["Simulation Number"]
        except KeyError:
            sim_num = row.iloc[0]

        num_string = f"{search_string}_{sim_num}" if search_string is not None else str(sim_num)
        aspect_row = row.values
        aspect_row = np.array([aspect_row])
        collect_row = summary_df.filter(items=[num_string], axis=0).values
        if collect_row.shape[0] == 0:
            raise ValueError(
                f"Simulation '{num_string}' was not found in the summary file. "
                "The summary file may not match the current dataset."
            )
        collect_row = np.concatenate([aspect_row, collect_row], axis=1)
        compare_array = np.append(compare_array, collect_row, axis=0)

    cols = aspect_cols.append(int_cols)
    compare_df = pd.DataFrame(compare_array, columns=cols)
    full_df = compare_df.sort_values(by=["Simulation Number"], ignore_index=True)
    return full_df


def summary_has_starting_delmu(summary_csv):
    """Return True if the summary CSV contains a 'starting_delmu' column."""
    df = pd.read_csv(summary_csv, nrows=0, encoding="utf-8", encoding_errors="replace")
    return any(col.startswith("starting_delmu") for col in df.columns)


def apply_supersat_mode(df, mode):
    """Resolve which supersaturation column(s) to keep.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that may contain both a native 'Supersaturation' column
        (from simulation_parameters.txt) and a 'starting_delmu' column
        (from a summary CSV).
    mode : str
        'native'         – keep 'Supersaturation', drop 'starting_delmu' (default)
        'starting_delmu' – drop 'Supersaturation', rename 'starting_delmu' → 'Supersaturation'
        'both'           – keep both columns unchanged
    """
    delmu_cols = [c for c in df.columns if c.startswith("starting_delmu")]
    if not delmu_cols:
        return df
    if mode == "native":
        df = df.drop(columns=delmu_cols, errors="ignore")
    elif mode == "starting_delmu":
        df = df.drop(columns=["Supersaturation"], errors="ignore")
        df = df.rename(columns={delmu_cols[0]: "Supersaturation"})
    # 'both' – no changes
    return df


def combine_xyz_cda(CDA_df, XYZ_df):
    cda_cols = CDA_df.columns[1:]
    xyz_cols = XYZ_df.columns

    logger.debug(
        "Attempting to combine CDA [%s] and XYZ [%s] dataframes",
        CDA_df.shape,
        XYZ_df.shape,
    )
    # Initialize a list to collect rows
    collected_rows = []
    combine_array = np.empty((0, len(xyz_cols) + len(cda_cols)))

    for index, row in XYZ_df.iterrows():
        sim_num = int(row["Simulation Number"]) - 1
        xyz_row = row.values
        xyz_row = np.array([xyz_row])
        cda_row = CDA_df.iloc[sim_num, 1:].values.reshape(1, -1)

        # Concatenate xyz_row with cda_row along axis 1 to form a combined row
        combined_row = np.concatenate([xyz_row, cda_row], axis=1)
        collected_rows.append(combined_row[0])

    # Convert the collected rows into a DataFrame
    combined_array = np.asarray(collected_rows)
    logger.debug("Combined array shape: %s", combined_array.shape)
    cols = xyz_cols.tolist() + cda_cols.tolist()
    logger.debug("Combined column titles: %s", cols)
    compare_df = pd.DataFrame(combined_array, columns=cols)
    combine_df = compare_df.sort_values(by=["Simulation Number"], ignore_index=True)

    logger.debug("Combined df:\n%s", combine_df)

    return combine_df


@dataclass
class MolAtom:
    symbol: str
    frac: np.ndarray  # fractional coordinates, shape (3,)


@dataclass
class MolTemplate:
    formula: str
    mol_type: int
    atoms: list[MolAtom]
    bonds: list[tuple[int, int]]  # 0-based atom index pairs


def parse_molecular_data(file_path: str | Path) -> dict[int, MolTemplate] | None:
    """Parse per-molecule-type atom coordinates and bonds from a CrystalGrower structure file.

    After the 'Non primitive data' lattice block, the file has one block per molecule type:

        FORMULA  TYPE  N_ATOMS/N_BONDS
        INDEX  SYMBOL  FRAC_X  FRAC_Y  FRAC_Z
        ...
        (blank line)
        ATOM1  ATOM2
        ...
        (blank line)

    Returns a dict mapping mol_type (int) → MolTemplate, or None on failure.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning("Structure file not found: %s", file_path)
        return None

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Locate the "Non primitive data" section
        start = None
        for i, line in enumerate(lines):
            if "Non primitive data" in line:
                start = i + 3  # skip the two lattice-parameter lines
                break

        if start is None:
            logger.warning("'Non primitive data' section not found in %s", file_path)
            return None

        templates: dict[int, MolTemplate] = {}
        i = start

        while i < len(lines):
            raw = lines[i].strip()
            parts = raw.split()

            # Molecule header: FORMULA  TYPE  N_ATOMS/N_BONDS
            if len(parts) == 3 and "/" in parts[2]:
                try:
                    formula = parts[0]
                    mol_type = int(parts[1])
                    n_atoms, n_bonds = map(int, parts[2].split("/"))
                except ValueError:
                    i += 1
                    continue

                i += 1
                atoms: list[MolAtom] = []
                for _ in range(n_atoms):
                    if i >= len(lines):
                        break
                    aparts = lines[i].strip().split()
                    if len(aparts) >= 5:
                        symbol = aparts[1]
                        frac = np.array(
                            [float(aparts[2]), float(aparts[3]), float(aparts[4])],
                            dtype=np.float64,
                        )
                        atoms.append(MolAtom(symbol=symbol, frac=frac))
                    i += 1

                # Skip blank line(s) between atoms and bonds
                while i < len(lines) and not lines[i].strip():
                    i += 1

                bonds: list[tuple[int, int]] = []
                for _ in range(n_bonds):
                    if i >= len(lines):
                        break
                    bparts = lines[i].strip().split()
                    if len(bparts) == 2:
                        try:
                            bonds.append((int(bparts[0]) - 1, int(bparts[1]) - 1))
                        except ValueError:
                            pass
                    i += 1

                templates[mol_type] = MolTemplate(
                    formula=formula, mol_type=mol_type, atoms=atoms, bonds=bonds
                )
            else:
                i += 1

        if not templates:
            logger.warning("No molecule templates found in %s", file_path)
            return None

        logger.info("Parsed %d molecule template(s) from %s", len(templates), file_path.name)
        return templates

    except Exception as e:
        logger.error("Error parsing molecular data from %s: %s", file_path, e)
        return None


def parse_structure_file(file_path: str | Path) -> Cell | None:
    """Parse a CrystalGrower structure file to extract lattice parameters.

    The file format has a section starting with 'Non primitive data' followed by
    lattice parameters:
        a   b   c
        alpha beta gamma

    Parameters
    ----------
    file_path : str or Path
        Path to the CrystalGrower structure file.

    Returns
    -------
    Cell or None
        Cell object with lattice parameters, or None if parsing fails.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning("Structure file not found: %s", file_path)
        return None

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # Find the "Non primitive data" section
        non_primitive_index = None
        for i, line in enumerate(lines):
            if "Non primitive data" in line:
                non_primitive_index = i
                break

        if non_primitive_index is None:
            logger.warning("Could not find 'Non primitive data' section in %s", file_path)
            return None

        # The next line should contain a, b, c
        abc_line = lines[non_primitive_index + 1].strip()
        a, b, c = map(float, abc_line.split())

        # The line after that should contain alpha, beta, gamma
        angles_line = lines[non_primitive_index + 2].strip()
        alpha, beta, gamma = map(float, angles_line.split())

        logger.info(
            "Parsed lattice parameters from %s: a=%.4f b=%.4f c=%.4f α=%.3f° β=%.3f° γ=%.3f°",
            file_path.name,
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
        )

        return Cell(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

    except Exception as e:
        logger.error("Error parsing structure file %s: %s", file_path, e)
        return None

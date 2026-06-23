import logging
import os
from itertools import permutations
from pathlib import Path
import pandas as pd

from .shape_analysis import ShapeAnalyser
from ..fileio.xyz_file import CrystalCloud
from ..utils.naming import simulation_id_from_path

LOG = logging.getLogger("CA:AR-Dataframes")


def merge_dicts(dict_list):
    merged = {}

    for d in dict_list:
        for k, v in d.items():
            if k in merged:
                # If the key exists, merge values
                if isinstance(merged[k], list) and isinstance(v, list):
                    merged[k] += v
                elif isinstance(merged[k], set) and isinstance(v, set):
                    merged[k] |= v
                else:
                    raise TypeError(f"Unsupported types for merge: {type(merged[k])}, {type(v)}")
            else:
                merged[k] = v

    return merged


def parse_simulation_parameters_file(path, directions, sim_num):
    path = Path(path)
    lines = path.read_text().splitlines()

    ar_keys = ["Simulation Number"] + directions
    ar_dict = {k: [] for k in ar_keys}
    frame: int | None = None
    for line in lines:
        if line.startswith("Size of crystal at frame output"):
            frame = lines.index(line) + 1
    if frame is None:
        raise ValueError(
            f"The line 'Size of crystal at frame output' was not found in the file\nFile: {path}."
        )
    ar_dict["Simulation Number"].append(sim_num + 1)
    for len_line in lines[frame:]:
        for direction in directions:
            if len_line.startswith(direction):
                ar_dict[direction].append(float(len_line.split()[-2]))
    return ar_dict


def populate_aspect_ratios_for_selected_columns(df, selected):
    for i in range(len(selected) - 1):
        LOG.debug("Aspect Ratio [%s] : [%s] : [%s]", i, selected[i], selected[i + 1])
        df[f"Ratio_{selected[i]}:{selected[i + 1]}"] = df[selected[i]] / df[selected[i + 1]]


def build_cda(folders, folderpath, savefolder, directions, selected, singals=None):
    path = Path(folderpath)

    ar_keys = ["Simulation Number"] + directions
    LOG.debug("AR_keys %s", ar_keys)

    simulation_parameters_files = []
    # find all simulation_parameters.txt files
    for folder in folders:
        for f in Path(folder).iterdir():
            if f.name.startswith("._"):
                continue
            elif f.name.endswith("simulation_parameters.txt"):
                simulation_parameters_files.append(f)

    ar_dict = merge_dicts(
        [
            parse_simulation_parameters_file(f, directions, i)
            for i, f in enumerate(simulation_parameters_files)
        ]
    )

    ar_dict = treat_inconsistent_dict(ar_dict)
    print_keys_and_value_lengths(ar_dict)

    LOG.debug("CDA data frame dictionary:\n%s", ar_dict)
    print(ar_dict)

    try:
        df = pd.DataFrame.from_dict(ar_dict)
    except ValueError as v:
        LOG.error(
            "Potential corrupted input files. Please check "
            "if simulation_parameters.txt files has all the information.\n%s",
            v,
        )

    populate_aspect_ratios_for_selected_columns(df, selected)

    return df


def get_cda_shape_percentage(df: pd.DataFrame, savefolder: str | Path):
    shape_columns = ["Shape"]
    cda_shape_data = []
    savefolder = Path(savefolder)

    for shape_column in shape_columns:
        grouped = df.groupby(["CDA_Permutation", shape_column]).size().reset_index(name="Count")
        for cda_permutation, group in grouped.groupby("CDA_Permutation"):
            for shape, count in zip(group[shape_column], group["Count"]):
                source = "OBA" if shape_column == "OBA Shape" else "PCA"
                cda_shape_data.append(
                    {
                        "CDA_Permutation": cda_permutation,
                        "Shape": shape,
                        "Count": count,
                        "Source": source,
                    }
                )

    cda_shape_df = pd.concat([pd.DataFrame([data]) for data in cda_shape_data], ignore_index=True)
    cda_shape_df.sort_values("CDA_Permutation", inplace=True)
    cda_shape_df.to_csv(savefolder / "shapes_permutations.csv")


def build_ratio_equations(directions, ar_df=None, csv=None, filepath: str | Path = "."):
    """Defining CDA aspect ratio equations depending on the selected directions from the gui.
    This means we will also need to input the selected directions into the function.

    Permutations:   a b c
                    a c b
                    b a c
                    b c a
                    c a b
                    c b a
    """
    filepath = Path(filepath)
    if ar_df is None:
        if csv is None:
            LOG.error("A pandas DataFrame or a path to a CSV file must be provided.")
        ar_df = pd.read_csv(Path(csv), encoding="utf-8", encoding_errors="replace")
    if not isinstance(ar_df, pd.DataFrame):
        LOG.error("Dataframe not found AR_DF: %s CSV: ", type(ar_df), type(csv))
    if len(directions) != 3:
        raise ValueError("Directions should be a list of three elements.")

    d_perms = list(permutations(directions))

    # Write the permutations to a file
    with open(Path(filepath) / "cda_equations.txt", "w") as outfile:
        for i, perm in enumerate(d_perms, 1):
            outfile.write(f"CDA Permutation Number {i}: {perm}\n")

    # Iterate over the permutations and dynamically assign number
    for i, perm in enumerate(d_perms, 1):
        condition = (ar_df[perm[0]] <= ar_df[perm[1]]) & (ar_df[perm[1]] <= ar_df[perm[2]])
        ar_df.loc[condition, "CDA_Permutation"] = str(i)

    return ar_df


def get_xyz_shape_percentage(df, savefolder):
    """
    Calculates the frequency
    that lath, needle, plate and block are found in
    the Zingg Shape column to create a csv
    that shows the percentage of each shape found.
    """
    zingg_shape_counts = df["Shape"].str.lower().value_counts()
    zingg_total_count = zingg_shape_counts.sum()
    zingg_shape_percentages = zingg_shape_counts / zingg_total_count * 100

    result_df = pd.DataFrame(
        {
            "Shape": zingg_shape_counts.index,
            "Count": zingg_shape_counts,
            "Percentage": zingg_shape_percentages,
        }
    )
    total_shapes_csv = f"{savefolder}/shape_counts.csv"
    result_df.to_csv(total_shapes_csv, index=False)


def collect_all(folder: Path = None, xyz_files: list[Path] = None, signals=None):
    """
    This collects all the crystal shape
    information from each of the relevant functions
    and congregates that into the final DataFrame
    """

    col_headings = [
        "Simulation Number",
        "Frame Index",
        "PC1",
        "PC2",
        "PC3",
        "Length X",
        "Length Y",
        "Length Z",
        "S:M",
        "M:L",
        "Shape",
        "Surface Area",
        "Volume",
        "SA:Vol Ratio",
    ]

    if xyz_files is None:
        if folder is None:
            LOG.error(
                "Received no folder or xyz filesFolder: %s XYZ-files: %s",
                folder,
                xyz_files,
            )
            return
        LOG.warning(
            "XYZ files were not passed to calculations threadCurrently being read from %s",
            folder,
        )

        def not_empty(x):
            return x.stat().st_size > 0

        xyz_files = [filepath for filepath in Path(folder).rglob("*.XYZ") if not_empty(filepath)]

    if not isinstance(xyz_files, list) or not all(isinstance(item, Path) for item in xyz_files):
        LOG.error("XYZ files as type %s expected, got %s", type(list[Path]), type(xyz_files))

    n_xyzs = len(xyz_files)
    if n_xyzs < 1:
        LOG.error("No XYZ files found in directory/subdirectories of [%s]", str(folder))
        return

    # List for collecting data
    data_list = []
    i = 1
    shape_analyser = ShapeAnalyser()

    for file in xyz_files:
        if signals is not None and signals.cancel_flag.is_set():
            LOG.info("Aspect ratio analysis cancelled after %d / %d files.", i - 1, n_xyzs)
            signals.cancelled.emit()
            return None
        sim_num = simulation_id_from_path(file)
        try:
            crystal = CrystalCloud.from_file(file)
            shape_analyser.analyse_crystal(crystal, frame_idx=None)

            for frame_idx, metrics in shape_analyser.get_all_frame_metrics().items():
                if metrics is None:
                    continue
                data_row = [
                    sim_num,
                    frame_idx,
                    metrics.pc1,
                    metrics.pc2,
                    metrics.pc3,
                    metrics.x,
                    metrics.y,
                    metrics.z,
                    metrics.aspect1,
                    metrics.aspect2,
                    metrics.shape,
                    metrics.surface_area,
                    metrics.volume,
                    metrics.surface_area_to_volume_ratio,
                ]
                data_list.append(data_row)
            if signals:
                signals.progress.emit(int((i / n_xyzs) * 100))
            i += 1
        except (StopIteration, UnicodeDecodeError):
            continue

    # Convert data to a DataFrame if not empty
    if data_list:
        print(col_headings)
        df = pd.DataFrame(data_list, columns=col_headings)
        return df
    else:
        LOG.warning("Couldn't create Aspect Ratio Dataframe.Please check the XYZ files provided.")
        return


def print_keys_and_value_lengths(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, list):
            l = len(value)
        elif isinstance(value, (int, str, float)):
            l = 1
        else:
            raise TypeError(
                "Expected a list of values or a single value of type int, str, or float."
                f" Recieved: {value} Type[{value.__class__}]"
            )

        LOG.info(f"{key}: {l}")


def treat_inconsistent_dict(input_dict: dict):
    """
    Attempts to recover the last length
    from a dictionary that is potentially a result
    of multiple lenght entries (a movie)

    Parameters:
        - input_dict (dict): Dictionary with direction lengths

    Raises:
        ValueError: When inconsistency is not due to multiple frames
    """

    index_col = None
    for col in input_dict.keys():
        if col.lower() in {"simulation number" or "solvent"}:
            index_col = col

    if index_col is None:
        index_col = input_dict.keys()[0]
        LOG.warning(
            "'Simulation Number' or 'Solvent' not found as indexing columns. "
            "Using the first column: %s",
            index_col,
        )

    nsims = len(input_dict[index_col])

    last_values = dict()
    for key, values in input_dict.items():
        if key == index_col:
            last_values[key] = values
        elif key != index_col and len(values) % nsims == 0:
            last_values[key] = values[-1]
        else:
            raise ValueError(
                "Found inconsistencies in the AR dictionary."
                f" Length of values ({len(values)}) not "
                f"divisioble by the number of simulations found: {nsims}"
            )
    return last_values

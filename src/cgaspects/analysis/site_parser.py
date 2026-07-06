"""
Parser for crystallisation events and population CSV files.

This module provides functionality to parse the complex CSV structure of
*_crystallisation_events.csv and *_populations.csv files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("CA:SiteParser")


def parse_site_csv(csv_path: Path) -> Dict:
    """
    Parse a crystallisation events or populations CSV file.

    The CSV has a complex structure where:
    - First three columns contain: supersaturation, time, iterations
    - Row 1: sitenumbers (site IDs as column headers)
    - Row 2: tile type (per site)
    - Row 3: energies (per site)
    - Row 4: grown(1) ungrown(0) - occupation status
    - Row 5: coordination (per site)
    - Row 6: (empty headers row)
    - Row 7: TOTAL EVENTS or TOTAL POPULATION (per site)
    - Rows 8+: time series data (supersaturation, time, iterations in first 3 cols,
               then event/population values per site)

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with structure:
        {
            'supersaturation': numpy array of float values,
            'time': numpy array of float values,
            'iterations': numpy array of int values,
            'file_type': 'events' or 'population',
            'sites': {
                site_number (int): {
                    'tile_type': int or None,
                    'energy': float or None,
                    'occupation': bool or None,
                    'coordination': int or None,
                    'total_events': int or None (total events if events file),
                    'total_population': int or None (total population if population file),
                    'events': numpy array or None (growth events time series),
                    'population': numpy array or None (population time series)
                },
                ...
            }
        }

        Note: sites is a dictionary where keys are site numbers (int) and values
        are dictionaries containing the site data.
    """
    logger.debug(f"Parsing site CSV: {csv_path}")

    # Read the entire CSV without headers to handle the complex structure
    df = pd.read_csv(
        csv_path, header=None, encoding="utf-8", encoding_errors="replace", low_memory=False
    )

    # Initialize the result dictionary
    result = {
        "supersaturation": None,
        "time": None,
        "iterations": None,
        "file_type": None,
        "sites": {},
    }

    # The structure is fixed, so we can use direct indexing
    # Row 0: sitenumbers header + site numbers
    # Row 1: tile type header + values
    # Row 2: energies header + values
    # Row 3: grown(1) ungrown(0) header + values
    # Row 4: coordination header + values
    # Row 5: Empty row with supersaturation, time, iterations headers
    # Row 6: TOTAL EVENTS/TOTAL POPULATION header + totals
    # Row 7+: Time series data

    # Find the row with site numbers
    site_numbers_idx = None
    tile_type_idx = None
    energies_idx = None
    occupation_idx = None
    coordination_idx = None
    total_idx = None
    data_row_idx = None

    headers = df.iloc[:7, 3]

    for idx, value in enumerate(headers):
        if pd.isna(value):
            continue
        val_str = str(value).strip().lower()

        if "sitenumbers" in val_str or "site numbers" in val_str:
            site_numbers_idx = idx
        elif "tile type" in val_str or "tile_type" in val_str:
            tile_type_idx = idx
        elif "energies" in val_str or "energy" in val_str:
            energies_idx = idx
        elif "grown" in val_str and "ungrown" in val_str:
            occupation_idx = idx
        elif "coordination" in val_str:
            coordination_idx = idx
        elif "total" in val_str and ("events" in val_str or "population" in val_str):
            total_idx = idx
            # Detect file type from the total row header
            if "events" in val_str:
                result["file_type"] = "events"
            elif "population" in val_str:
                result["file_type"] = "population"
            # Data starts after the total row
            data_row_idx = idx + 1
            break

    if site_numbers_idx is None:
        logger.error(f"Could not find site numbers row in {csv_path}")
        return result

    supersat_val = df.iloc[data_row_idx:, 0]
    time_val = df.iloc[data_row_idx:, 1]
    iter_val = df.iloc[data_row_idx:, 2]

    if supersat_val.notna().any():
        result["supersaturation"] = supersat_val.to_numpy(dtype=float)

    if time_val.notna().any():
        result["time"] = time_val.to_numpy(dtype=float)

    if iter_val.notna().any():
        result["iterations"] = iter_val.to_numpy(dtype=int)

    # Extract site numbers from the header row (starting from column 3)
    # Columns 0-2 are supersaturation, time, iterations
    numerical_block = df.iloc[:5, 4:]
    sites = result["sites"]
    # row that contains the site numbers
    site_numbers = numerical_block.iloc[site_numbers_idx]

    for col_idx, site_num in site_numbers.items():
        if pd.isna(site_num):
            continue

        site_num = int(site_num)

        sites[site_num] = {
            "tile_type": None,
            "energy": None,
            "occupation": None,
            "coordination": None,
            "total_events": None,
            "total_population": None,
            "events": None,
            "population": None,
        }

        if tile_type_idx is not None:
            val = df.iloc[tile_type_idx, col_idx]
            if pd.notna(val):
                sites[site_num]["tile_type"] = int(val)

        if energies_idx is not None:
            val = df.iloc[energies_idx, col_idx]
            if pd.notna(val):
                sites[site_num]["energy"] = float(val)

        if occupation_idx is not None:
            val = df.iloc[occupation_idx, col_idx]
            if pd.notna(val):
                sites[site_num]["occupation"] = bool(int(val))

        if coordination_idx is not None:
            val = df.iloc[coordination_idx, col_idx]
            if pd.notna(val):
                sites[site_num]["coordination"] = int(val)

        if total_idx is not None:
            val = df.iloc[total_idx, col_idx]
            if pd.notna(val):
                total_value = int(val)
                if result["file_type"] == "events":
                    sites[site_num]["total_events"] = total_value
                elif result["file_type"] == "population":
                    sites[site_num]["total_population"] = total_value

        if data_row_idx is not None:
            val = df.iloc[data_row_idx:, col_idx]
            if pd.notna(val).any():
                data_array = val.to_numpy(dtype=int)
                if result["file_type"] == "events":
                    sites[site_num]["events"] = data_array
                elif result["file_type"] == "population":
                    sites[site_num]["population"] = data_array

    logger.debug(f"Parsed {len(result['sites'])} sites from {csv_path.name}")
    logger.debug(
        f"Global parameters - Supersaturation points: {len(result['supersaturation'])}, "
        f"Time points: {len(result['time'])}, Iterations: {len(result['iterations'])}"
    )

    return result


def parse_count(f: Path) -> dict[int, dict[int, int]]:
    """
    Parse a count file to extract interaction information per site.

    Args:
        f: Path to the count file

    Returns:
        Dictionary mapping site numbers to their interaction dictionaries.
        Each interaction dictionary maps interaction IDs to their frequencies.
    """
    count_info: dict[int, dict[int, int]] = dict()

    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(("empty", "grown")):
                continue

            tokens = line.split("coord")[0].split()
            site = int(tokens[3])

            interactions: dict[int, int] = dict()
            for tok in tokens:
                if "(" in tok and ")" in tok:
                    freq, rest = tok.split("(")
                    inter_id = rest.rstrip(")")
                    interactions[int(inter_id)] = int(freq)

            count_info[site] = interactions

    return count_info


def merge_interactions(sites: dict, interactions: dict[int, dict[int, int]]):
    """
    Merge interaction data into site dictionaries.

    Args:
        sites: Dictionary of site data (site_number -> site_data dict)
        interactions: Dictionary of interactions (site_number -> interaction_dict)
    """
    for site_str, site_data in sites.items():
        site = int(site_str)
        site_data.update({"interactions": interactions.get(site, {})})


def extract_file_prefix(file_path: Path) -> str:
    """
    Extract the prefix from a site CSV filename.

    Removes '_crystallisation_events' or '_populations' suffix to get the base prefix.

    Args:
        file_path: Path to the CSV file

    Returns:
        The file prefix (e.g., 'run1' from 'run1_crystallisation_events.csv')
    """
    filename = file_path.stem  # Get filename without extension
    # Remove known suffixes
    for suffix in ["_crystallisation_events", "_populations"]:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    # If no known suffix, return the whole filename
    return filename


def merge_site_results(results_with_paths: List[tuple]) -> Dict[str, Dict]:
    """
    Merge site data from multiple CSV files by matching file prefixes.

    Files with the same prefix (e.g., 'run1_crystallisation_events.csv' and
    'run1_populations.csv') are merged together. Each site will have both
    events and population data if available.

    Args:
        results_with_paths: List of (parsed_result, file_path) tuples

    Returns:
        Dict of merged results, keyed by file prefix
    """
    from collections import defaultdict

    # Group results by prefix
    prefix_groups = defaultdict(list)
    for result, file_path in results_with_paths:
        prefix = extract_file_prefix(file_path)
        prefix_groups[prefix].append((result, file_path))

    merged_results = {}

    for prefix, group in prefix_groups.items():
        logger.debug(f"Merging {len(group)} file(s) with prefix '{prefix}'")

        # Initialize merged result with data from first file
        first_result, _ = group[0]
        merged = {
            "supersaturation": first_result["supersaturation"],
            "time": first_result["time"],
            "iterations": first_result["iterations"],
            "file_prefix": prefix,
            "source_files": [fp.name for _, fp in group],
            "sites": {},
        }

        # Collect all unique site numbers across all files in this group
        all_site_numbers = set()
        for result, _ in group:
            all_site_numbers.update(result["sites"].keys())

        # Merge data for each site
        for site_num in all_site_numbers:
            merged_site = {
                "tile_type": None,
                "energy": None,
                "occupation": None,
                "coordination": None,
                "total_events": None,
                "total_population": None,
                "events": None,
                "population": None,
                "interactions": None,
            }

            # Track if we've seen metadata for validation
            first_metadata = None

            for result, file_path in group:
                if site_num not in result["sites"]:
                    continue

                site_data = result["sites"][site_num]
                file_type = result["file_type"]

                # Get or set metadata (from first occurrence)
                if first_metadata is None:
                    first_metadata = {
                        "tile_type": site_data["tile_type"],
                        "energy": site_data["energy"],
                        "occupation": site_data["occupation"],
                        "coordination": site_data["coordination"],
                    }
                    # Set metadata in merged site
                    for key in first_metadata:
                        merged_site[key] = first_metadata[key]
                else:
                    # Validate metadata matches
                    for key in ["tile_type", "energy", "occupation", "coordination"]:
                        if (
                            site_data[key] is not None
                            and first_metadata[key] is not None
                            and site_data[key] != first_metadata[key]
                        ):
                            logger.warning(
                                f"Site {site_num} metadata mismatch for '{key}': "
                                f"{first_metadata[key]} != {site_data[key]} in {file_path.name}"
                            )

                # Merge type-specific data
                if file_type == "events":
                    if site_data["total_events"] is not None:
                        merged_site["total_events"] = site_data["total_events"]
                    if site_data["events"] is not None:
                        merged_site["events"] = site_data["events"]
                elif file_type == "population":
                    if site_data["total_population"] is not None:
                        merged_site["total_population"] = site_data["total_population"]
                    if site_data["population"] is not None:
                        merged_site["population"] = site_data["population"]

            merged["sites"][site_num] = merged_site

        logger.debug(
            f"Merged result for '{prefix}': {len(merged['sites'])} sites from {merged['source_files']}"
        )
        merged_results[prefix] = merged

    return merged_results


def parse_multiple_site_csvs(csv_paths: List[Path]) -> List[Dict]:
    """
    Parse multiple site CSV files.

    Args:
        csv_paths: List of paths to CSV files

    Returns:
        List of dictionaries, one for each CSV file
    """
    results = []
    for csv_path in csv_paths:
        try:
            result = parse_site_csv(csv_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error parsing {csv_path}: {e}", exc_info=True)

    return results


# Per-site colour fields exposed to the checkpoint viewer. Keys are the labels
# shown in the "Color By" combo; values pull a scalar out of a parsed site dict.
SITE_METADATA_FIELDS: dict[str, callable] = {
    "Coordination": lambda s: s.get("coordination"),
    "Energy": lambda s: s.get("energy"),
    "Events/Population": lambda s: (
        s.get("total_events")
        if s.get("total_events") is not None
        else s.get("total_population")
    ),
}


def build_site_metadata_maps(merged_results: Dict[str, Dict]) -> Dict[str, Dict[int, float]]:
    """Build {field_label: {site_number: value}} maps from merged site results.

    Site numbers are global across files, so sites from every prefix are merged
    into a single lookup per field (a later prefix wins on collision). Sites with
    a ``None`` value for a field are simply omitted from that field's map.
    """
    maps: Dict[str, Dict[int, float]] = {label: {} for label in SITE_METADATA_FIELDS}

    for result in merged_results.values():
        for site_num, site_data in result.get("sites", {}).items():
            site_num = int(site_num)
            for label, getter in SITE_METADATA_FIELDS.items():
                value = getter(site_data)
                if value is not None:
                    maps[label][site_num] = float(value)

    return maps


def load_site_metadata_maps(json_path: Path) -> Dict[str, Dict[int, float]]:
    """Load the saved site-analysis JSON and build per-field site→value maps."""
    import json

    with open(json_path, encoding="utf-8") as fh:
        merged_results = json.load(fh)

    return build_site_metadata_maps(merged_results)


def get_site_summary(parsed_data: Dict) -> Dict:
    """
    Generate a summary of the parsed site data.

    Args:
        parsed_data: Dictionary from parse_site_csv or merged result

    Returns:
        Summary dictionary with statistics
    """
    summary = {
        "total_sites": len(parsed_data["sites"]),
        "occupied_sites": sum(
            1 for site in parsed_data["sites"].values() if site.get("occupation")
        ),
        "unoccupied_sites": sum(
            1 for site in parsed_data["sites"].values() if not site.get("occupation")
        ),
        "time_points": len(parsed_data["time"]) if parsed_data["time"] is not None else 0,
        "iterations": len(parsed_data["iterations"])
        if parsed_data["iterations"] is not None
        else 0,
        "supersaturation_points": len(parsed_data["supersaturation"])
        if parsed_data["supersaturation"] is not None
        else 0,
        "tile_types": set(),
        "energy_range": [None, None],
        "coordination_range": [None, None],
        "sites_with_events": 0,
        "sites_with_population": 0,
        "sites_with_both": 0,
    }

    # Gather tile types, energy range, coordination range, and data availability
    energies = []
    coordinations = []
    for site_data in parsed_data["sites"].values():
        if site_data.get("tile_type") is not None:
            summary["tile_types"].add(site_data["tile_type"])
        if site_data.get("energy") is not None:
            energies.append(site_data["energy"])
        if site_data.get("coordination") is not None:
            coordinations.append(site_data["coordination"])

        # Count data availability
        has_events = site_data.get("events") is not None
        has_population = site_data.get("population") is not None

        if has_events:
            summary["sites_with_events"] += 1
        if has_population:
            summary["sites_with_population"] += 1
        if has_events and has_population:
            summary["sites_with_both"] += 1

    if energies:
        summary["energy_range"] = [min(energies), max(energies)]

    if coordinations:
        summary["coordination_range"] = [min(coordinations), max(coordinations)]

    summary["tile_types"] = sorted(list(summary["tile_types"]))

    return summary

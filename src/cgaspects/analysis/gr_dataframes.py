import logging
from typing import List
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.naming import simulation_id_from_path

logger = logging.getLogger("CA:GR-Dataframes")


def get_x_axis(df: pd.DataFrame, *, time_col: str = "time", tol: float = 1e-12):
    """Return a suitable x-axis for growth-rate fitting.

    Uses the time column if it has a sufficient span; otherwise falls back
    to the row index.
    """
    if time_col not in df.columns:
        logger.debug("Using row indices as x-axis (time column not found)")
        return np.arange(len(df), dtype=float)

    x_time = df[time_col].to_numpy(dtype=float)

    if not np.isfinite(x_time).all() or len(x_time) < 2 or np.ptp(x_time) < tol:
        logger.debug("Using row indices as x-axis (time column unsuitable)")
        return np.arange(len(df), dtype=float)

    logger.debug("Using time column as x-axis")
    return x_time


def build_growthrates(
    size_file_list: List[str | Path],
    supersat_list: List[float],
    directions: List[str],
    signals=None,
    time_tol: float = 1e-12,
    xaxis_mode: str = "auto",
):
    """Generate the growth rate dataframe from size.csv files.

    Parameters
    ----------
    xaxis_mode : str
        How to choose the x-axis for linear fitting:
        - ``"auto"``  – use the time column when present and valid,
          otherwise fall back to row index for all files.
        - ``"time"``  – always use the time column; files without a
          valid time column are skipped.
        - ``"index"`` – always use row index, ignoring any time column.
    """
    n_size_files = len(size_file_list)

    if n_size_files == 0:
        return None

    if len(supersat_list) != n_size_files:
        raise ValueError(
            f"supersat_list length ({len(supersat_list)}) does not match "
            f"size_file_list length ({n_size_files})"
        )

    logger.info("%s size files used to calculate growth rate data", n_size_files)
    logger.info("X-axis mode: %s", xaxis_mode)
    logger.info("Directions: %s", directions)

    growth_list = []
    kept_supersats = []
    sim_numbers = []

    # In "index" mode we never look at the time column
    use_index_for_all = xaxis_mode == "index"
    restart = True

    while restart:
        restart = False
        growth_list = []
        kept_supersats = []
        sim_numbers = []

        for i, f in enumerate(size_file_list):
            if signals is not None and signals.cancel_flag.is_set():
                logger.info("Growth rate analysis cancelled after %d / %d files.", i, n_size_files)
                signals.cancelled.emit()
                return None
            f = Path(f)
            lt_df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace")

            missing_directions = [d for d in directions if d not in lt_df.columns]
            if missing_directions:
                logger.warning(
                    "Skipping file %s: missing direction columns %s",
                    f.name,
                    missing_directions,
                )
                continue

            # Determine x-axis data
            if use_index_for_all:
                x_data = np.arange(len(lt_df), dtype=float)
            else:
                # Check if time column is suitable
                time_col = "time"
                if time_col not in lt_df.columns:
                    if xaxis_mode == "time":
                        logger.warning(
                            "Skipping file %s: time column not found (forced time mode)",
                            f.name,
                        )
                        continue
                    # auto mode: fall back to index for all files
                    logger.info(
                        "Time column missing in file %s - restarting with index for all files",
                        f.name,
                    )
                    use_index_for_all = True
                    restart = True
                    break
                else:
                    x_time = lt_df[time_col].to_numpy(dtype=float)
                    if (
                        not np.isfinite(x_time).all()
                        or len(x_time) < 2
                        or np.ptp(x_time) < time_tol
                    ):
                        if xaxis_mode == "time":
                            logger.warning(
                                "Skipping file %s: time column unsuitable (forced time mode)",
                                f.name,
                            )
                            continue
                        # auto mode: fall back to index for all files
                        logger.info(
                            "Time column unsuitable in file %s - restarting with index for all files",
                            f.name,
                        )
                        use_index_for_all = True
                        restart = True
                        break
                    else:
                        x_data = x_time

            sim_num = simulation_id_from_path(f)

            # Keep rows only up to the first row where any direction is 0
            all_positive = np.all(
                [np.asarray(lt_df[d], dtype=float) > 0 for d in directions],
                axis=0,
            )
            first_false = np.argmin(all_positive)
            # If all True, argmin returns 0 — check if the first element is actually True
            cutoff = first_false if not all_positive[first_false] else len(all_positive)
            mask = np.zeros(len(all_positive), dtype=bool)
            mask[:cutoff] = True

            gr_list = []
            for direction in directions:
                y_data = np.asarray(lt_df[direction], dtype=float)
                if mask.sum() < 2:
                    gr_list.append(0.0)
                    continue
                model = np.polyfit(x_data[mask], y_data[mask], 1)
                gr_list.append(model[0])

            growth_list.append(gr_list)
            kept_supersats.append(supersat_list[i])
            sim_numbers.append(sim_num)

            if signals:
                prog = (100 * (i + 1)) // n_size_files
                signals.progress.emit(prog)

    if not growth_list:
        logger.warning("No files were processed successfully")
        return None

    growth_array = np.asarray(growth_list, dtype=float)
    gr_df = pd.DataFrame(growth_array, columns=directions)
    gr_df.insert(0, "Simulation Number", sim_numbers)
    gr_df.insert(1, "Supersaturation", kept_supersats)

    return gr_df

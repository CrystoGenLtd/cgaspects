"""Cluster analysis module using KDTree nearest-neighbour connectivity on CrystalGrower XYZ files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler

from PySide6.QtCore import QThreadPool, Qt

from ..fileio.find_data import create_aspects_folder, summary_compare
from ..fileio.xyz_file import CrystalCloud
from ..gui.dialogs.cluster_dialog import ClusterAnalysisDialog
from ..utils.data_structures import cluster_options_tuple, results_tuple
from .gui_threads import WorkerClusters

logger = logging.getLogger("CA:Clusters")


# ---------------------------------------------------------------------------
# Core clustering helpers
# ---------------------------------------------------------------------------


def _cluster(
    coords: np.ndarray, eps: float, min_cluster_size: int, scale: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Connect points within *eps* via KDTree.

    Returns
    -------
    labels : np.ndarray (int)
        Per-particle cluster label; -1 = noise (component smaller than min_cluster_size).
    coord_numbers : np.ndarray (int32)
        Per-particle neighbour count within *eps*, reusing the same KDTree pairs.
    """
    X = StandardScaler().fit_transform(coords) if scale else coords
    n = len(X)
    pairs = KDTree(X).query_pairs(r=eps, output_type="ndarray")

    coord_numbers = np.zeros(n, dtype=np.int32)
    if len(pairs):
        np.add.at(coord_numbers, pairs[:, 0], 1)
        np.add.at(coord_numbers, pairs[:, 1], 1)
        rows, cols = pairs[:, 0], pairs[:, 1]
        data = np.ones(len(rows), dtype=np.float32)
        adj = csr_matrix(
            (np.tile(data, 2), (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
            shape=(n, n),
        )
    else:
        adj = csr_matrix((n, n), dtype=np.float32)

    _, comp = connected_components(adj, directed=False)
    sizes = np.bincount(comp, minlength=comp.max() + 1)
    labels = np.where(sizes[comp] < min_cluster_size, -1, comp.astype(int))
    # Renumber surviving clusters 0, 1, 2, …
    valid = labels >= 0
    if valid.any():
        _, labels[valid] = np.unique(labels[valid], return_inverse=True)
    return labels, coord_numbers


def _global_stats(labels: np.ndarray) -> dict:
    noise = labels == -1
    cl = labels[~noise]
    if len(cl) == 0:
        return {"n_clusters": 0, "avg_size": 0.0, "max_size": 0, "noise_frac": 1.0, "size_std": 0.0}
    _, counts = np.unique(cl, return_counts=True)
    return {
        "n_clusters": len(counts),
        "avg_size": float(counts.mean()),
        "max_size": int(counts.max()),
        "noise_frac": float(noise.sum() / len(labels)),
        "size_std": float(counts.std()),
    }


def analyse_frame(
    frame,
    eps: float,
    min_samples: int,
    scale: bool,
    downsample: float = 1.0,
    ratios_only: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run clustering on a Frame and return (metrics_dict, labels_array, coord_numbers_array).

    Parameters
    ----------
    downsample : float
        Fraction of particles to keep (0 < downsample <= 1.0).  A value of 1.0
        disables downsampling.  Particles are drawn with a fixed random seed so
        results are reproducible.
    ratios_only : bool
        If True, skip clustering entirely and only compute particle-type ratios.
    """
    types = frame.raw[:, 0].astype(int)
    unique_types = np.unique(types)
    total_particles = len(types)

    out: dict = {}

    # Pairwise ratios between types (type_i / type_j counts) — no clustering needed
    type_counts = {int(t): int((types == t).sum()) for t in unique_types}
    for t in unique_types:
        out[f"type{t}_ratio"] = float(type_counts[int(t)] / total_particles)
    for i, ti in enumerate(unique_types):
        for tj in unique_types[i + 1 :]:
            count_j = type_counts[int(tj)]
            out[f"type{ti}_to_type{tj}_ratio"] = (
                float(type_counts[int(ti)] / count_j) if count_j > 0 else float("inf")
            )

    if ratios_only:
        return out, np.array([]), np.array([])

    all_coords = frame.coords
    full_n = len(all_coords)

    # Downsample before clustering to speed up large frames.
    # Coord numbers are always computed on the full set so the array length
    # matches the displayed particle count in the viewport.
    if 0.0 < downsample < 1.0:
        rng = np.random.default_rng(seed=42)
        n_keep = max(1, round(full_n * downsample))
        idx = rng.choice(full_n, size=n_keep, replace=False)
        keep_mask = np.zeros(full_n, dtype=bool)
        keep_mask[idx] = True
        coords = all_coords[keep_mask]
        types = types[keep_mask]
        labels, _ = _cluster(coords, eps, min_samples, scale)
        # Build coord numbers on full set separately
        full_pairs = KDTree(all_coords).query_pairs(r=eps, output_type="ndarray")
        coord_numbers = np.zeros(full_n, dtype=np.int32)
        if len(full_pairs):
            np.add.at(coord_numbers, full_pairs[:, 0], 1)
            np.add.at(coord_numbers, full_pairs[:, 1], 1)
    else:
        labels, coord_numbers = _cluster(all_coords, eps, min_samples, scale)

    # Coordination-number aggregate stats
    out["coord_mean"] = float(coord_numbers.mean())
    out["coord_std"] = float(coord_numbers.std())
    out["coord_max"] = int(coord_numbers.max())
    out["coord_min"] = int(coord_numbers.min())

    # Global metrics
    for k, v in _global_stats(labels).items():
        out[f"global_{k}"] = v

    # Per-particle-type cluster metrics
    for t in unique_types:
        t_mask = types == t
        cl_ids = np.unique(labels[t_mask & (labels >= 0)])
        key = f"type{t}"
        if len(cl_ids) == 0:
            out.update(
                {
                    f"{key}_n_clusters": 0,
                    f"{key}_avg_size": 0.0,
                    f"{key}_max_size": 0,
                    f"{key}_noise_frac": 1.0,
                    f"{key}_size_std": 0.0,
                }
            )
        else:
            sizes = np.array([(t_mask & (labels == c)).sum() for c in cl_ids])
            n_noise = int((t_mask & (labels == -1)).sum())
            out.update(
                {
                    f"{key}_n_clusters": len(cl_ids),
                    f"{key}_avg_size": float(sizes.mean()),
                    f"{key}_max_size": int(sizes.max()),
                    f"{key}_noise_frac": n_noise / int(t_mask.sum()),
                    f"{key}_size_std": float(sizes.std()),
                }
            )

    # Mixed-cluster metrics
    all_cl = np.unique(labels[labels >= 0])
    if len(all_cl):
        n_types_per_cl = [len(np.unique(types[labels == c])) for c in all_cl]
        out["mixed_cluster_frac"] = sum(n > 1 for n in n_types_per_cl) / len(all_cl)
        out["avg_types_per_cluster"] = float(np.mean(n_types_per_cl))
    else:
        out["mixed_cluster_frac"] = 0.0
        out["avg_types_per_cluster"] = 0.0

    return out, labels, coord_numbers


def run_cluster_analysis(
    xyz_files: list,
    information,
    options: cluster_options_tuple,
    output_folder: Path,
    signals=None,
) -> tuple[Path | None, dict, dict]:
    """
    Run cluster analysis on all XYZ files.

    Returns
    -------
    csv_path : Path
        Path to the saved cluster_analysis.csv
    labels_cache : dict[str, np.ndarray]
        Mapping from str(xyz_path) → per-particle cluster label array.
    coord_cache : dict[str, np.ndarray]
        Mapping from str(xyz_path) → per-particle coordination-number array.
    """
    records = []
    labels_cache: dict[str, np.ndarray] = {}
    coord_cache: dict[str, np.ndarray] = {}
    total = len(xyz_files)

    for i, xyz_path in enumerate(xyz_files):
        if signals is not None and signals.cancel_flag.is_set():
            logger.info("Cluster analysis cancelled after %d / %d files processed.", i, total)
            signals.cancelled.emit()
            return None, {}, {}
        xyz_path = Path(xyz_path)
        try:
            frames = CrystalCloud.parse_xyz_file(xyz_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", xyz_path.name, e)
            continue

        if len(frames) == 0:
            logger.warning("No frames found in %s", xyz_path.name)
            continue

        frame_idx = options.frame_index
        if frame_idx == -1 or frame_idx >= len(frames):
            frame_idx = len(frames) - 1
        frame = frames[frame_idx]

        if len(frame.coords) == 0:
            logger.warning("Empty frame in %s", xyz_path.name)
            continue

        try:
            metrics, labels, coord_numbers = analyse_frame(
                frame,
                eps=options.eps,
                min_samples=options.min_samples,
                scale=options.scale,
                downsample=options.downsample,
                ratios_only=options.ratios_only,
            )
        except Exception as e:
            logger.warning("Clustering failed for %s: %s", xyz_path.name, e)
            continue

        metrics["Simulation Number"] = i + 1
        records.append(metrics)
        labels_cache[str(xyz_path)] = labels
        coord_cache[str(xyz_path)] = coord_numbers

        if signals is not None:
            progress = int((i + 1) / total * 80)
            signals.progress.emit(progress)

    if not records:
        raise RuntimeError("No XYZ files could be clustered.")

    if options.files_to_analyse is not None:
        if signals is not None:
            signals.progress.emit(100)
        return None, labels_cache, coord_cache

    if signals is not None:
        signals.progress.emit(85)

    cluster_df = pd.DataFrame(records)
    # Move Simulation Number to first column
    cols = ["Simulation Number"] + [c for c in cluster_df.columns if c != "Simulation Number"]
    cluster_df = cluster_df[cols]

    if signals is not None:
        signals.progress.emit(90)

    # Merge with summary file
    summary_file = information.summary_file if information is not None else None
    if summary_file and Path(summary_file).is_file():
        try:
            cluster_df = summary_compare(summary_csv=summary_file, aspect_df=cluster_df)
            logger.info("Merged cluster results with summary file: %s", summary_file)
        except Exception as e:
            logger.warning("summary_compare failed: %s", e)

    if signals is not None:
        signals.progress.emit(95)

    csv_path = output_folder / "cluster_analysis.csv"
    cluster_df.to_csv(csv_path, index=False)
    logger.info("Cluster analysis CSV saved: %s", csv_path)

    if signals is not None:
        signals.progress.emit(100)

    return csv_path, labels_cache, coord_cache


# ---------------------------------------------------------------------------
# ClusterAnalysis class — mirrors AspectRatio pattern
# ---------------------------------------------------------------------------


class ClusterAnalysis:
    def __init__(self, signals):
        self.input_folder: Path | None = None
        self.output_folder: Path | None = None
        self.information = None
        self.xyz_files: list[Path] = []
        self.current_file: Path | None = None
        self.options: cluster_options_tuple | None = None
        self.threadpool = QThreadPool()
        self.worker = None
        self.plotting_csv: Path | None = None
        self.labels_cache: dict[str, np.ndarray] = {}
        self.coord_cache: dict[str, np.ndarray] = {}
        self.signals = signals

        self.dialog = ClusterAnalysisDialog()
        self.dialog.runRequested.connect(self._run_analysis)

    def set_folder(self, folder):
        self.input_folder = Path(folder)

    def set_information(self, information):
        self.information = information

    def set_xyz_files(self, xyz_files: list[Path]):
        self.xyz_files = list(xyz_files)

    def set_current_file(self, path: Path | None):
        self.current_file = path

    def update_progress(self, value: int):
        self.signals.progress.emit(value)

    def get_location(self, location):
        self.output_folder = location
        self.signals.location.emit(location)

    def set_plotting(self, result):
        """Called by worker when done — result is (csv_path, labels_cache, coord_cache)."""
        csv_path, labels_cache, coord_cache = result
        self.plotting_csv = csv_path
        self.labels_cache.update(labels_cache)
        self.coord_cache.update(coord_cache)
        self.dialog.update_analysis_status(self.labels_cache, self.coord_cache, self.xyz_files)
        self.signals.finished.emit()
        r = results_tuple(csv=csv_path, selected=None, folder=self.output_folder)
        self.signals.result.emit(r)

    def calculate_clusters(self):
        if not self.xyz_files:
            logger.warning("No XYZ files set for cluster analysis.")
            return
        self.dialog.set_file_context(self.current_file, self.xyz_files)
        self.dialog.update_analysis_status(self.labels_cache, self.coord_cache, self.xyz_files)
        self.dialog.show()
        self.dialog.raise_()

    def _run_analysis(self, options: cluster_options_tuple):
        self.options = options

        # Respect file scope selection from the dialog
        if options.files_to_analyse is not None:
            xyz_files = [p for p in options.files_to_analyse if p in self.xyz_files or Path(p) in self.xyz_files]
            if not xyz_files:
                xyz_files = options.files_to_analyse
        else:
            xyz_files = self.xyz_files

        if self.output_folder is None:
            self.output_folder = create_aspects_folder(self.input_folder)
            self.signals.location.emit(self.output_folder)

        if self.threadpool:
            self.worker = WorkerClusters(
                information=self.information,
                options=self.options,
                input_folder=self.input_folder,
                output_folder=self.output_folder,
                xyz_files=xyz_files,
            )
            self.worker.signals.progress.connect(self.update_progress, Qt.QueuedConnection)
            self.worker.signals.result.connect(self.set_plotting, Qt.QueuedConnection)
            self.worker.signals.location.connect(self.get_location, Qt.QueuedConnection)
            self.worker.signals.cancelled.connect(self.signals.finished.emit, Qt.QueuedConnection)
            self.worker.signals.error.connect(self.signals.error.emit, Qt.QueuedConnection)
            self.signals.started.emit()
            self.threadpool.start(self.worker)
        else:
            logger.warning("Running cluster analysis on GUI thread (no threadpool).")
            self.run_on_same_thread()

    def run_on_same_thread(self):
        if self.output_folder is None:
            self.output_folder = create_aspects_folder(self.input_folder)
            self.signals.location.emit(self.output_folder)
        try:
            csv_path, labels_cache, coord_cache = run_cluster_analysis(
                xyz_files=self.xyz_files,
                information=self.information,
                options=self.options,
                output_folder=self.output_folder,
                signals=self.signals,
            )
            self.set_plotting((csv_path, labels_cache, coord_cache))
        except Exception as e:
            logger.error("Cluster analysis failed: %s", e)

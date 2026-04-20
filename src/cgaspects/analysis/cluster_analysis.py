"""Cluster analysis module using DBSCAN/OPTICS on CrystoGen XYZ files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtWidgets import QDialog

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
    coords: np.ndarray, algo: str, eps: float, min_samples: int, scale: bool
) -> np.ndarray:
    """Cluster *coords* and return integer label array (-1 = noise)."""
    X = StandardScaler().fit_transform(coords) if scale else coords
    if algo == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    if algo == "OPTICS":
        max_eps = eps if eps > 0 else np.inf
        return OPTICS(min_samples=min_samples, max_eps=max_eps).fit_predict(X)
    raise ValueError(f"Unknown clustering algorithm: {algo!r}")


def _global_stats(labels: np.ndarray) -> dict:
    noise = labels == -1
    cl = labels[~noise]
    if len(cl) == 0:
        return dict(
            n_clusters=0, avg_size=0.0, max_size=0, noise_frac=1.0, size_std=0.0
        )
    _, counts = np.unique(cl, return_counts=True)
    return dict(
        n_clusters=int(len(counts)),
        avg_size=float(counts.mean()),
        max_size=int(counts.max()),
        noise_frac=float(noise.sum() / len(labels)),
        size_std=float(counts.std()),
    )


def analyse_frame(
    frame,
    algo: str,
    eps: float,
    min_samples: int,
    scale: bool,
    downsample: float = 1.0,
    ratios_only: bool = False,
) -> tuple[dict, np.ndarray]:
    """Run clustering on a Frame and return (metrics_dict, labels_array).

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
        return out, np.array([])

    coords = frame.coords

    # Downsample before clustering to speed up large frames
    full_n = len(coords)
    if 0.0 < downsample < 1.0:
        rng = np.random.default_rng(seed=42)
        n_keep = max(1, int(round(full_n * downsample)))
        idx = rng.choice(full_n, size=n_keep, replace=False)
        keep_mask = np.zeros(full_n, dtype=bool)
        keep_mask[idx] = True
        coords = coords[keep_mask]
        types = types[keep_mask]

    labels = _cluster(coords, algo, eps, min_samples, scale)

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
                    f"{key}_n_clusters": int(len(cl_ids)),
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

    return out, labels


def run_cluster_analysis(
    xyz_files: list,
    information,
    options: cluster_options_tuple,
    output_folder: Path,
    signals=None,
) -> tuple[Path | None, dict]:
    """
    Run cluster analysis on all XYZ files.

    Returns
    -------
    csv_path : Path
        Path to the saved cluster_analysis.csv
    labels_cache : dict[str, np.ndarray]
        Mapping from str(xyz_path) → per-particle label array for the analysed frame.
    """
    records = []
    labels_cache: dict[str, np.ndarray] = {}
    total = len(xyz_files)

    for i, xyz_path in enumerate(xyz_files):
        if signals is not None and signals.cancel_flag.is_set():
            logger.info(
                "Cluster analysis cancelled after %d / %d files processed.", i, total
            )
            signals.cancelled.emit()
            return None, {}
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
            metrics, labels = analyse_frame(
                frame,
                algo=options.algorithm,
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

        if signals is not None:
            progress = int((i + 1) / total * 80)
            signals.progress.emit(progress)

    if not records:
        raise RuntimeError("No XYZ files could be clustered.")

    if signals is not None:
        signals.progress.emit(85)

    cluster_df = pd.DataFrame(records)
    # Move Simulation Number to first column
    cols = ["Simulation Number"] + [
        c for c in cluster_df.columns if c != "Simulation Number"
    ]
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

    return csv_path, labels_cache


# ---------------------------------------------------------------------------
# ClusterAnalysis class — mirrors AspectRatio pattern
# ---------------------------------------------------------------------------


class ClusterAnalysis:
    def __init__(self, signals):
        self.input_folder: Path | None = None
        self.output_folder: Path | None = None
        self.information = None
        self.xyz_files: list[Path] = []
        self.options: cluster_options_tuple | None = None
        self.threadpool = QThreadPool()
        self.worker = None
        self.plotting_csv: Path | None = None
        self.labels_cache: dict[str, np.ndarray] = {}
        self.signals = signals

    def set_folder(self, folder):
        self.input_folder = Path(folder)

    def set_information(self, information):
        self.information = information

    def set_xyz_files(self, xyz_files: list[Path]):
        self.xyz_files = list(xyz_files)

    def update_progress(self, value: int):
        self.signals.progress.emit(value)

    def get_location(self, location):
        self.output_folder = location
        self.signals.location.emit(location)

    def set_plotting(self, result):
        """Called by worker when done — result is (csv_path, labels_cache)."""
        csv_path, labels_cache = result
        self.plotting_csv = csv_path
        self.labels_cache = labels_cache
        r = results_tuple(csv=csv_path, selected=None, folder=self.output_folder)
        self.signals.finished.emit()
        self.signals.result.emit(r)

    def calculate_clusters(self):
        if not self.xyz_files:
            logger.warning("No XYZ files set for cluster analysis.")
            return

        dialog = ClusterAnalysisDialog()
        if dialog.exec() != QDialog.Accepted:
            logger.info("Cluster analysis cancelled by user.")
            return
        self.options = dialog.get_options()

        if self.output_folder is None:
            self.output_folder = create_aspects_folder(self.input_folder)
            self.signals.location.emit(self.output_folder)

        if self.threadpool:
            self.worker = WorkerClusters(
                information=self.information,
                options=self.options,
                input_folder=self.input_folder,
                output_folder=self.output_folder,
                xyz_files=self.xyz_files,
            )
            self.worker.signals.progress.connect(
                self.update_progress, Qt.QueuedConnection
            )
            self.worker.signals.result.connect(self.set_plotting, Qt.QueuedConnection)
            self.worker.signals.location.connect(self.get_location, Qt.QueuedConnection)
            self.worker.signals.cancelled.connect(
                self.signals.finished.emit, Qt.QueuedConnection
            )
            self.worker.signals.error.connect(
                self.signals.error.emit, Qt.QueuedConnection
            )
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
            csv_path, labels_cache = run_cluster_analysis(
                xyz_files=self.xyz_files,
                information=self.information,
                options=self.options,
                output_folder=self.output_folder,
                signals=self.signals,
            )
            self.set_plotting((csv_path, labels_cache))
        except Exception as e:
            logger.error("Cluster analysis failed: %s", e)

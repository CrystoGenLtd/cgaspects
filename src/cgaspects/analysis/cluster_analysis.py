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
    out["CN_mean"] = float(coord_numbers.mean())
    out["CN_std"] = float(coord_numbers.std())
    out["CN_max"] = int(coord_numbers.max())
    out["CN_min"] = int(coord_numbers.min())

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


# ---------------------------------------------------------------------------
# Radial profile (distance from origin) using site-analysis metadata
# ---------------------------------------------------------------------------

# Above this many distinct coordination/energy levels we skip the per-level
# proportion columns and keep only the shell mean, to avoid an unwieldy CSV.
RADIAL_MAX_LEVELS = 40


def radial_profile(
    coords: np.ndarray,
    site_numbers: np.ndarray | None,
    site_metadata: dict[str, dict[int, float]] | None,
    nbins: int,
    origin: np.ndarray | None = None,
) -> pd.DataFrame:
    """Radial profile of point density and site-metadata proportions.

    Points are binned into ``nbins`` shells by distance from *origin* (default
    (0, 0, 0), the CrystalGrower nucleation seed). Coordination number and
    energy are looked up per point from the site-analysis metadata maps using
    the point's site number — the same mapping the checkpoint viewer uses to
    colour by coordination/energy.

    Returns one row per shell (wide format). Besides the geometry/density
    columns, every distinct coordination number and energy level gets a
    ``coord_<k>_frac`` / ``energy_<v>_frac`` column giving the proportion of
    that shell's metadata-bearing points holding that value — one series per
    value, ready for a multi-line custom plot.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)

    dist = np.linalg.norm(coords - origin, axis=1) if n else np.zeros(0)
    r_max = float(dist.max()) if n else 0.0
    edges = np.linspace(0.0, r_max if r_max > 0 else 1.0, nbins + 1)
    r_lo, r_hi = edges[:-1], edges[1:]
    bin_idx = (
        np.clip(np.digitize(dist, edges) - 1, 0, nbins - 1)
        if n else np.zeros(0, dtype=int)
    )

    counts = np.bincount(bin_idx, minlength=nbins)
    total = int(counts.sum())
    shell_vol = (4.0 / 3.0) * np.pi * (r_hi ** 3 - r_lo ** 3)

    df = pd.DataFrame(
        {
            "r_lo": r_lo,
            "r_mid": 0.5 * (r_lo + r_hi),
            "r_hi": r_hi,
            "n_points": counts,
            "density": counts / total if total else np.zeros(nbins),
            "number_density": np.divide(
                counts, shell_vol, out=np.zeros(nbins), where=shell_vol > 0
            ),
        }
    )

    if site_numbers is None or not site_metadata or n == 0:
        return df

    sites = np.asarray(site_numbers, dtype=float).astype(int)

    def add_metric(label: str, mean_col: str, level_name) -> None:
        vmap = site_metadata.get(label)
        if not vmap:
            return
        vals = np.array([vmap.get(int(s), np.nan) for s in sites], dtype=float)
        has = ~np.isnan(vals)
        if not has.any():
            return

        # Denominator: points that carry a value for this metric, per shell —
        # so per-shell proportions sum to 1 regardless of missing metadata.
        valid_counts = np.bincount(bin_idx[has], minlength=nbins)
        sums = np.bincount(bin_idx[has], weights=vals[has], minlength=nbins)
        df[mean_col] = np.divide(
            sums, valid_counts, out=np.full(nbins, np.nan), where=valid_counts > 0
        )

        levels = np.unique(np.round(vals[has], 3))
        if len(levels) > RADIAL_MAX_LEVELS:
            logger.info(
                "Radial: %d distinct %s levels (> %d) — writing %s only",
                len(levels), label, RADIAL_MAX_LEVELS, mean_col,
            )
            return

        rounded = np.round(vals, 3)
        for v in levels:
            sel = has & (rounded == v)
            per_bin = np.bincount(bin_idx[sel], minlength=nbins)
            df[level_name(v)] = np.divide(
                per_bin, valid_counts, out=np.zeros(nbins), where=valid_counts > 0
            )

    # Coordination-number columns are named CN<k> (e.g. CN6); energy stays
    # energy_<v>_frac. Both hold the per-shell proportion of points at that level.
    add_metric("Coordination", "CN_mean", lambda v: f"CN{v:g}")
    add_metric("Energy", "energy_mean", lambda v: f"energy_{v:g}_frac")
    return df


def _write_radial_csv(radial_frames: list[pd.DataFrame], output_folder: Path) -> Path:
    """Concatenate per-file radial frames and write ``radial_analysis.csv``."""
    radial_df = pd.concat(radial_frames, ignore_index=True)
    # Different files can expose different coord/energy levels; a level absent
    # from a file means zero proportion there, so fill the aligned NaNs with 0.
    # Proportion columns are the per-level ones (CN<k> and energy_<v>_frac), not
    # the *_mean summaries which stay NaN where a shell has no data.
    frac_cols = [
        c for c in radial_df.columns
        if c.endswith("_frac") or (c.startswith("CN") and c != "CN_mean")
    ]
    if frac_cols:
        radial_df[frac_cols] = radial_df[frac_cols].fillna(0.0)
    radial_csv = output_folder / "radial_analysis.csv"
    radial_df.to_csv(radial_csv, index=False)
    logger.info("Radial analysis CSV saved: %s", radial_csv)
    return radial_csv


def _checkpoint_points(
    checkpoint_file: Path,
    n_tiles: int,
    crystallography,
    include_middle: bool = False,
    read_callback=None,
    expand_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a checkpoint grid to centred Cartesian points and their site numbers.

    Mirrors the checkpoint viewer: every occupied (cell, tile) becomes one point,
    the site number is the grid value there, and coordinates are frac→Cartesian
    then centred so the origin sits at the crystal centre.

    ``read_callback(current, total)`` tracks the file parse; ``expand_callback``
    reports the (vectorised) expansion at coarse stages so a GUI can show both.

    Returns ``(coords (M, 3), site_numbers (M,))``.
    """
    from ..fileio.cg_checkpoint import Checkpoint

    chk = Checkpoint.from_file(
        checkpoint_file, n_tiles, crysallography=crystallography,
        progress_callback=read_callback,
    )
    if include_middle or chk.edge_mask is None:
        data = chk.data
    else:
        data = np.where(chk.edge_mask[..., None], chk.data, 0)

    occupied = np.argwhere(data)  # (M, 4): i, j, k, tile
    if expand_callback is not None:
        expand_callback(1, 2)
    if not len(occupied):
        if expand_callback is not None:
            expand_callback(2, 2)
        return np.zeros((0, 3)), np.zeros(0, dtype=int)

    ijk = occupied[:, :3]
    tiles = occupied[:, 3]
    site_vals = data[ijk[:, 0], ijk[:, 1], ijk[:, 2], tiles].astype(int)
    coords = crystallography.frac_to_cart(ijk.astype(float))
    if len(coords):
        coords = coords - coords.mean(axis=0)  # centre → origin at crystal centre
    if expand_callback is not None:
        expand_callback(2, 2)
    return coords, site_vals


def _run_checkpoint_radial(
    files: list,
    options: cluster_options_tuple,
    output_folder: Path,
    signals,
    site_metadata,
    crystallography,
    n_tiles: int,
) -> Path | None:
    """Radial analysis over checkpoint files (KDTree clustering is skipped).

    Coordination/energy come from the site-analysis metadata, looked up per
    point by the site number stored in the checkpoint grid.
    """
    if crystallography is None or n_tiles is None:
        raise RuntimeError(
            "Checkpoint radial analysis needs a loaded structure "
            "(crystallography + tile count)."
        )

    radial_frames: list[pd.DataFrame] = []
    total = len(files)
    for i, path in enumerate(files):
        if signals is not None and signals.cancel_flag.is_set():
            signals.cancelled.emit()
            return None
        path = Path(path)
        if signals is not None:
            signals.message.emit(
                f"Reading checkpoint {i + 1}/{total}: {path.name}"
            )

        # Map this file's read (first 70%) and expansion (last 30%) into its
        # slice [base, base + span) of the overall 0–95% progress range.
        base = i / total * 95.0
        span = 95.0 / total

        def read_cb(cur, tot, _b=base, _s=span):
            if signals is not None and tot:
                signals.progress.emit(int(_b + (cur / tot) * _s * 0.7))

        def expand_cb(cur, tot, _b=base, _s=span):
            if signals is not None and tot:
                signals.progress.emit(int(_b + _s * 0.7 + (cur / tot) * _s * 0.3))

        try:
            coords, site_numbers = _checkpoint_points(
                path, n_tiles, crystallography,
                include_middle=getattr(options, "radial_include_middle", False),
                read_callback=read_cb, expand_callback=expand_cb,
            )
        except Exception as e:  # noqa: BLE001 - keep going through the batch
            logger.warning("Failed to load checkpoint %s: %s", path.name, e)
            continue
        if len(coords) == 0:
            logger.warning("No occupied cells in %s", path.name)
            continue

        prof = radial_profile(coords, site_numbers, site_metadata, options.radial_bins)
        prof.insert(0, "Simulation Number", i + 1)
        radial_frames.append(prof)

        if signals is not None:
            signals.progress.emit(int((i + 1) / total * 95))

    if not radial_frames:
        raise RuntimeError("No checkpoint files could be analysed.")

    radial_csv = _write_radial_csv(radial_frames, output_folder)
    if signals is not None:
        signals.progress.emit(100)
    return radial_csv


def run_cluster_analysis(
    xyz_files: list,
    information,
    options: cluster_options_tuple,
    output_folder: Path,
    signals=None,
    site_metadata: dict[str, dict[int, float]] | None = None,
    crystallography=None,
    n_tiles: int | None = None,
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
    radial = getattr(options, "radial", False)
    radial_source = getattr(options, "radial_source", "xyz")

    # Checkpoint radial mode: no XYZ clustering — expand the grid and profile it.
    if radial and radial_source == "checkpoint":
        radial_csv = _run_checkpoint_radial(
            xyz_files, options, output_folder, signals,
            site_metadata, crystallography, n_tiles,
        )
        return radial_csv, {}, {}

    records = []
    labels_cache: dict[str, np.ndarray] = {}
    coord_cache: dict[str, np.ndarray] = {}
    radial_frames: list[pd.DataFrame] = []
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

        # Radial profile from the XYZ point cloud: site numbers live in raw col 6,
        # coordination/energy come from the site-analysis metadata maps.
        if radial and not options.ratios_only:
            raw = frame.raw
            site_numbers = raw[:, 6] if raw.ndim == 2 and raw.shape[1] > 6 else None
            prof = radial_profile(
                frame.coords, site_numbers, site_metadata, options.radial_bins
            )
            prof.insert(0, "Simulation Number", i + 1)
            radial_frames.append(prof)

        if signals is not None:
            progress = int((i + 1) / total * 80)
            signals.progress.emit(progress)

    if not records:
        raise RuntimeError("No XYZ files could be clustered.")

    if radial and radial_frames:
        _write_radial_csv(radial_frames, output_folder)

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

        # Context for the radial site-metadata profile. site_metadata is the
        # {field: {site_number: value}} map from the site-analysis workflow;
        # crystallography/n_tiles are needed only for the checkpoint source.
        self.site_metadata: dict[str, dict[int, float]] = {}
        self.crystallography = None
        self.n_tiles: int | None = None

        self.dialog = ClusterAnalysisDialog()
        self.dialog.runRequested.connect(self._run_analysis)

    def set_folder(self, folder):
        self.input_folder = Path(folder)

    def set_information(self, information):
        self.information = information

    def set_xyz_files(self, xyz_files: list[Path]):
        self.xyz_files = list(xyz_files)

    def set_site_metadata(self, maps: dict[str, dict[int, float]] | None):
        self.site_metadata = maps or {}

    def set_checkpoint_context(self, crystallography, n_tiles: int | None):
        """Structure context (from the session) used for checkpoint radial mode."""
        self.crystallography = crystallography
        self.n_tiles = n_tiles

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
                site_metadata=self.site_metadata,
                crystallography=self.crystallography,
                n_tiles=self.n_tiles,
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
                site_metadata=self.site_metadata,
                crystallography=self.crystallography,
                n_tiles=self.n_tiles,
            )
            self.set_plotting((csv_path, labels_cache, coord_cache))
        except Exception as e:
            logger.error("Cluster analysis failed: %s", e)

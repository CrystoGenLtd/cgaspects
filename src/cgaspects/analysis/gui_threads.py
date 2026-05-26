import functools
import logging
import threading
from pathlib import Path
from typing import NamedTuple

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .ar_dataframes import (
    collect_all,
    get_xyz_shape_percentage,
    build_cda,
    build_ratio_equations,
    get_cda_shape_percentage,
)
from .gr_dataframes import build_growthrates
from ..fileio.find_data import (
    apply_supersat_mode,
    summary_compare,
    create_aspects_folder,
    combine_xyz_cda,
)
from .shape_analysis import ShapeAnalyser

logger = logging.getLogger("CA:Threads")


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals:
    finished
        No data
    error
        tuple (exctype, value, traceback.format_exc() )
    result
        object data returned from processing, anything
    progress
        int indicating % progress
    """

    started = Signal()
    finished = Signal()
    cancelled = Signal()
    error = Signal(tuple)
    result = Signal(object)
    location = Signal(object)
    progress = Signal(int)
    message = Signal(str)

    cancel_flag = None  # set to threading.Event() in __init__; class-level attr keeps it in spec

    def __init__(self):
        super().__init__()
        self.cancel_flag = threading.Event()


class CancellableRunnable(QRunnable):
    """QRunnable with a cooperative cancel flag stored on its WorkerSignals."""

    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def cancel(self):
        self.signals.cancel_flag.set()

    @property
    def is_cancelled(self) -> bool:
        return self.signals.cancel_flag.is_set()


def emit_error_on_exception(run_method):
    """Decorator: catches unhandled exceptions in a worker's run() and emits signals.error."""
    @functools.wraps(run_method)
    def wrapper(self, *args, **kwargs):
        try:
            return run_method(self, *args, **kwargs)
        except Exception as e:
            logger.error("%s error: %s", type(self).__name__, e, exc_info=True)
            self.signals.error.emit((type(e), e, str(e)))
            self.signals.finished.emit()
    return wrapper


class WorkerXYZ(CancellableRunnable):
    def __init__(self, centroids):
        super().__init__()
        self.centroids = centroids
        self.analyser = ShapeAnalyser()

    @Slot()
    def run(self):
        shape_info = self.analyser.shape_info(self.centroids)
        if self.is_cancelled:
            self.signals.cancelled.emit()
            self.signals.finished.emit()
            return
        if shape_info is None:
            self.signals.message.emit("Not enough points to calculate shape.")
            self.signals.result.emit(None)
            self.signals.finished.emit()
            return
        self.signals.progress.emit(100)
        self.signals.result.emit(shape_info)
        self.signals.message.emit("Calculations Complete!")
        self.signals.finished.emit()


class WorkerAspectRatios(CancellableRunnable):
    def __init__(
        self,
        information: NamedTuple,
        options: NamedTuple,
        input_folder: Path,
        output_folder: Path,
        xyz_files: list[Path],
    ):
        super().__init__()
        # Store constructor arguments (re-used for processing)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.information = information
        self.options = options
        self.xyz_files = xyz_files
        self.plotting_csv = None

    @emit_error_on_exception
    def run(self):
        self.output_folder = create_aspects_folder(self.input_folder)
        self.signals.location.emit(self.output_folder)
        summary_file = self.information.summary_file
        folders = self.information.folders

        if not (
            self.options.selected_ar
            or (
                self.options.selected_cda
                and self.options.checked_directions
                and self.options.selected_directions
            )
        ):
            logger.error("Condtions not met: AR AND/OR CDA (with checked AND selected directions)")
            return

        if self.options.selected_ar:
            xyz_df = collect_all(
                folder=self.input_folder, xyz_files=self.xyz_files, signals=self.signals
            )
            if xyz_df is None:
                # cancelled signal already emitted from inside collect_all
                self.signals.finished.emit()
                return
            xyz_combine = xyz_df
            if summary_file:
                xyz_df = summary_compare(summary_csv=summary_file, aspect_df=xyz_df)
            xyz_df_final_csv = self.output_folder / "aspectratio.csv"
            xyz_df.to_csv(xyz_df_final_csv, index=None)
            get_xyz_shape_percentage(df=xyz_df, savefolder=self.output_folder)
            logger.info("Plotting CSV created from: PCA/OBA")
            self.plotting_csv = xyz_df_final_csv

        if self.options.selected_cda and not self.options.checked_directions:
            logger.warning(
                "You have selected CDA option but have not checked any directions used to collect length information."
                "Please set this and try again!"
            )
            return
        if self.options.selected_cda and not self.options.selected_directions:
            logger.warning(
                "You have selected CDA option but have not set the three directions used for aspect ratio calculations."
                "Please set this and try again!"
            )
            return

        if self.options.selected_cda:
            cda_df = build_cda(
                folderpath=self.input_folder,
                folders=folders,
                directions=self.options.checked_directions,
                selected=self.options.selected_directions,
                savefolder=self.output_folder,
            )
            zn_df = build_ratio_equations(
                directions=self.options.selected_directions,
                ar_df=cda_df,
                filepath=self.output_folder,
            )
            if summary_file:
                zn_df = summary_compare(summary_csv=summary_file, aspect_df=zn_df)

            zn_df_final_csv = self.output_folder / "cda.csv"
            zn_df.to_csv(zn_df_final_csv, index=None)
            logger.info("Plotting CSV created from: CDA")
            self.plotting_csv = zn_df_final_csv
            if self.is_cancelled:
                self.signals.cancelled.emit()
                self.signals.finished.emit()
                return

            if self.options.selected_ar and self.options.selected_cda:
                combined_df = combine_xyz_cda(CDA_df=zn_df, XYZ_df=xyz_combine)
                final_cda_xyz_csv = self.output_folder / "crystalaspects.csv"
                combined_df.to_csv(final_cda_xyz_csv, index=None)
                get_cda_shape_percentage(df=combined_df, savefolder=self.output_folder)
                logger.info("Plotting CSV created from: CDA + PCA/OBA")
                self.plotting_csv = final_cda_xyz_csv

        if self.options.selected_solvent_screen:
            raise NotImplementedError("Currently in progress.")

        self.signals.result.emit(self.plotting_csv)


class WorkerGrowthRates(CancellableRunnable):
    def __init__(self, information, selected_directions, xaxis_mode="auto", supersat_mode="native"):
        super().__init__()
        self.information = information
        self.selected_directions = selected_directions
        self.xaxis_mode = xaxis_mode
        self.supersat_mode = supersat_mode

    @emit_error_on_exception
    def run(self):
        growth_rate_df = build_growthrates(
            size_file_list=self.information.size_files,
            supersat_list=self.information.supersats,
            directions=self.selected_directions,
            signals=self.signals,
            xaxis_mode=self.xaxis_mode,
        )

        logger.debug("build_growthrates returned: %s, shape=%s", type(growth_rate_df), getattr(growth_rate_df, "shape", None))

        if self.is_cancelled:
            # cancelled signal already emitted from inside build_growthrates
            self.signals.finished.emit()
            return

        if growth_rate_df is not None and not growth_rate_df.empty:
            summary_file = self.information.summary_file
            if summary_file:
                logger.info("Merging growth rates with summary file: %s", summary_file)
                growth_rate_df = summary_compare(
                    summary_csv=summary_file, aspect_df=growth_rate_df
                )
                growth_rate_df = apply_supersat_mode(growth_rate_df, self.supersat_mode)

        self.plotting_csv = growth_rate_df
        self.signals.result.emit(self.plotting_csv)


class WorkerSiteAnalysis(CancellableRunnable):
    def __init__(
        self,
        input_folder: Path,
        output_folder: Path,
        crystallisation_files: list[Path],
        population_files: list[Path],
        count_files: list[Path],
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.crystallisation_files = crystallisation_files
        self.population_files = population_files
        self.count_files = count_files

    def run(self):
        from .site_parser import (
            parse_site_csv,
            merge_site_results,
            get_site_summary,
            parse_count,
            merge_interactions,
            extract_file_prefix,
        )

        self.output_folder = create_aspects_folder(self.input_folder)
        self.signals.location.emit(self.output_folder)

        try:
            # Parse all CSV files
            results_with_paths = []

            # Calculate total number of files to process for progress tracking
            total_files = len(self.crystallisation_files) + len(self.population_files)
            files_processed = 0

            if self.crystallisation_files:
                logger.info(
                    f"Parsing {len(self.crystallisation_files)} crystallisation event files"
                )
                for csv_path in self.crystallisation_files:
                    if self.is_cancelled:
                        self.signals.cancelled.emit()
                        self.signals.finished.emit()
                        return
                    try:
                        self.signals.message.emit(f"Parsing crystallisation file: {csv_path.name}")
                        result = parse_site_csv(csv_path)
                        results_with_paths.append((result, csv_path))
                        files_processed += 1
                        progress = int((files_processed / total_files) * 80)  # Use 80% for parsing
                        self.signals.progress.emit(progress)
                    except Exception as e:
                        logger.error(f"Error parsing {csv_path}: {e}", exc_info=True)
                        files_processed += 1
                        progress = int((files_processed / total_files) * 80)
                        self.signals.progress.emit(progress)

            if self.population_files:
                logger.info(f"Parsing {len(self.population_files)} population files")
                for csv_path in self.population_files:
                    if self.is_cancelled:
                        self.signals.cancelled.emit()
                        self.signals.finished.emit()
                        return
                    try:
                        self.signals.message.emit(f"Parsing population file: {csv_path.name}")
                        result = parse_site_csv(csv_path)
                        results_with_paths.append((result, csv_path))
                        files_processed += 1
                        progress = int((files_processed / total_files) * 80)
                        self.signals.progress.emit(progress)
                    except Exception as e:
                        logger.error(f"Error parsing {csv_path}: {e}", exc_info=True)
                        files_processed += 1
                        progress = int((files_processed / total_files) * 80)
                        self.signals.progress.emit(progress)

            # Merge results by file prefix
            self.signals.message.emit("Merging results by file prefix...")
            self.signals.progress.emit(85)
            merged_results = merge_site_results(results_with_paths)

            # Parse and merge count files if available
            if self.count_files:
                self.signals.message.emit("Processing interaction count files...")
                self.signals.progress.emit(87)
                logger.info(f"Parsing {len(self.count_files)} count files")

                for count_file in self.count_files:
                    if self.is_cancelled:
                        self.signals.cancelled.emit()
                        self.signals.finished.emit()
                        return
                    try:
                        # Extract prefix from count file
                        count_filename = count_file.stem
                        if count_filename.endswith("_count"):
                            prefix = count_filename[:-6]  # Remove "_count"
                        elif count_filename == "count":
                            # Use parent directory name or try to match with available prefixes
                            prefix = count_file.parent.name
                            # If that doesn't match, try the first available prefix
                            if prefix not in merged_results and len(merged_results) == 1:
                                prefix = next(iter(merged_results))
                        else:
                            prefix = count_filename

                        # Parse the count file
                        interactions = parse_count(count_file)

                        # Merge interactions into the corresponding merged result
                        if prefix in merged_results:
                            merge_interactions(merged_results[prefix]["sites"], interactions)
                            logger.info(
                                f"Merged {len(interactions)} site interactions from {count_file.name} "
                                f"into prefix '{prefix}'"
                            )
                        else:
                            logger.warning(
                                f"Could not find matching prefix '{prefix}' for count file {count_file.name}. "
                                f"Available prefixes: {list(merged_results.keys())}"
                            )

                    except Exception as e:
                        logger.error(f"Error processing count file {count_file}: {e}", exc_info=True)

            # Generate summaries
            self.signals.message.emit("Generating summaries...")
            self.signals.progress.emit(90)
            for prefix, result in merged_results.items():
                if self.is_cancelled:
                    self.signals.cancelled.emit()
                    self.signals.finished.emit()
                    return
                summary = get_site_summary(result)
                logger.debug(
                    f"{prefix}: {summary['total_sites']} sites, "
                    f"{summary['occupied_sites']} occupied, "
                    f"{summary['time_points']} time points"
                )

            # Save parsed data summary to a file
            self.signals.message.emit("Saving summary file...")
            self.signals.progress.emit(95)
            summary_path = self.output_folder / "site_analysis_summary.txt"
            self._save_summary(merged_results, summary_path)

            # Save parsed data as JSON for plotting
            self.signals.message.emit("Saving JSON data for plotting...")
            self.signals.progress.emit(98)
            json_path = self.output_folder / "site_analysis_data.json"
            self._save_json(merged_results, json_path)

            logger.info(f"Site analysis complete. Results saved to {self.output_folder}")
            self.signals.message.emit("Site analysis complete!")
            self.signals.progress.emit(100)

            # Convert Path to string for signal emission
            logger.info(f"About to emit result signal with path: {json_path}")
            self.signals.result.emit(str(json_path))
            logger.info("Result signal emitted successfully")

            # Emit finished signal to indicate worker completion
            logger.info("Emitting finished signal from worker")
            self.signals.finished.emit()
            logger.info("Finished signal emitted from worker")

        except Exception as e:
            logger.error(f"Error during site analysis: {e}", exc_info=True)
            self.signals.error.emit((type(e), e, str(e)))

    def _save_summary(self, merged_results: dict[str, dict], output_path: Path):
        """Save a text summary of the merged parsed data."""
        from .site_parser import (
            get_site_summary,
        )

        with open(output_path, "w") as f:
            f.write("Site Analysis Summary\n")
            f.write("=" * 80 + "\n\n")

            for prefix, result in merged_results.items():
                summary = get_site_summary(result)
                source_files = result.get("source_files", [])

                f.write(f"Prefix: {prefix}\n")
                f.write(f"Source Files: {', '.join(source_files)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total sites: {summary['total_sites']}\n")
                f.write(f"Occupied sites: {summary['occupied_sites']}\n")
                f.write(f"Unoccupied sites: {summary['unoccupied_sites']}\n")
                f.write(f"Sites with events data: {summary.get('sites_with_events', 'N/A')}\n")
                f.write(
                    f"Sites with population data: {summary.get('sites_with_population', 'N/A')}\n"
                )
                f.write(f"Sites with both: {summary.get('sites_with_both', 'N/A')}\n")
                f.write(f"Time points: {summary['time_points']}\n")
                f.write(f"Iterations: {summary['iterations']}\n")
                f.write(f"Supersaturation points: {summary['supersaturation_points']}\n")
                f.write(f"Tile types: {summary['tile_types']}\n")
                f.write(f"Energy range: {summary['energy_range']}\n")
                f.write(f"Coordination range: {summary['coordination_range']}\n")
                f.write("\n")

                # Write site details
                f.write("Site Details (first 10 sites):\n")
                site_items = list(result["sites"].items())[:10]
                for site_num, site_data in site_items:
                    energy_str = (
                        f"{site_data['energy']:.2f}" if site_data["energy"] is not None else "None"
                    )
                    events_len = len(site_data["events"]) if site_data["events"] is not None else 0
                    pop_len = (
                        len(site_data["population"]) if site_data["population"] is not None else 0
                    )

                    f.write(f"  Site {site_num}:\n")
                    f.write(
                        f"    Metadata: tile_type={site_data['tile_type']}, "
                        f"energy={energy_str}, occupation={site_data['occupation']}, "
                        f"coordination={site_data['coordination']}\n"
                    )
                    f.write(
                        f"    Events: total={site_data['total_events']}, data_points={events_len}\n"
                    )
                    f.write(
                        f"    Population: total={site_data['total_population']}, "
                        f"data_points={pop_len}\n"
                    )

                if len(result["sites"]) > 10:
                    f.write(f"  ... and {len(result['sites']) - 10} more sites\n")

                f.write("\n" + "=" * 80 + "\n\n")

    def _save_json(self, merged_results: dict[str, dict], output_path: Path):
        """Save the merged parsed data as JSON for plotting."""
        import json
        import numpy as np

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        # Convert the results to JSON-serializable format
        serializable_results = convert_to_serializable(merged_results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Saved site analysis data to {output_path}")


class WorkerClusters(CancellableRunnable):
    def __init__(
        self,
        information,
        options,
        input_folder: Path,
        output_folder: Path,
        xyz_files: list[Path],
    ):
        super().__init__()
        self.information = information
        self.options = options
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.xyz_files = xyz_files

    @emit_error_on_exception
    def run(self):
        from .cluster_analysis import run_cluster_analysis

        if self.output_folder is None:
            self.output_folder = create_aspects_folder(self.input_folder)
        self.signals.location.emit(self.output_folder)
        self.signals.progress.emit(1)
        self.signals.message.emit(f"Starting cluster analysis on {len(self.xyz_files)} files…")

        try:
            csv_path, labels_cache, coord_cache = run_cluster_analysis(
                xyz_files=self.xyz_files,
                information=self.information,
                options=self.options,
                output_folder=self.output_folder,
                signals=self.signals,
            )
            single_file_mode = getattr(self.options, "files_to_analyse", None) is not None
            if csv_path is None and not single_file_mode:
                pass  # cancelled — signal already emitted from inside run_cluster_analysis
            else:
                self.signals.result.emit((csv_path, labels_cache, coord_cache))
        except Exception as e:
            logger.error("WorkerClusters failed: %s", e)
            self.signals.result.emit((None, {}, {}))
        finally:
            self.signals.finished.emit()

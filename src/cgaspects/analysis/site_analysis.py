"""
Site Analysis module for processing crystallisation events and population data.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QDialog

from .site_parser import (
    get_site_summary,
    merge_site_results,
    parse_multiple_site_csvs,
    parse_site_csv,
    parse_count,
    merge_interactions,
)
from .gui_threads import WorkerSiteAnalysis
from ..fileio import find_data as fd
from ..utils.data_structures import results_tuple

logger = logging.getLogger("CA:SiteAnalysis")


class SiteAnalysis:
    """
    Handler for site analysis calculations using crystallisation events
    and population CSV files.
    """

    def __init__(self, signals):
        self.input_folder = None
        self.output_folder = None
        self.information = None
        self.xyz_files: List[Path] | None = None
        self.signals = signals
        self.threadpool = QThreadPool()
        self.worker = None
        self.result_tuple = results_tuple
        self.plotting_csv = None

        self.crystallisation_files: List[Path] = []
        self.population_files: List[Path] = []
        self.count_files: List[Path] = []
        self.parsed_data = None

    def update_progress(self, value):
        """Emit progress update signal."""
        self.signals.progress.emit(value)

    def set_folder(self, folder):
        """Set the input folder for analysis."""
        self.input_folder = Path(folder)
        logger.info("Folder set for site analysis")

    def set_information(self, information):
        """Set the file information for analysis."""
        self.information = information
        logger.info("Information set for site analysis")

    def set_xyz_files(self, xyz_files: List[Path]):
        """Set the list of XYZ files."""
        self.xyz_files = xyz_files

    def set_site_files(
        self,
        crystallisation_files: List[Path],
        population_files: List[Path],
        count_files: List[Path],
    ):
        """Set the crystallisation events and population CSV files."""
        self.crystallisation_files = crystallisation_files
        self.population_files = population_files
        self.count_files = count_files
        logger.info(
            f"Set {len(crystallisation_files)} crystallisation files and "
            f"{len(population_files)} population files"
            f"{len(count_files)} count files"
        )

    def calculate_site_analysis(self):
        """
        Perform site analysis by parsing crystallisation events and population CSVs.
        """
        logger.debug("Called site analysis method at directory: %s", self.input_folder)

        if not self.input_folder:
            logger.warning("Input folder not set")
            return

        if self.information is None:
            logger.warning("Method called without information, looking for information now.")
            self.information = fd.find_info(self.input_folder)

        if not (self.crystallisation_files or self.population_files):
            logger.warning("No crystallisation events or population files found")
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(
                None,
                "No Site Data",
                "No crystallisation events or population CSV files were found in the directory.\n"
                "Please make sure the directory contains files matching:\n"
                "*_crystallisation_events.csv or *_populations.csv",
            )
            return

        logger.info("Attempting to perform Site Analysis...")

        # For now, we'll create a simple dialog to let users choose what to analyze
        # TODO: Create a proper dialog for site analysis options

        if self.threadpool:
            self.worker = WorkerSiteAnalysis(
                input_folder=self.input_folder,
                output_folder=self.output_folder,
                crystallisation_files=self.crystallisation_files,
                population_files=self.population_files,
                count_files=self.count_files,
            )
            # Use Qt.QueuedConnection for thread-safe signal delivery
            from PySide6.QtCore import Qt

            self.worker.signals.progress.connect(self.update_progress, Qt.QueuedConnection)
            self.worker.signals.result.connect(self.set_plotting, Qt.QueuedConnection)
            self.worker.signals.location.connect(self.get_location, Qt.QueuedConnection)
            self.worker.signals.finished.connect(self.on_worker_finished, Qt.QueuedConnection)
            self.signals.started.emit()
            self.threadpool.start(self.worker)

        else:
            logger.warning("Running Calculation on the same (GUI) thread!")
            self.run_on_same_thread()
            self.set_plotting(plotting_csv=self.plotting_csv)

    def on_worker_finished(self):
        """Handle worker thread completion."""
        logger.info("Worker thread finished signal received")

    def set_plotting(self, plotting_csv):
        logger.info(f"set_plotting called with csv: {plotting_csv}")
        result = self.result_tuple(csv=plotting_csv, selected=None, folder=self.output_folder)
        logger.info("Emitting finished signal")
        self.signals.finished.emit()
        logger.info(f"Emitting result signal with: {result}")
        self.signals.result.emit(result)
        logger.info("Plotting information sent to GUI successfully")

    def get_location(self, location):
        self.output_folder = location
        self.signals.location.emit(location)

    def run_on_same_thread(self):
        """Run site analysis on the same thread (fallback when threadpool is not available)."""
        self.output_folder = fd.create_aspects_folder(self.input_folder)
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
            self.parsed_data = merged_results

            # Parse and merge count files if available
            if self.count_files:
                self.signals.message.emit("Processing interaction count files...")
                self.signals.progress.emit(87)
                logger.info(f"Parsing {len(self.count_files)} count files")

                for count_file in self.count_files:
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
                summary = get_site_summary(result)
                logger.info(
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
            self.plotting_csv = json_path

        except Exception as e:
            logger.error(f"Error during site analysis: {e}", exc_info=True)
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None, "Site Analysis Error", f"An error occurred during site analysis:\n{str(e)}"
            )

    def _save_summary(self, merged_results: dict[str, dict], output_path: Path):
        """Save a text summary of the merged parsed data."""
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

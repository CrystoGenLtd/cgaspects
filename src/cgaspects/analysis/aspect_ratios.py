import logging
from collections import namedtuple
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, Signal
from PySide6.QtWidgets import QDialog, QWidget

from ..fileio.find_data import (combine_xyz_cda, create_aspects_folder,
                                find_info, summary_compare)
from ..gui.dialogs.aspectratio_dialog import AnalysisOptionsDialog
from ..plot.plot_data import Plotting
from ..utils.data_structures import ar_selection_tuple, results_tuple
from .ar_dataframes import (build_cda, build_ratio_equations, collect_all,
                            get_cda_shape_percentage, get_xyz_shape_percentage)
from .gui_threads import WorkerAspectRatios

logger = logging.getLogger("CA:A-Ratios")


class AspectRatio(QWidget):
    def __init__(self, signals):
        self.input_folder = None
        self.output_folder = None
        self.directions = None
        self.information = None
        self.xyz_files: list[Path] | None = None
        self.directions = None
        self.selected_direction = None
        self.options: namedtuple | None = None
        self.threadpool = None
        self.threadpool = QThreadPool()
        self.worker = None
        self.result_tuple = results_tuple

        self.signals = signals
        self.plotting_csv = None

    def update_progress(self, value):
        self.signals.progress.emit(value)

    def set_folder(self, folder):
        self.input_folder = Path(folder)
        logger.info("Folder set for aspect ratio calculations")

    def set_information(self, information):
        self.information = information
        logger.info("Information set for aspect ratio calculations")

    def set_xyz_files(self, xyz_files: list[Path]):
        self.xyz_files = xyz_files

    def calculate_aspect_ratio(self):
        logger.debug("Called aspect ratio method at directory: %s", self.input_folder)
        # Check if the user selected a folder
        if self.input_folder:
            if self.information is None:
                logger.warning(
                    "Method called without information, looking for information now."
                )
                # Read the information from the selected folder
                self.information = find_info(self.input_folder)
            logger.info("Attempting to calculate Aspect Ratios...")

            """Future Note: The follwing renders the 
            dataprocessor unable to process just XYZ files
            in the envent no *simulation_parameters.txt files are not present"""

        if not self.input_folder and self.information:
            logger.warning("Input folder and Information not set properly")
            return

        # Get the directions from the information
        self.directions = self.information.directions
        logger.debug("All Directions: %s", self.directions)

        if self.directions:

            # Create the analysis options dialog
            dialog = AnalysisOptionsDialog(directions=self.directions)

            if dialog.exec() != QDialog.Accepted:
                logger.warning("Selecting aspect ratio options cancelled")
                return

            # if dialog.exec() == QDialog.Accepted:
            # Retrieve the selected options
            self.options = dialog.get_options()
        else:

            self.options = ar_selection_tuple(True, False, [], [], False, False)

        logger.info(
            "Options:: AR: %s  CDA: %s  Checked: %s   Selected: %s   Auto Plot: %s",
            self.options.selected_ar,
            self.options.selected_cda,
            self.options.checked_directions,
            self.options.selected_directions,
            self.options.plotting,
        )

        # self.circular_progress = CircularProgress(calc_type="Aspect Ratio")
        # self.circular_progress.show()
        # self.circular_progress.raise_()
        # # Display the information in a QMessageBox
        # self.circular_progress.update_options(self.options)

        if self.threadpool:
            self.worker = WorkerAspectRatios(
                information=self.information,
                options=self.options,
                input_folder=self.input_folder,
                output_folder=self.output_folder,
                xyz_files=self.xyz_files,
            )
            self.worker.signals.progress.connect(self.update_progress, Qt.QueuedConnection)
            self.worker.signals.result.connect(self.set_plotting, Qt.QueuedConnection)
            self.worker.signals.location.connect(self.get_location, Qt.QueuedConnection)
            self.worker.signals.cancelled.connect(self.signals.finished.emit, Qt.QueuedConnection)
            self.worker.signals.error.connect(self.signals.error.emit, Qt.QueuedConnection)
            self.signals.started.emit()
            self.threadpool.start(self.worker)

        else:
            logger.warning("Running Calculation on the same (GUI) thread!")
            self.run_on_same_thread()
            self.set_plotting(plotting_csv=self.plotting_csv)

    def set_plotting(self, plotting_csv):
        result = self.result_tuple(csv=plotting_csv, selected=None, folder=None)
        self.signals.finished.emit()
        self.signals.result.emit(result)
        logger.debug("Sending plotting information to GUI: %s", result)
        self.plot(plotting_csv=plotting_csv)

    def plot(self, plotting_csv):
        # Generate the static plot files. The interactive PlottingDialog is
        # opened by the main window in response to the result signal emitted in
        # set_plotting() -- opening one here too would show the window twice.
        if self.options.plotting:
            self.perform_plotting(
                csv_file=plotting_csv,
                folderpath=self.output_folder,
                selected_directions=self.options.selected_directions,
            )

    def perform_plotting(self, csv_file, folderpath, selected_directions=None):
        plotting = Plotting()
        # Handle XYZ plotting
        if self.options.selected_ar:
            plotting.plot_zingg(csv=csv_file, folderpath=folderpath)
            plotting.plot_sa_vol(csv=csv_file, folderpath=folderpath)

        # Handle CDA plotting
        if self.options.selected_cda:
            plotting.plot_cda_extended(
                csv=csv_file, folderpath=folderpath, selected=selected_directions
            )

        # Handle combined CDA and XYZ plotting
        if self.options.selected_ar and self.options.selected_cda:
            plotting.plot_zingg_permuations(csv=csv_file, folderpath=folderpath)

    def get_location(self, location):
        self.output_folder = location
        self.signals.location.emit(location)

    def run_on_same_thread(self):
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
            logger.error(
                "Condtions not met: AR AND/OR CDA (with checked AND selected directions)"
            )
            return

        if self.options.selected_ar:
            xyz_df = collect_all(folder=self.input_folder, signals=self.signals)
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

            if self.options.selected_ar and self.options.selected_cda:
                combined_df = combine_xyz_cda(CDA_df=zn_df, XYZ_df=xyz_combine)
                final_cda_xyz_csv = self.output_folder / "crystalaspects.csv"
                combined_df.to_csv(final_cda_xyz_csv, index=None)
                get_cda_shape_percentage(df=combined_df, savefolder=self.output_folder)
                logger.info("Plotting CSV created from: CDA + PCA/OBA")
                self.plotting_csv = final_cda_xyz_csv

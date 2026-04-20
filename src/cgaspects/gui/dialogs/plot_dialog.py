import logging
import sys
from itertools import permutations
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cgaspects.gui.dialogs.axes_customization_dialog import AxesCustomizationDialog
from cgaspects.gui.dialogs.data_filter_dialog import DataFilterDialog
from cgaspects.gui.dialogs.plotsavedialog import PlotSaveDialog
from cgaspects.gui.dialogs.smoothing_dialog import SmoothingDialog
from cgaspects.gui.widgets.plot_axes_widget import PlotAxesWidget
from cgaspects.gui.widgets.time_series_widget import TimeSeriesWidget
from cgaspects.utils.data_smoothing import process_series
from cgaspects.utils.data_structures import plot_obj_tuple
from cgaspects.utils.plot_label import PlotAxisLabel, format_label as _format_label

matplotlib.use("QTAgg")

logger = logging.getLogger("CA:PlotDialog")


# Re-exported for backwards compatibility (e.g. existing tests import from here)
format_label = _format_label


class NavigationToolbar(NavigationToolbar2QT):
    toolitems = (
        ("Home", "Reset original view", "home", "home"),
        ("Back", "Back to previous view", "back", "back"),
        ("Forward", "Forward to next view", "forward", "forward"),
        (None, None, None, None),
        (
            "Pan",
            "Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect",
            "move",
            "pan",
        ),
        ("Zoom", "Zoom to rectangle\nx/y fixes axis", "zoom_to_rect", "zoom"),
    )


class PlottingDialog(QDialog):
    # Emitted after every replot with the current label objects so that any
    # open AxesCustomizationDialog can stay in sync.
    labels_updated = QtCore.Signal(dict)

    def __init__(self, csv, signals=None, parent=None, summary_df=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 800, 600)

        # Set the dialog to be non-modal
        self.setWindowModality(QtCore.Qt.NonModal)

        self.signals = signals
        self.summary_df = summary_df  # Summary file dataframe for mapping file prefixes to values
        self._axes_dialog = None  # persistent AxesCustomizationDialog instance
        self.annot = None
        self.trendline = None
        self.trendline_text = None

        self.selected_plot = None
        self.plot_obj_tuple = plot_obj_tuple
        self.plot_objects = {}
        self.plot_types = []
        self.directions = []
        self.permutations = []
        self.permutation_labels = []
        self.sa_permutation_labels = []
        self.interaction_columns = []
        self.gr_x_candidates: list = []
        self.gr_x_col: str = "Supersaturation"
        self.gr_grouped: bool = False
        self.site_analysis_columns = []  # Separate columns for site analysis mode
        self.trendline_text = []
        self.plotting_mode = "default"

        # User overrides — persist across replots; None = use auto-generated default
        self.custom_xlabel: PlotAxisLabel | None = None
        self.custom_ylabel: PlotAxisLabel | None = None
        self.custom_cbar_label: PlotAxisLabel | None = None
        self.custom_title: str = ""

        self.setPlotDefaults()
        self.create_widgets()
        self.create_layout()
        self.setCSV(csv)
        self.trigger_plot()

    def setPlotDefaults(self):
        self.grid = None
        self.cbar = None
        self.point_size = None
        self.plot_type = None
        self.permutation = None
        self.variable = None
        self.custom_x = None
        self.custom_y = None
        self.custom_c = None
        self.cluster_mat = None

        self.x_data = None
        self.y_data = None
        self.z_data = None
        self.gr_y_err = None
        self.c_data = None
        self.c_name = None
        self.c_label = PlotAxisLabel()
        self.x_label = PlotAxisLabel()
        self.y_label = PlotAxisLabel()
        self.title = ""

        # Colour scheme settings (universal across all modes)
        self.cmap = "viridis"

        # Data filtering
        self.data_filters = []  # List of filter configurations
        self.interaction_filters = {}  # Dictionary of interaction filter configurations

        # Smoothing/interpolation/extrapolation settings (for Growth Rates mode)
        self.smoothing_configs = {}  # Dictionary mapping series names to their configurations
        self.smoothing_legend_mode = "Show Both Original and Processed"  # Legend display mode

    def setCSV(self, csv):
        # Check if we're reloading the same CSV file
        csv_changed = not hasattr(self, "csv") or self.csv != csv

        self.csv = csv
        self.site_analysis_data = None

        if isinstance(self.csv, pd.DataFrame):
            self.df_original = self.csv.copy()
            self.df = self.csv
        else:
            # Check if this is a site analysis JSON file
            if str(csv).endswith("site_analysis_data.json"):
                import json

                with open(csv, "r") as f:
                    self.site_analysis_data = json.load(f)

                # For site analysis, we DON'T convert to DataFrame
                # Data will be extracted directly from the dictionary in _set_data()
                # Set df to None to indicate this is dict-based data
                self.df = None
                self.df_original = None

                # Populate site_analysis_columns for the variables dropdown
                self.site_analysis_columns = [
                    "None",
                    "tile_type",
                    "energy",
                    "occupation",
                    "coordination",
                    "total_events",
                    "total_population",
                    "file_prefix",
                    "filter",
                ]
                # Only clear time series widget file prefixes if the CSV file actually changed
                # This preserves the user's data source selection when replotting
                if csv_changed:
                    self.time_series_widget.file_prefixes = []
                logger.debug(
                    f"Loaded site analysis data with {len(self.site_analysis_data)} file prefixes"
                )
            else:
                self.df = pd.read_csv(self.csv, encoding="utf-8", encoding_errors="replace")
                self.df_original = self.df.copy()

        if self.df is not None:
            logger.debug("Dataframe read:\n%s", self.df)

        plotting = None
        self.growth_rate = None

        # Only process DataFrame columns if we have a DataFrame (not site analysis)
        if self.df is not None:
            for col in self.df.columns:
                if col.startswith("Supersaturation"):
                    plotting = "Growth Rates"

            self.interaction_columns = [
                col
                for col in self.df.columns
                if col.startswith(
                    (
                        "interaction",
                        "Int_",
                        "int_",
                        "tile",
                        "temperature",
                        "starting_delmu",
                        "excess",
                    )
                )
            ]
            self.interaction_columns.insert(0, "None")
            logger.debug("Interaction energy columns: %s", self.interaction_columns)
        else:
            # For site analysis (dict-based), interaction_columns is not used
            self.interaction_columns = []

        # List to store the results
        self.plot_types = []
        self.directions = []

        # Only check DataFrame columns if we have a DataFrame (not site analysis)
        if self.df is not None:
            # Check each column heading
            for column in self.df.columns:
                if column in ["PC1", "PC2", "PC3"] and "Zingg" not in self.plot_types:
                    self.plot_types.append("Zingg")
                if column.startswith("Ratio"):
                    column = column.replace("Ratio_", "")
                    directions = column.split(":")
                    for direction in directions:
                        if direction not in self.directions:
                            self.directions.append(direction)
                    if "CDA" not in self.plot_types:
                        self.plot_types.append("CDA")
                if column == "SA:Vol Ratio" and "SA:Vol Ratio" not in self.plot_types:
                    self.plot_types.append("SA:Vol Ratio")

            self.permutation_labels = ["None"]
            if len(self.directions) == 3:
                self.permutations = list(permutations(self.directions))
                for permutation in self.permutations:
                    p_str = f"({permutation[0].strip()}) : ({permutation[1].strip()}) : ({permutation[2].strip()})"
                    self.permutation_labels.append(p_str)

            if plotting == "Growth Rates":
                self.growth_rate = True
                self.directions = []
                for col in self.df.columns:
                    # Direction columns start with space or minus sign (Miller indices)
                    if col.startswith((" ", "-")):
                        self.directions.append(col)
                self.plot_types = ["Growth Rates"]

                # Build x-axis candidates: Supersaturation first, then all non-direction columns
                self.gr_x_candidates = []
                if "Supersaturation" in self.df.columns:
                    self.gr_x_candidates.append("Supersaturation")
                exclude = set(self.directions) | {"Simulation Number", "Supersaturation"}
                for col in self.df.columns:
                    if col not in exclude:
                        self.gr_x_candidates.append(col)

                # Auto-select default: Supersaturation if it varies, else first varying other col
                supersat_varies = (
                    "Supersaturation" in self.df.columns
                    and self.df["Supersaturation"].nunique() > 1
                )
                if not supersat_varies and len(self.gr_x_candidates) > 1:
                    self.gr_x_col = self.gr_x_candidates[1]
                else:
                    self.gr_x_col = "Supersaturation"

        # Check if this is site analysis data
        if self.site_analysis_data is not None:
            self.plot_types.append("Site Analysis")
            # Store SA-specific labels separately so they don't overwrite CDA permutation labels
            self.sa_permutation_labels = [
                "Events/Population vs Energy",
                "Sites vs Events/Population",
                "Events vs Population",
            ]

        logger.info("Default plot types found: %s", self.plot_types)
        logger.info("Directions found: %s", self.directions)

        # Add 4D mode when there are enough numeric columns
        if self.df is not None and len(self.df.select_dtypes(include="number").columns) >= 3:
            self.plot_types.append("4D")

        self.plot_types.append("Heatmap")
        self.plot_types.append("Custom")
        self.updatePlotWidgets()

        # Set default plot type: prioritize Site Analysis, then cluster data, otherwise index 0
        # Block signals to prevent trigger_plot being called during this default selection
        self.plot_types_combobox.blockSignals(True)
        if self.site_analysis_data is not None:
            site_analysis_index = self.plot_types_combobox.findText("Site Analysis")
            if site_analysis_index >= 0:
                self.plot_types_combobox.setCurrentIndex(site_analysis_index)
        elif self.df is not None and "global_n_clusters" in self.df.columns:
            heatmap_index = self.plot_types_combobox.findText("Heatmap")
            if heatmap_index >= 0:
                self.plot_types_combobox.setCurrentIndex(heatmap_index)
        self.plot_types_combobox.blockSignals(False)

    def updatePlotWidgets(self):
        # Block signals during widget updates to prevent multiple trigger_plot calls
        self.plot_permutations_combobox.blockSignals(True)
        self.plot_types_combobox.blockSignals(True)
        self.variables_combobox.blockSignals(True)

        self.custom_plot_widget.xAxisListWidget.clear()
        self.custom_plot_widget.yAxisListWidget.clear()
        self.custom_plot_widget.yAxisListWidget_single.clear()
        self.custom_plot_widget.colorListWidget.clear()
        self.plot_permutations_combobox.clear()
        self.plot_types_combobox.clear()
        self.variables_combobox.clear()

        self.plot_permutations_combobox.addItems(self.permutation_labels)
        self.plot_types_combobox.addItems(self.plot_types)

        # For site analysis, use site_analysis_columns; otherwise use interaction_columns
        if self.site_analysis_data is not None:
            self.variables_combobox.addItems(self.site_analysis_columns)
        else:
            self.variables_combobox.addItems(self.interaction_columns)

        # Only populate custom plot widgets if we have DataFrame data
        if self.df is not None:
            self.custom_plot_widget.xAxisListWidget.addItems(self.df.columns)
            self.custom_plot_widget.yAxisListWidget.addItems(self.df.columns)
            self.custom_plot_widget.yAxisListWidget_single.addItems(self.df.columns)
            self.custom_plot_widget.colorListWidget.addItems(["None"])
            self.custom_plot_widget.colorListWidget.addItems(self.df.columns)

        # Unblock signals after all updates are complete
        self.plot_permutations_combobox.blockSignals(False)
        self.plot_types_combobox.blockSignals(False)
        self.variables_combobox.blockSignals(False)

    def create_widgets(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.label_pointsize = QLabel("Point Size: ")

        # Main plotting classes
        self.plot_types_label = QLabel("Plot Type: ")
        self.plot_types_label.setAlignment(QtCore.Qt.AlignRight)
        self.plot_types_combobox = QComboBox(self)

        # if "CDA" in self.plot_types:
        self.plot_permutations_label = QLabel("Permutations: ")
        self.plot_permutations_label.setAlignment(QtCore.Qt.AlignRight)
        self.plot_permutations_label.setToolTip(
            "Crystallographic face-to-face distance ratios in ascending order"
        )
        self.plot_permutations_combobox = QComboBox(self)

        self.variables_label = QLabel("Variable: ")
        self.variables_label.setAlignment(QtCore.Qt.AlignRight)
        self.variables_combobox = QComboBox(self)

        self.custom_plot_widget = PlotAxesWidget()

        # Time-series widget for site analysis mode
        self.time_series_widget = TimeSeriesWidget()
        self.time_series_widget.hide()  # Hidden by default

        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 100)

        self.button_save = QPushButton("Export")
        self.exportIcon = QIcon(":material_icons/material_icons/png/content-save-custom.png")
        self.button_save.setIcon(self.exportIcon)
        self.button_add_trendline = QPushButton("Add Trendline")
        self.button_customize_labels = QPushButton("Customise Axes")
        self.button_filter_data = QPushButton("Filter Data")
        self.button_filter_data.setToolTip("Filter the data shown in the plot")
        self.button_smooth_data = QPushButton("Smooth/Extrapolate Data")
        self.button_smooth_data.setToolTip(
            "Apply smoothing, interpolation, and extrapolation to growth rate data"
        )
        self.button_smooth_data.hide()  # Hidden by default, only shown in Growth Rates mode

        self.button_reset_layout = QPushButton("Reset Layout")
        self.button_reset_layout.setToolTip(
            "Reset the figure layout (useful after switching to/from 4D mode)"
        )

        self.button_group_data = QPushButton("Group")
        self.button_group_data.setToolTip(
            "Group growth rates by X-Axis variable, averaging across all other variables with error bars"
        )
        self.button_group_data.hide()  # Hidden by default, only shown in Growth Rates mode

        # Initialize the variables
        self.point_size = 12
        self.spin_point_size.setValue(self.point_size)
        self.colorbar = False

        # Initialize checkboxes
        self.checkbox_grid = QCheckBox("Show Grid")
        self.checkbox_legend = QCheckBox("Show Legend")
        self.checkbox_zingg = QCheckBox("Zingg")
        self.checkbox_corr_mat = QCheckBox("Correlation Matix")
        self.checkbox_cluster_mat = QCheckBox("Hierarchical Clustering")
        self.checkbox_hide_bulk = QCheckBox("Hide Bulk Site")
        self.checkbox_hide_bulk.setChecked(True)  # Checked by default
        self.checkbox_hide_bulk.setToolTip("Hide the site with the highest population (bulk site)")

        # Colour scheme controls (universal across all modes)
        self.cmap_label = QLabel("Colour Scheme:")
        self.cmap_label.setAlignment(QtCore.Qt.AlignRight)
        self.cmap_combobox = QComboBox(self)
        self.cmap_combobox.addItems(
            [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "coolwarm",
                "seismic",
                "RdYlBu",
                "RdYlGn",
                "Spectral",
                "jet",
                "rainbow",
                "turbo",
            ]
        )

        self.canvas.mpl_connect("motion_notify_event", lambda event: self.on_hover(event))

        self.button_save.clicked.connect(self.save)
        self.button_customize_labels.clicked.connect(self.open_axes_customization_dialog)
        self.button_filter_data.clicked.connect(self.open_filter_dialog)
        self.button_smooth_data.clicked.connect(self.open_smoothing_dialog)
        self.button_group_data.clicked.connect(self.toggle_gr_grouping)
        self.checkbox_grid.stateChanged.connect(self.trigger_plot)
        self.checkbox_legend.stateChanged.connect(self.trigger_plot)
        self.checkbox_zingg.stateChanged.connect(
            lambda: self._handle_plot_checkboxes(self.checkbox_zingg)
        )
        self.checkbox_corr_mat.stateChanged.connect(
            lambda: self._handle_plot_checkboxes(self.checkbox_corr_mat)
        )
        self.checkbox_cluster_mat.stateChanged.connect(
            lambda: self._handle_plot_checkboxes(self.checkbox_cluster_mat)
        )
        self.checkbox_hide_bulk.stateChanged.connect(self.trigger_plot)
        self.button_add_trendline.clicked.connect(self.toggle_trendline)
        self.button_reset_layout.clicked.connect(self._reset_figure_layout)
        self.spin_point_size.valueChanged.connect(self.set_point_size)
        self.plot_types_combobox.currentIndexChanged.connect(self.trigger_plot)

        # Colour scheme controls
        self.cmap_combobox.currentIndexChanged.connect(self.trigger_plot)

        if self.plot_permutations_combobox is not None:
            self.plot_permutations_combobox.currentIndexChanged.connect(self.trigger_plot)
        if self.variables_combobox is not None:
            self.variables_combobox.currentIndexChanged.connect(self.trigger_plot)

        self.custom_plot_widget.xAxisListWidget.currentItemChanged.connect(self.trigger_plot)
        self.custom_plot_widget.yAxisListWidget.itemCheckedStateChanged.connect(self.trigger_plot)
        self.custom_plot_widget.yAxisListWidget_single.currentItemChanged.connect(self.trigger_plot)
        self.custom_plot_widget.colorListWidget.currentItemChanged.connect(self.trigger_plot)

        # Connect time-series widget signals
        self.time_series_widget.time_point_changed.connect(self.trigger_plot)
        self.time_series_widget.plotting_mode_changed.connect(self.trigger_plot)
        self.time_series_widget.file_prefix_changed.connect(self._on_file_prefix_changed)

        # Axis inversion checkboxes (all modes)
        self.checkbox_invert_x = QCheckBox("Invert X")
        self.checkbox_invert_y = QCheckBox("Invert Y")
        self.checkbox_invert_z = QCheckBox("Invert Z")
        self.checkbox_invert_z.setEnabled(False)
        self.checkbox_invert_x.stateChanged.connect(self.trigger_plot)
        self.checkbox_invert_y.stateChanged.connect(self.trigger_plot)
        self.checkbox_invert_z.stateChanged.connect(self.trigger_plot)

        # View angle sliders for 4D mode (hidden by default)
        self.label_elev = QLabel("Elev:")
        self.slider_elev = QSlider(QtCore.Qt.Horizontal)
        self.slider_elev.setRange(0, 90)
        self.slider_elev.setValue(35)
        self.slider_elev.setToolTip("Elevation angle for 3D view")
        self.label_azim = QLabel("Azim:")
        self.slider_azim = QSlider(QtCore.Qt.Horizontal)
        self.slider_azim.setRange(0, 360)
        self.slider_azim.setValue(135)
        self.slider_azim.setToolTip("Azimuth angle for 3D view")
        self.label_elev.hide()
        self.slider_elev.hide()
        self.label_azim.hide()
        self.slider_azim.hide()
        self.slider_elev.valueChanged.connect(self.trigger_plot)
        self.slider_azim.valueChanged.connect(self.trigger_plot)

    def create_layout(self):
        main_layout = QVBoxLayout()

        # Create a splitter
        splitter = QSplitter(QtCore.Qt.Vertical)

        # Canvas widget
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.addWidget(self.canvas)
        canvas_widget.setMinimumHeight(400)  # Set a minimum height for the canvas

        # Controls widget
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Row 1: toolbar + Show Legend + Show Grid + Point Size + Export
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.toolbar)
        hbox1.addStretch()
        hbox1.addWidget(self.checkbox_legend)
        hbox1.addWidget(self.checkbox_grid)
        hbox1.addWidget(self.label_pointsize)
        hbox1.addWidget(self.spin_point_size)
        hbox1.addWidget(self.button_save)

        grid1 = QGridLayout()
        for col in (1, 3, 5, 7):
            grid1.setColumnStretch(col, 1)

        # Row 0: all dropdowns
        grid1.addWidget(self.plot_types_label, 0, 0)
        grid1.addWidget(self.plot_types_combobox, 0, 1)
        grid1.addWidget(self.plot_permutations_label, 0, 2)
        grid1.addWidget(self.plot_permutations_combobox, 0, 3)
        grid1.addWidget(self.variables_label, 0, 4)
        grid1.addWidget(self.variables_combobox, 0, 5)
        grid1.addWidget(self.cmap_label, 0, 6)
        grid1.addWidget(self.cmap_combobox, 0, 7)

        # Row 1: buttons on the left, spacer, checkboxes on the right
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.button_filter_data, 1)
        hbox2.addWidget(self.button_customize_labels, 1)
        hbox2.addWidget(self.button_smooth_data, 1)
        hbox2.addWidget(self.button_group_data, 1)
        hbox2.addWidget(self.button_add_trendline, 1)
        hbox2.addWidget(self.button_reset_layout, 1)
        hbox2.addStretch(2)
        hbox2.addWidget(self.checkbox_zingg)
        hbox2.addWidget(self.checkbox_corr_mat)
        hbox2.addWidget(self.checkbox_cluster_mat)
        hbox2.addWidget(self.checkbox_hide_bulk)
        hbox2.addWidget(self.checkbox_invert_x)
        hbox2.addWidget(self.checkbox_invert_y)
        hbox2.addWidget(self.checkbox_invert_z)
        grid1.addLayout(hbox2, 1, 0, 1, 9)

        grid1.addWidget(self.custom_plot_widget, 2, 0, 1, 9)

        # Add time-series widget (for Site Analysis mode)
        grid1.addWidget(self.time_series_widget, 3, 0, 1, 9)

        # View angle sliders for 4D mode
        hbox_sliders = QHBoxLayout()
        hbox_sliders.addWidget(self.label_elev)
        hbox_sliders.addWidget(self.slider_elev)
        hbox_sliders.addWidget(self.label_azim)
        hbox_sliders.addWidget(self.slider_azim)
        self._sliders_layout = hbox_sliders
        grid1.addLayout(hbox_sliders, 4, 0, 1, 9)

        # Set alignment to right for labels only
        for i in range(grid1.rowCount()):
            for j in range(grid1.columnCount()):
                item = grid1.itemAtPosition(i, j)
                if item and item.widget() and isinstance(item.widget(), QLabel):
                    grid1.setAlignment(item.widget(), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        controls_layout.addLayout(hbox1)
        controls_layout.addLayout(grid1)

        # Add widgets to splitter
        splitter.addWidget(canvas_widget)
        splitter.addWidget(controls_widget)

        # Set initial sizes (70% canvas, 30% controls)
        splitter.setSizes([700, 300])

        main_layout.addWidget(splitter)

        # Set window properties
        self.setWindowTitle("Plot Window")
        self.setGeometry(100, 100, 850, 1000)
        self.setLayout(main_layout)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def set_point_size(self):
        """Update point size for all scatter plots."""
        self.point_size = self.spin_point_size.value()
        for plot in self.plot_objects.values():
            if plot.scatter is not None:
                plot.scatter.set_sizes([self.point_size] * len(plot.scatter.get_offsets()))
        self.canvas.draw()

    def change_mode(self, mode):
        # Restore permutations combobox label and content to defaults before any mode overrides
        self.plot_permutations_label.setText("Permutations: ")
        self.plot_permutations_combobox.blockSignals(True)
        self.plot_permutations_combobox.clear()
        self.plot_permutations_combobox.addItems(self.permutation_labels)
        self.plot_permutations_combobox.blockSignals(False)

        # Show/hide Growth-Rates-only buttons
        if mode == "Growth Rates":
            self.button_smooth_data.show()
            self.button_group_data.show()
        else:
            self.button_smooth_data.hide()
            self.button_group_data.hide()

        if mode == "Site Analysis":
            # Repopulate combobox with SA-specific permutation labels
            self.plot_permutations_combobox.blockSignals(True)
            self.plot_permutations_combobox.clear()
            self.plot_permutations_combobox.addItems(self.sa_permutation_labels)
            self.plot_permutations_combobox.blockSignals(False)
            # Enable permutation controls for Site Analysis (3 permutations)
            self.plot_permutations_label.setEnabled(True)
            self.plot_permutations_combobox.setEnabled(True)
            # Enable variable controls for Site Analysis
            self.variables_label.setEnabled(True)
            self.variables_combobox.setEnabled(True)
            self.custom_plot_widget.hide()
            self.checkbox_corr_mat.setEnabled(False)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_zingg.setEnabled(False)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_cluster_mat.setEnabled(False)
            self.checkbox_cluster_mat.setChecked(False)
            # Enable filter data button in Site Analysis mode (now supports filtering)
            self.button_filter_data.setEnabled(True)
            # Show time-series widget for Site Analysis
            self.time_series_widget.show()
            # Only initialize if the widget hasn't been set up yet
            # (check if file_prefixes list is empty)
            if not self.time_series_widget.file_prefixes:
                self._initialize_time_series_widget()
            # Update hide bulk checkbox visibility based on plotting mode
            self._update_hide_bulk_visibility()
        elif mode == "Heatmap":
            self.plot_permutations_label.setEnabled(False)
            self.plot_permutations_combobox.setEnabled(False)
            self.plot_permutations_combobox.setCurrentIndex(0)
            self.variables_label.setEnabled(False)
            self.variables_combobox.setEnabled(False)
            self.variables_combobox.setCurrentIndex(0)
            self.custom_plot_widget.show()

            # Disable checkboxes in Heatmap mode
            self.checkbox_corr_mat.setEnabled(False)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_zingg.setEnabled(False)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_cluster_mat.setEnabled(False)
            self.checkbox_cluster_mat.setChecked(False)
            # Enable filter data button in Heatmap mode
            self.button_filter_data.setEnabled(True)
            # Hide hide bulk checkbox in Heatmap mode
            self.checkbox_hide_bulk.hide()

            # Show single selection Y-axis widget, hide checkable one
            self.custom_plot_widget.yAxisListWidget.hide()
            self.custom_plot_widget.yAxisListWidget_single.show()
            self.custom_plot_widget.y_axis_label.setText("Y-axis (select one)")
            self.custom_plot_widget.color_label.setText("Value (select one)")

            # Hide time-series widget
            self.time_series_widget.hide()

        elif mode == "Custom":
            self.plot_permutations_label.setEnabled(False)
            self.plot_permutations_combobox.setEnabled(False)
            self.plot_permutations_combobox.setCurrentIndex(0)
            self.variables_label.setEnabled(False)
            self.variables_combobox.setEnabled(False)
            self.variables_combobox.setCurrentIndex(0)
            self.custom_plot_widget.show()

            # Enable checkboxes in Custom mode
            self.checkbox_corr_mat.setEnabled(True)
            self.checkbox_corr_mat.show()
            self.checkbox_zingg.setEnabled(True)
            self.checkbox_zingg.show()
            self.checkbox_cluster_mat.setEnabled(True)
            self.checkbox_cluster_mat.show()
            # Enable filter data button in Custom mode
            self.button_filter_data.setEnabled(True)
            # Hide hide bulk checkbox in Custom mode
            self.checkbox_hide_bulk.hide()

            # Show checkable Y-axis widget, hide single selection one
            self.custom_plot_widget.yAxisListWidget.show()
            self.custom_plot_widget.yAxisListWidget_single.hide()
            self.custom_plot_widget.y_axis_label.setText("Y-axis (check multiple)")
            self.custom_plot_widget.color_label.setText("Color By (select one)")

            # Hide time-series widget
            self.time_series_widget.hide()

        elif mode == "Growth Rates":
            # Repopulate permutations combobox as X-Axis selector
            self.plot_permutations_label.setText("X-Axis: ")
            self.plot_permutations_combobox.blockSignals(True)
            self.plot_permutations_combobox.clear()
            self.plot_permutations_combobox.addItems(
                self.gr_x_candidates if self.gr_x_candidates else ["Supersaturation"]
            )
            idx = self.plot_permutations_combobox.findText(self.gr_x_col)
            self.plot_permutations_combobox.setCurrentIndex(max(idx, 0))
            self.plot_permutations_combobox.blockSignals(False)
            self.plot_permutations_label.setEnabled(True)
            self.plot_permutations_combobox.setEnabled(True)
            # Populate variables combobox with all other x-axis candidates for colouring
            current_var = self.variables_combobox.currentText()
            self.variables_combobox.blockSignals(True)
            self.variables_combobox.clear()
            other_vars = ["None"] + [c for c in self.gr_x_candidates if c != self.gr_x_col]
            self.variables_combobox.addItems(other_vars)
            restored_idx = self.variables_combobox.findText(current_var)
            self.variables_combobox.setCurrentIndex(max(restored_idx, 0))
            self.variables_combobox.blockSignals(False)
            self.variables_label.setEnabled(True)
            self.variables_combobox.setEnabled(True)
            self.custom_plot_widget.hide()
            self.checkbox_zingg.setEnabled(True)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_corr_mat.setEnabled(True)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_corr_mat.hide()
            self.checkbox_zingg.hide()
            self.checkbox_cluster_mat.setEnabled(True)
            self.checkbox_cluster_mat.setChecked(False)
            self.checkbox_cluster_mat.hide()
            self.button_filter_data.setEnabled(True)
            self.checkbox_hide_bulk.hide()
            self.custom_plot_widget.yAxisListWidget.show()
            self.custom_plot_widget.yAxisListWidget_single.hide()
            self.time_series_widget.hide()

        elif mode == "4D":
            self.plot_permutations_label.setEnabled(False)
            self.plot_permutations_combobox.setEnabled(False)
            self.plot_permutations_combobox.setCurrentIndex(0)
            # variables_combobox is the colour/4th dimension — populate with numeric df columns.
            # Preserve the current selection so repeated change_mode calls don't reset it.
            current_color = self.variables_combobox.currentText()
            self.variables_combobox.blockSignals(True)
            self.variables_combobox.clear()
            if self.df is not None:
                numeric_cols = list(self.df.select_dtypes(include="number").columns)
                self.variables_combobox.addItems(["None"] + numeric_cols)
            else:
                self.variables_combobox.addItems(["None"])
            restored = self.variables_combobox.findText(current_color)
            self.variables_combobox.setCurrentIndex(max(restored, 0))
            self.variables_combobox.blockSignals(False)
            self.variables_label.setText("Color By:")
            self.variables_label.setEnabled(True)
            self.variables_combobox.setEnabled(True)
            self.custom_plot_widget.show()

            self.checkbox_corr_mat.setEnabled(False)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_zingg.setEnabled(False)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_cluster_mat.setEnabled(False)
            self.checkbox_cluster_mat.setChecked(False)
            self.checkbox_hide_bulk.hide()
            # Enable Z inversion only in 4D mode
            self.checkbox_invert_z.setEnabled(True)

            self.button_filter_data.setEnabled(True)
            self.time_series_widget.hide()

            # Y-axis single selection; color widget becomes Z-axis
            self.custom_plot_widget.yAxisListWidget.hide()
            self.custom_plot_widget.yAxisListWidget_single.show()
            self.custom_plot_widget.y_axis_label.setText("Y-axis (select one)")
            self.custom_plot_widget.color_label.setText("Z-axis (select one)")

            # Show view-angle sliders
            self.label_elev.show()
            self.slider_elev.show()
            self.label_azim.show()
            self.slider_azim.show()

        elif mode in ("SA:Vol Ratio", "Zingg"):
            self.plot_permutations_label.setEnabled(True)
            self.plot_permutations_combobox.setEnabled(True)
            # Populate variables_combobox with numeric df columns for colour-by
            current_var = self.variables_combobox.currentText()
            self.variables_combobox.blockSignals(True)
            self.variables_combobox.clear()
            if self.df is not None:
                numeric_cols = list(self.df.select_dtypes(include="number").columns)
                self.variables_combobox.addItems(["None"] + numeric_cols)
            else:
                self.variables_combobox.addItems(["None"])
            restored = self.variables_combobox.findText(current_var)
            self.variables_combobox.setCurrentIndex(max(restored, 0))
            self.variables_combobox.blockSignals(False)
            self.variables_label.setText("Color By:")
            self.variables_label.setEnabled(True)
            self.variables_combobox.setEnabled(True)
            self.custom_plot_widget.hide()
            self.checkbox_zingg.setEnabled(True)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_corr_mat.setEnabled(True)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_corr_mat.hide()
            self.checkbox_zingg.hide()
            self.checkbox_cluster_mat.setEnabled(True)
            self.checkbox_cluster_mat.setChecked(False)
            self.checkbox_cluster_mat.hide()
            self.button_filter_data.setEnabled(True)
            self.checkbox_hide_bulk.hide()
            self.custom_plot_widget.yAxisListWidget.show()
            self.custom_plot_widget.yAxisListWidget_single.hide()
            self.time_series_widget.hide()

        else:
            self.plot_permutations_label.setEnabled(True)
            self.plot_permutations_combobox.setEnabled(True)
            # Restore variables_combobox to interaction columns when leaving 4D
            self.variables_combobox.blockSignals(True)
            self.variables_combobox.clear()
            if self.site_analysis_data is not None:
                self.variables_combobox.addItems(self.site_analysis_columns)
            else:
                self.variables_combobox.addItems(self.interaction_columns)
            self.variables_combobox.blockSignals(False)
            self.variables_label.setText("Variable: ")
            self.variables_label.setEnabled(True)
            self.variables_combobox.setEnabled(True)
            self.custom_plot_widget.hide()
            self.checkbox_zingg.setEnabled(True)
            self.checkbox_zingg.setChecked(False)
            self.checkbox_corr_mat.setEnabled(True)
            self.checkbox_corr_mat.setChecked(False)
            self.checkbox_corr_mat.hide()
            self.checkbox_zingg.hide()
            self.checkbox_cluster_mat.setEnabled(True)
            self.checkbox_cluster_mat.setChecked(False)
            self.checkbox_cluster_mat.hide()
            # Enable filter data button in default mode
            self.button_filter_data.setEnabled(True)
            # Hide hide bulk checkbox in default mode
            self.checkbox_hide_bulk.hide()

            # Ensure checkable widget is shown when hiding custom widget
            self.custom_plot_widget.yAxisListWidget.show()
            self.custom_plot_widget.yAxisListWidget_single.hide()

            # Hide time-series widget
            self.time_series_widget.hide()

        # Hide view-angle sliders and disable Z-inversion for all non-4D modes
        if mode != "4D":
            self.label_elev.hide()
            self.slider_elev.hide()
            self.label_azim.hide()
            self.slider_azim.hide()
            self.checkbox_invert_z.setEnabled(False)
            self.checkbox_invert_z.setChecked(False)

    def _update_hide_bulk_visibility(self):
        """Update the visibility of the Hide Bulk Site checkbox based on plotting mode and permutation."""
        if self.plot_type != "Site Analysis":
            self.checkbox_hide_bulk.hide()
            return

        # Get current plotting mode
        plotting_mode = self.time_series_widget.get_plotting_mode()

        # Show checkbox only for population-based plotting modes
        # Available in all permutations when in population mode
        if plotting_mode in ["Total Population", "Population per Step"]:
            self.checkbox_hide_bulk.show()
        else:
            self.checkbox_hide_bulk.hide()

    def _initialize_time_series_widget(self):
        """Initialize the time-series widget with site analysis data."""
        if self.site_analysis_data is None:
            logger.warning("No site analysis data available for time-series widget")
            return

        # Set the available file prefixes
        file_prefixes = list(self.site_analysis_data.keys())
        self.time_series_widget.set_file_prefixes(file_prefixes)
        logger.debug(f"Set {len(file_prefixes)} file prefixes: {file_prefixes}")

        # Get the first dataset to extract time-series parameters
        first_dataset = next(iter(self.site_analysis_data.values()), None)
        if first_dataset is None:
            logger.warning("No datasets found in site analysis data")
            return

        # Extract time-series arrays
        supersaturation = first_dataset.get("supersaturation")
        time = first_dataset.get("time")
        iterations = first_dataset.get("iterations")

        # Set the time-series data in the widget
        self.time_series_widget.set_time_data(supersaturation, time, iterations)
        logger.debug("Initialized time-series widget with data")

        # After initialization, manually trigger the initial file prefix sync
        # This ensures the XYZ visualization matches the initially selected prefix
        selected_prefix = self.time_series_widget.get_selected_file_prefix()
        if selected_prefix and selected_prefix != "All Data":
            xyz_index = self._find_xyz_index_for_prefix(selected_prefix)
            if xyz_index is not None and self.signals:
                logger.debug(
                    f"Initial sync: setting XYZ to index {xyz_index} for prefix {selected_prefix}"
                )
                self.signals.sim_id.emit(xyz_index)

    def _on_file_prefix_changed(self, selected_prefix):
        """Handle file prefix selection change.

        This updates the displayed XYZ file when the data source dropdown is changed.
        """
        logger.debug(f"_on_file_prefix_changed called with prefix: {selected_prefix}")

        if self.site_analysis_data is None:
            logger.debug("No site analysis data, skipping file prefix change handling")
            return

        # If we have signals and this is not "All Data", try to find the corresponding XYZ file
        if self.signals and selected_prefix != "All Data":
            # Try to find the index of the XYZ file that matches this prefix
            xyz_index = self._find_xyz_index_for_prefix(selected_prefix)
            logger.debug(f"Found XYZ index {xyz_index} for prefix {selected_prefix}")
            if xyz_index is not None:
                logger.debug(
                    f"Emitting sim_id signal for prefix {selected_prefix} -> index {xyz_index}"
                )
                self.signals.sim_id.emit(xyz_index)
            else:
                logger.warning(f"Could not find XYZ index for prefix {selected_prefix}")
        else:
            if not self.signals:
                logger.debug("No signals object available")
            if selected_prefix == "All Data":
                logger.debug("Selected 'All Data', not changing XYZ visualization")

        # Trigger a replot - data will be extracted from dictionary in _set_data()
        logger.debug("Triggering plot after file prefix change")
        self.trigger_plot()

    def _find_xyz_index_for_prefix(self, file_prefix):
        """Try to find the XYZ file index that corresponds to a file prefix.

        Since the site analysis data and XYZ files are in the same order,
        we can use the position of the prefix in the site analysis data keys
        as the index for the XYZ files.

        Args:
            file_prefix: The file prefix to search for

        Returns:
            int or None: The index of the matching XYZ file, or None if not found
        """
        if self.site_analysis_data is None:
            return None

        # Get the ordered list of file prefixes from site analysis data
        file_prefixes = list(self.site_analysis_data.keys())

        # Find the index of this prefix in the ordered list
        try:
            prefix_index = file_prefixes.index(file_prefix)
            logger.debug(
                f"Found prefix '{file_prefix}' at index {prefix_index} in site_analysis_data"
            )

            # Verify this index is valid for xyz_files if available
            if hasattr(self.parent(), "xyz_files"):
                xyz_files = self.parent().xyz_files
                if 0 <= prefix_index < len(xyz_files):
                    return prefix_index
                else:
                    logger.warning(
                        f"Prefix index {prefix_index} out of range for xyz_files (length {len(xyz_files)})"
                    )
                    return None
            else:
                # If we don't have access to xyz_files, just return the index
                return prefix_index
        except ValueError:
            logger.warning(f"Prefix '{file_prefix}' not found in site_analysis_data keys")
            return None

    def open_axes_customization_dialog(self):
        """Open (or raise) the persistent, non-modal axes customisation dialog."""
        if self._axes_dialog is None or not self._axes_dialog.isVisible():
            current_labels = {
                "title": self.custom_title or self.title,
                "title_auto": self.title,
                "xlabel": self.custom_xlabel
                or (self.x_label if isinstance(self.x_label, PlotAxisLabel) else PlotAxisLabel()),
                "ylabel": self.custom_ylabel
                or (self.y_label if isinstance(self.y_label, PlotAxisLabel) else PlotAxisLabel()),
                "cbar_label": self.custom_cbar_label
                or (self.c_label if isinstance(self.c_label, PlotAxisLabel) else PlotAxisLabel()),
            }
            self._axes_dialog = AxesCustomizationDialog(current_labels, parent=self)
            self._axes_dialog.setWindowModality(QtCore.Qt.NonModal)
            self._axes_dialog.axes_applied.connect(self._update_labels_and_replot)
            self.labels_updated.connect(self._axes_dialog.refresh_labels)
            self._axes_dialog.finished.connect(
                lambda: self.labels_updated.disconnect(self._axes_dialog.refresh_labels)
            )
        self._axes_dialog.show()
        self._axes_dialog.raise_()
        self._axes_dialog.activateWindow()

    def _update_labels_and_replot(self, result: dict):
        """Apply label/conversion changes from :class:`AxesCustomizationDialog` and replot."""
        self.custom_title = result.get("title", "")

        xlabel = result.get("xlabel")
        ylabel = result.get("ylabel")
        cbar = result.get("cbar_label")

        self.custom_xlabel = (
            xlabel if isinstance(xlabel, PlotAxisLabel) and xlabel.is_user_set else None
        )
        self.custom_ylabel = (
            ylabel if isinstance(ylabel, PlotAxisLabel) and ylabel.is_user_set else None
        )
        self.custom_cbar_label = (
            cbar if isinstance(cbar, PlotAxisLabel) and cbar.is_user_set else None
        )

        logger.debug("Axes updated: %s", result)
        self.trigger_plot()

    def toggle_gr_grouping(self):
        """Toggle grouped/ungrouped mode for Growth Rates plots."""
        self.gr_grouped = not self.gr_grouped
        self.button_group_data.setText("Ungroup" if self.gr_grouped else "Group")
        self.trigger_plot()

    def open_smoothing_dialog(self):
        """Open the smoothing/interpolation/extrapolation dialog."""
        if self.plot_type != "Growth Rates":
            logger.warning("Smoothing dialog only available in Growth Rates mode")
            return

        # Get the list of series (directions) available
        series_names = self.directions if self.directions else []

        if not series_names:
            logger.warning("No data series available for smoothing")
            return

        dialog = SmoothingDialog(
            series_names=series_names, existing_configs=self.smoothing_configs, parent=self
        )

        if dialog.exec() == QDialog.Accepted:
            # Get the configurations
            configs, legend_mode = dialog.get_configs()
            self.smoothing_configs = configs
            self.smoothing_legend_mode = legend_mode

            # Update button text to indicate active smoothing
            if configs:
                self.button_smooth_data.setText(f"Smooth/Extrapolate Data... ({len(configs)})")
            else:
                self.button_smooth_data.setText("Smooth/Extrapolate Data")

            # Replot with smoothed data
            self.trigger_plot()

    def open_filter_dialog(self):
        """Open the data filter dialog."""
        # Use the original unfiltered dataframe for the filter dialog
        original_df = self._get_original_df()

        # Pass site_analysis_data if in site analysis mode
        site_analysis_data = self.site_analysis_data if self.plot_type == "Site Analysis" else None

        dialog = DataFilterDialog(
            df=original_df,
            current_filters=self.data_filters,
            parent=self,
            site_analysis_data=site_analysis_data,
            current_interaction_filters=self.interaction_filters,
        )

        # Connect Apply button signal to update filters without closing dialog
        dialog.filters_applied.connect(self._update_filters_and_replot)

        if dialog.exec() == QDialog.Accepted:
            # Update filters when OK is clicked
            data_filters = dialog.get_filters()
            interaction_filters = dialog.get_interaction_filters()
            self._update_filters_and_replot(data_filters, interaction_filters)

    def _update_filters_and_replot(self, data_filters, interaction_filters=None):
        """Update data and interaction filters and trigger replot.

        Args:
            data_filters: List of data filter dictionaries
            interaction_filters: Dictionary of interaction filters (optional)
        """
        self.data_filters = data_filters
        if interaction_filters is not None:
            self.interaction_filters = interaction_filters

        # Update button text to indicate active filters
        total_filters = len(self.data_filters)
        if self.interaction_filters:
            total_filters += len(self.interaction_filters)

        if total_filters > 0:
            self.button_filter_data.setText(f"Filter Data... ({total_filters})")
        else:
            self.button_filter_data.setText("Filter Data")

        # Replot with filtered data
        self.trigger_plot()

    def _get_original_df(self):
        """Get the original unfiltered dataframe."""
        return self.df_original if hasattr(self, "df_original") else self.df

    def _check_site_data_filter(self, site_data: dict, filter_config: dict) -> bool:
        """Check if a site passes a data filter.

        Args:
            site_data: Dictionary containing site data
            filter_config: Filter configuration dict with 'column', 'operator', 'value'

        Returns:
            True if the site passes the filter, False otherwise
        """
        column = filter_config["column"]
        operator = filter_config["operator"]
        value_str = filter_config["value"]

        # Get the site's value for this column
        if column not in site_data:
            return False

        site_value = site_data[column]

        # Handle None values - exclude sites with None unless using != operator
        if site_value is None:
            if operator == "!=":
                return True  # None != any value is True
            else:
                return False  # None fails all other comparisons

        try:
            # Try to convert to numeric if both are numeric
            if isinstance(site_value, (int, float)):
                value = float(value_str)
            else:
                value = value_str

            # Apply the filter based on operator
            if operator == "==":
                return site_value == value
            elif operator == "!=":
                return site_value != value
            elif operator == ">":
                return site_value > value
            elif operator == ">=":
                return site_value >= value
            elif operator == "<":
                return site_value < value
            elif operator == "<=":
                return site_value <= value
            elif operator == "contains":
                return str(value).lower() in str(site_value).lower()
            elif operator == "not contains":
                return str(value).lower() not in str(site_value).lower()
            else:
                logger.warning(f"Unknown operator: {operator}")
                return True

        except Exception as e:
            logger.error(f"Error applying filter {filter_config} to site data: {e}")
            return False  # Changed from True to False - if filter fails, exclude the site

    def _apply_data_filters(self):
        """Apply data filters to the dataframe."""
        # Skip for site analysis - filters are applied in _extract_site_analysis_data()
        if self.plot_type == "Site Analysis" or self.df_original is None:
            return

        if not self.data_filters:
            self.df = self.df_original.copy()
            return

        filtered_df = self.df_original.copy()

        for filter_config in self.data_filters:
            column = filter_config["column"]
            operator = filter_config["operator"]
            value_str = filter_config["value"]

            if column not in filtered_df.columns:
                logger.warning(f"Column '{column}' not found in dataframe")
                continue

            try:
                # Try to convert value to numeric if the column is numeric
                if pd.api.types.is_numeric_dtype(filtered_df[column]):
                    value = pd.to_numeric(value_str)
                else:
                    value = value_str

                # Apply the filter based on operator
                if operator == "==":
                    mask = filtered_df[column] == value
                elif operator == "!=":
                    mask = filtered_df[column] != value
                elif operator == ">":
                    mask = filtered_df[column] > value
                elif operator == ">=":
                    mask = filtered_df[column] >= value
                elif operator == "<":
                    mask = filtered_df[column] < value
                elif operator == "<=":
                    mask = filtered_df[column] <= value
                elif operator == "contains":
                    mask = filtered_df[column].astype(str).str.contains(str(value), case=False)
                elif operator == "not contains":
                    mask = ~filtered_df[column].astype(str).str.contains(str(value), case=False)
                else:
                    logger.warning(f"Unknown operator: {operator}")
                    continue

                filtered_df = filtered_df[mask]

            except Exception as e:
                logger.error(f"Error applying filter {filter_config}: {e}")
                continue

        self.df = filtered_df
        logger.debug(
            f"Applied {len(self.data_filters)} filters: {len(self.df)} rows remaining from {len(self.df_original)} original rows"
        )

    def trigger_plot(self):
        data_filters = self.data_filters
        interaction_filters = self.interaction_filters

        self.setPlotDefaults()

        # Apply data filters
        self.data_filters = data_filters
        self.interaction_filters = interaction_filters
        self._apply_data_filters()

        # Update filter button text
        total_filters = len(self.data_filters)
        if self.interaction_filters:
            total_filters += len(self.interaction_filters)

        if total_filters > 0:
            self.button_filter_data.setText(f"Filter Data... ({total_filters})")
        else:
            self.button_filter_data.setText("Filter Data")

        self.grid = self.checkbox_grid.isChecked()
        self.show_legend = self.checkbox_legend.isChecked()
        self.zingg = self.checkbox_zingg.isChecked()
        self.covmat = self.checkbox_corr_mat.isChecked()
        self.cluster_mat = self.checkbox_cluster_mat.isChecked()
        self.point_size = self.spin_point_size.value()
        self.plot_type = self.plot_types_combobox.currentText()
        self.variable = self.variables_combobox.currentText()
        (
            self.custom_x,
            self.custom_y,
            self.custom_c,
        ) = self.custom_plot_widget.get_selections()
        self.custom_y = [tmp[1] for tmp in self.custom_y]

        # Get colour scheme settings
        self.cmap = self.cmap_combobox.currentText()

        # Save current permutation index before change_mode clears and resets the combobox
        saved_perm_idx = self.plot_permutations_combobox.currentIndex()

        # change_mode must run first for Growth Rates — it repopulates the
        # permutations combobox with gr_x_candidates before we read from it.
        self.change_mode(mode=self.plot_type)

        if self.plot_type == "Growth Rates":
            self.gr_x_col = self.plot_permutations_combobox.currentText() or "Supersaturation"
        else:
            # Clamp to valid range in case a mode switch changed the item count
            restored_idx = min(saved_perm_idx, self.plot_permutations_combobox.count() - 1)
            self.permutation = restored_idx
            self.plot_permutations_combobox.blockSignals(True)
            self.plot_permutations_combobox.setCurrentIndex(restored_idx)
            self.plot_permutations_combobox.blockSignals(False)
        # Update hide bulk checkbox visibility (needed when plotting mode changes within Site Analysis)
        self._update_hide_bulk_visibility()
        self._set_data()
        self._set_c()
        self._mask_with_permutation()
        self._set_labels()
        self._set_c_label()
        self._apply_axis_conversions()
        self.plot()
        self.labels_updated.emit(
            {
                "title": self.custom_title or self.title,
                "title_auto": self.title,
                "xlabel": self.custom_xlabel or self.x_label,
                "ylabel": self.custom_ylabel or self.y_label,
                "cbar_label": self.custom_cbar_label or self.c_label,
            }
        )

    def _apply_axis_conversions(self):
        """Apply any active unit conversions stored on x/y/c labels to the data arrays."""
        pairs = [
            (self.custom_xlabel or self.x_label, "x_data"),
            (self.custom_ylabel or self.y_label, "y_data"),
            (self.custom_cbar_label or self.c_label, "c_data"),
        ]
        for label, attr in pairs:
            if not isinstance(label, PlotAxisLabel):
                continue
            if label.conversion is None:
                continue
            data = getattr(self, attr, None)
            if data is None:
                continue
            try:
                setattr(self, attr, label.conversion.apply(np.asarray(data, dtype=float)))
            except Exception as exc:
                logger.warning("Unit conversion failed for %s: %s", attr, exc)

    def _check_interaction_filter(self, site_interactions: dict, interaction_filters: dict) -> bool:
        """Check if a site passes the interaction filter.

        Args:
            site_interactions: Dictionary mapping interaction_id to frequency for this site
            interaction_filters: Dictionary mapping interaction_id to list of selected frequencies

        Returns:
            True if the site passes the filter (should be shown), False otherwise
        """
        if not interaction_filters:
            # No filters active, show all sites
            return True

        # Convert site_interactions keys and values to integers for comparison
        site_interactions_int = {int(k): int(v) for k, v in site_interactions.items()}

        # Site must match ALL selected interaction filters (AND logic)
        for interaction_id, selected_freqs in interaction_filters.items():
            # Convert interaction_id to integer for consistent comparison
            interaction_id_int = int(interaction_id)

            # Check if site has this interaction
            if interaction_id_int not in site_interactions_int:
                # Site doesn't have this interaction at all
                return False

            site_freq = site_interactions_int[interaction_id_int]

            # Check if the site's frequency matches any of the selected frequencies
            if "Any" in selected_freqs:
                # "Any" is selected, so any frequency for this interaction is acceptable
                continue
            else:
                # Convert selected frequencies to integers for comparison (except "Any")
                selected_freqs_int = [int(f) if f != "Any" else f for f in selected_freqs]
                if site_freq not in selected_freqs_int:
                    # Site's frequency doesn't match any selected frequency
                    return False

        # Site passed all filters
        return True

    def _extract_site_analysis_data(self):
        """Extract site analysis data directly from dictionary based on current mode and filters.

        Returns:
            tuple: (x_data_array, y_data_array, site_metadata_list)
                  where site_metadata_list contains dicts with site info for hover/click handling
        """
        # Get the selected file prefix
        selected_prefix = self.time_series_widget.get_selected_file_prefix()

        # Get current plotting mode and time point
        plotting_mode = self.time_series_widget.get_plotting_mode()
        _, _, time_index = self.time_series_widget.get_current_time_point()

        # Get current permutation (0, 1, or 2)
        permutation = self.permutation

        # Get interaction filters if active
        interaction_filters = self.interaction_filters

        # Determine which prefixes to include
        if selected_prefix == "All Data":
            prefixes_to_process = list(self.site_analysis_data.keys())
        else:
            prefixes_to_process = (
                [selected_prefix] if selected_prefix in self.site_analysis_data else []
            )

        x_values = []
        y_values = []
        site_metadata = []

        # Extract data from each file prefix
        for file_prefix in prefixes_to_process:
            dataset = self.site_analysis_data[file_prefix]
            sites_dict = dataset.get("sites", {})

            for site_num, site_data in sites_dict.items():
                # Apply data filters if any
                # If variable is "filter", we want to show ALL sites and color them (filtered vs unfiltered)
                # If variable is NOT "filter", we want to HIDE sites that don't pass the filter
                if self.data_filters and self.variable != "filter":
                    # Check if this site passes all data filters
                    passes_filters = True
                    for filter_config in self.data_filters:
                        column = filter_config["column"]

                        # Handle special fields that aren't in site_data dict
                        if column == "site_number":
                            # Check site_number filter directly
                            site_data_temp = {"site_number": int(site_num)}
                            if not self._check_site_data_filter(site_data_temp, filter_config):
                                passes_filters = False
                                break
                        elif column == "file_prefix":
                            # Check file_prefix filter directly
                            site_data_temp = {"file_prefix": file_prefix}
                            if not self._check_site_data_filter(site_data_temp, filter_config):
                                passes_filters = False
                                break
                        else:
                            # Check normal site_data fields
                            if not self._check_site_data_filter(site_data, filter_config):
                                passes_filters = False
                                break
                    if not passes_filters:
                        continue

                # Apply interaction filters if active
                # If variable is "filter", we want to show ALL sites and color them (filtered vs unfiltered)
                # If variable is NOT "filter", we want to HIDE sites that don't pass the filter
                if interaction_filters and self.variable != "filter":
                    site_interactions = site_data.get("interactions", {})
                    passes_interaction_filter = self._check_interaction_filter(
                        site_interactions, interaction_filters
                    )
                    if not passes_interaction_filter:
                        continue

                # Skip sites without energy data
                if site_data.get("energy") is None:
                    continue

                # Extract x and y values based on permutation
                x_value = None
                y_value = None

                # Permutation 0: Events/Population vs Energy (current behavior)
                if permutation == 0:
                    # Extract x value based on plotting mode
                    if plotting_mode == "Total Events":
                        if site_data.get("total_events") is not None:
                            # For events-based plotting: flip the sign
                            # Ungrown sites (growth) are positive, grown sites (dissolution) are negative
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * site_data["total_events"]

                    elif plotting_mode == "Total Population":
                        if site_data.get("total_population") is not None:
                            sign = 1 if site_data.get("occupation") else -1
                            x_value = sign * site_data["total_population"]

                    elif plotting_mode == "Events per Step":
                        events_series = site_data.get("events")
                        if (
                            events_series
                            and isinstance(events_series, list)
                            and time_index < len(events_series)
                        ):
                            # For events-based plotting: flip the sign
                            # Ungrown sites (growth) are positive, grown sites (dissolution) are negative
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * events_series[time_index]
                        elif site_data.get("total_events") is not None:
                            # Fallback to total events
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * site_data["total_events"]

                    elif plotting_mode == "Population per Step":
                        population_series = site_data.get("population")
                        if (
                            population_series
                            and isinstance(population_series, list)
                            and time_index < len(population_series)
                        ):
                            sign = 1 if site_data.get("occupation") else -1
                            x_value = sign * population_series[time_index]
                        elif site_data.get("total_population") is not None:
                            # Fallback to total population
                            sign = 1 if site_data.get("occupation") else -1
                            x_value = sign * site_data["total_population"]

                    # Y value is always energy
                    y_value = site_data["energy"]

                # Permutation 1: Sites vs Events/Population
                elif permutation == 1:
                    # X value is the site number
                    x_value = int(site_num)

                    # Y value based on plotting mode
                    if plotting_mode == "Total Events":
                        if site_data.get("total_events") is not None:
                            sign = -1 if site_data.get("occupation") else 1
                            y_value = sign * site_data["total_events"]

                    elif plotting_mode == "Total Population":
                        if site_data.get("total_population") is not None:
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * site_data["total_population"]

                    elif plotting_mode == "Events per Step":
                        events_series = site_data.get("events")
                        if (
                            events_series
                            and isinstance(events_series, list)
                            and time_index < len(events_series)
                        ):
                            sign = -1 if site_data.get("occupation") else 1
                            y_value = sign * events_series[time_index]
                        elif site_data.get("total_events") is not None:
                            sign = -1 if site_data.get("occupation") else 1
                            y_value = sign * site_data["total_events"]

                    elif plotting_mode == "Population per Step":
                        population_series = site_data.get("population")
                        if (
                            population_series
                            and isinstance(population_series, list)
                            and time_index < len(population_series)
                        ):
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * population_series[time_index]
                        elif site_data.get("total_population") is not None:
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * site_data["total_population"]

                # Permutation 2: Events vs Population
                elif permutation == 2:
                    # X value is events, Y value is population
                    if plotting_mode == "Total Events" or plotting_mode == "Total Population":
                        # Use total values
                        if site_data.get("total_events") is not None:
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * site_data["total_events"]
                        if site_data.get("total_population") is not None:
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * site_data["total_population"]

                    elif (
                        plotting_mode == "Events per Step" or plotting_mode == "Population per Step"
                    ):
                        # Use per-step values
                        events_series = site_data.get("events")
                        population_series = site_data.get("population")

                        if (
                            events_series
                            and isinstance(events_series, list)
                            and time_index < len(events_series)
                        ):
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * events_series[time_index]
                        elif site_data.get("total_events") is not None:
                            sign = -1 if site_data.get("occupation") else 1
                            x_value = sign * site_data["total_events"]

                        if (
                            population_series
                            and isinstance(population_series, list)
                            and time_index < len(population_series)
                        ):
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * population_series[time_index]
                        elif site_data.get("total_population") is not None:
                            sign = 1 if site_data.get("occupation") else -1
                            y_value = sign * site_data["total_population"]

                # Skip if we couldn't get x or y values
                if x_value is None or y_value is None:
                    continue

                # Store the data
                x_values.append(x_value)
                y_values.append(y_value)

                # Store metadata for this site (for hover/click handling)
                metadata = {
                    "file_prefix": file_prefix,
                    "site_number": int(site_num),
                    "tile_type": site_data.get("tile_type"),
                    "energy": site_data.get("energy"),
                    "occupation": site_data.get("occupation"),
                    "coordination": site_data.get("coordination"),
                    "total_events": site_data.get("total_events"),
                    "total_population": site_data.get("total_population"),
                    "interactions": site_data.get("interactions", {}),
                }
                site_metadata.append(metadata)

        # Filter out bulk site if checkbox is checked and we're in population mode
        # For permutation 0 and 2, check x_values; for permutation 1, check y_values
        if self.checkbox_hide_bulk.isChecked() and plotting_mode in [
            "Total Population",
            "Population per Step",
        ]:
            if x_values:
                # Determine which axis contains population data
                if permutation == 0:
                    # Population is on x-axis
                    abs_values = [abs(x) for x in x_values]
                elif permutation == 1:
                    # Population is on y-axis
                    abs_values = [abs(y) for y in y_values]
                elif permutation == 2:
                    # Population is on y-axis
                    abs_values = [abs(y) for y in y_values]
                else:
                    abs_values = []

                if abs_values:
                    max_idx = np.argmax(abs_values)
                    # Remove the bulk site from all lists
                    x_values.pop(max_idx)
                    y_values.pop(max_idx)
                    site_metadata.pop(max_idx)

        # Convert to numpy arrays
        x_array = np.array(x_values)
        y_array = np.array(y_values)

        return x_array, y_array, site_metadata

    def _set_data(self):
        if self.plot_type == "Zingg":
            self.x_data = self.df["S:M"]
            self.y_data = self.df["M:L"]
        if self.plot_type == "CDA":
            ratio_columns = [col for col in self.df.columns if col.startswith("Ratio_")]
            self.x_data = self.df[ratio_columns[0]]
            self.y_data = self.df[ratio_columns[1]]
        if self.plot_type == "SA:Vol Ratio":
            self.x_data = self.df["Surface Area"]
            self.y_data = self.df["Volume"]
        if self.plot_type == "Growth Rates":
            x_col = self.gr_x_col if self.gr_x_col in self.df.columns else "Supersaturation"
            if self.gr_grouped:
                grouped = self.df.groupby(x_col)[self.directions]
                self.x_data = grouped.mean().index.to_numpy()
                self.y_data = grouped.mean()
                self.gr_y_err = grouped.std()
            else:
                self.x_data = self.df[x_col]
                self.y_data = self.df[self.directions]
                self.gr_y_err = None
        if self.plot_type == "Site Analysis":
            # Extract data directly from dictionary
            x_array, y_array, site_metadata = self._extract_site_analysis_data()

            self.x_data = x_array
            self.y_data = y_array

            # Store site metadata for hover/click handling
            self._site_metadata = site_metadata
        if self.plot_type == "Heatmap":
            # For heatmap: X and Y are axes, C (color) is the value to display
            self.x_data = None
            self.y_data = None
            if self.df is not None:
                try:
                    # Set defaults: S:M for X, Supersaturation for Y if available, Frame Index for value
                    if self.custom_x:
                        self.x_data = self.df[self.custom_x]
                    elif "S:M" in self.df.columns:
                        self.x_data = self.df["S:M"]
                        self.custom_x = "S:M"

                    if self.custom_y and len(self.custom_y) > 0:
                        self.y_data = self.df[self.custom_y[0]]
                    elif "Supersaturation" in self.df.columns:
                        self.y_data = self.df["Supersaturation"]
                        self.custom_y = ["Supersaturation"]
                except KeyError as exc:
                    logger.warning("X or Y data not been set, invalid axis title!\n%s", exc)
        if self.plot_type == "Custom":
            self.x_data = None
            self.y_data = None
            if self.cluster_mat:
                # Hierarchical clustering builds its own data; skip standard x/y loading
                return
            if self.df is not None:
                try:
                    self.x_data = self.df[self.custom_x]
                    self.y_data = self.df[self.custom_y]
                except KeyError as exc:
                    logger.warning("X and Y data not been set, invalid axis title!\n%s", exc)

        if self.plot_type == "4D":
            self.x_data = None
            self.y_data = None
            self.z_data = None
            if self.df is not None:
                try:
                    if self.custom_x:
                        self.x_data = self.df[self.custom_x]
                    if self.custom_y and len(self.custom_y) > 0:
                        self.y_data = self.df[self.custom_y[0]]
                    # Z axis comes from the color widget (repurposed)
                    if self.custom_c and self.custom_c != "None":
                        self.z_data = self.df[self.custom_c]
                except KeyError as exc:
                    logger.warning("4D data axis error: %s", exc)

        if self.plot_type not in ("Heatmap", "4D"):
            self.y_data = self._ensure_pd_series(self.y_data)

    def _mask_with_permutation(self):
        # Skip permutation masking if we don't have a DataFrame (e.g., site analysis mode)
        if self.df is None:
            return
        # Growth Rates uses gr_x_col for its x-axis — no CDA_Permutation column exists
        if self.plot_type == "Growth Rates":
            return
        if self.permutation == 0:
            self._set_data()
            return
        mask = self.df["CDA_Permutation"] == int(self.permutation)
        self.x_data = self.x_data[mask]
        self.y_data = self.y_data[mask]
        if self.c_data is not None:
            self.c_data = self.c_data[mask]

    def _set_labels(self):
        """Set auto-generated x/y/title labels from the current plot type and data columns.

        Always overwrites the auto labels; user overrides are stored separately in
        ``custom_xlabel`` / ``custom_ylabel`` and applied at display time.
        """

        def _set_x(label: PlotAxisLabel) -> None:
            self.x_label = label

        def _set_y(label: PlotAxisLabel) -> None:
            self.y_label = label

        if self.plot_type == "Site Analysis":
            plotting_mode = self.time_series_widget.get_plotting_mode()
            param_name, param_value, time_index = self.time_series_widget.get_current_time_point()
            permutation = self.permutation

            if permutation == 0:
                if plotting_mode == "Total Events":
                    _set_x(PlotAxisLabel("Total Events (Growth (+), Dissolution (-))"))
                    self.title = "Site Analysis: Total Events vs Energy"
                elif plotting_mode == "Total Population":
                    _set_x(PlotAxisLabel("Total Population (Grown (+), Ungrown (-))"))
                    self.title = "Site Analysis: Total Population vs Energy"
                elif plotting_mode == "Events per Step":
                    if param_value is not None:
                        _set_x(
                            PlotAxisLabel(
                                f"Events at {param_name.capitalize()}={param_value:.2f} (Growth (+), Dissolution (-))"
                            )
                        )
                        self.title = f"Site Analysis: Events vs Energy (t={time_index})"
                    else:
                        _set_x(PlotAxisLabel("Events per Step (Growth (+), Dissolution (-))"))
                        self.title = "Site Analysis: Events vs Energy"
                elif plotting_mode == "Population per Step":
                    if param_value is not None:
                        _set_x(
                            PlotAxisLabel(
                                f"Population at {param_name.capitalize()}={param_value:.2f} (Grown (+), Ungrown (-))"
                            )
                        )
                        self.title = f"Site Analysis: Population vs Energy (t={time_index})"
                    else:
                        _set_x(PlotAxisLabel("Population per Step (Grown (+), Ungrown (-))"))
                        self.title = "Site Analysis: Population vs Energy"
                _set_y(PlotAxisLabel("Energy"))

            elif permutation == 1:
                _set_x(PlotAxisLabel("Site Number"))
                if plotting_mode == "Total Events":
                    _set_y(PlotAxisLabel("Total Events (Growth (+), Dissolution (-))"))
                    self.title = "Site Analysis: Sites vs Total Events"
                elif plotting_mode == "Total Population":
                    _set_y(PlotAxisLabel("Total Population (Grown (+), Ungrown (-))"))
                    self.title = "Site Analysis: Sites vs Total Population"
                elif plotting_mode == "Events per Step":
                    if param_value is not None:
                        _set_y(
                            PlotAxisLabel(
                                f"Events at {param_name.capitalize()}={param_value:.2f} (Growth (+), Dissolution (-))"
                            )
                        )
                        self.title = f"Site Analysis: Sites vs Events (t={time_index})"
                    else:
                        _set_y(PlotAxisLabel("Events per Step (Growth (+), Dissolution (-))"))
                        self.title = "Site Analysis: Sites vs Events per Step"
                elif plotting_mode == "Population per Step":
                    if param_value is not None:
                        _set_y(
                            PlotAxisLabel(
                                f"Population at {param_name.capitalize()}={param_value:.2f} (Grown (+), Ungrown (-))"
                            )
                        )
                        self.title = f"Site Analysis: Sites vs Population (t={time_index})"
                    else:
                        _set_y(PlotAxisLabel("Population per Step (Grown (+), Ungrown (-))"))
                        self.title = "Site Analysis: Sites vs Population per Step"

            elif permutation == 2:
                if plotting_mode in ["Total Events", "Total Population"]:
                    _set_x(PlotAxisLabel("Total Events (Growth (+), Dissolution (-))"))
                    _set_y(PlotAxisLabel("Total Population (Grown (+), Ungrown (-))"))
                    self.title = "Site Analysis: Events vs Population"
                else:
                    if param_value is not None:
                        _set_x(
                            PlotAxisLabel(
                                f"Events at {param_name.capitalize()}={param_value:.2f} (Growth (+), Dissolution (-))"
                            )
                        )
                        _set_y(
                            PlotAxisLabel(
                                f"Population at {param_name.capitalize()}={param_value:.2f} (Grown (+), Ungrown (-))"
                            )
                        )
                        self.title = f"Site Analysis: Events vs Population (t={time_index})"
                    else:
                        _set_x(PlotAxisLabel("Events per Step (Growth (+), Dissolution (-))"))
                        _set_y(PlotAxisLabel("Population per Step (Grown (+), Ungrown (-))"))
                        self.title = "Site Analysis: Events vs Population per Step"
            return

        if self.plot_type == "Zingg":
            _set_x(PlotAxisLabel("S:M"))
            _set_y(PlotAxisLabel("M:L"))
            self.title = "Zingg Diagram"
        if self.plot_type == "CDA":
            ratio_columns = [col for col in self.df.columns if col.startswith("Ratio_")]
            _set_x(PlotAxisLabel(ratio_columns[0].replace("Ratio_", "")))
            _set_y(PlotAxisLabel(ratio_columns[1].replace("Ratio_", "")))
            self.title = "Crystallographic Face-to-Face Distance Ratios"
        if self.plot_type == "SA:Vol Ratio":
            _set_x(PlotAxisLabel("Surface Area"))
            _set_y(PlotAxisLabel("Volume"))
            self.title = "Surface Area to Volume Ratio"
        if self.plot_type == "Growth Rates":
            x_col = getattr(self, "gr_x_col", "Supersaturation")
            if x_col == "Supersaturation" or x_col.startswith("starting_delmu"):
                _set_x(PlotAxisLabel(r"$\Delta\mu$", "kcal/mol"))
            else:
                _set_x(PlotAxisLabel.from_column(x_col))
            _set_y(PlotAxisLabel("Relative Growth Rate"))
            self.title = f"Growth Rates vs {x_col.replace('_', ' ').strip()}"
        if self.plot_type == "Heatmap":
            self.title = "Heatmap"
            _set_x(PlotAxisLabel.from_column(self.custom_x) if self.custom_x else PlotAxisLabel())
            _set_y(
                PlotAxisLabel.from_column(self.custom_y[0])
                if self.custom_y and len(self.custom_y) > 0
                else PlotAxisLabel()
            )
        if self.plot_type == "Custom":
            self.title = ""
            _set_x(PlotAxisLabel.from_column(self.custom_x) if self.custom_x else PlotAxisLabel())
            if len(self.custom_y) == 1:
                _set_y(PlotAxisLabel.from_column(self.custom_y[0]))
            elif len(self.custom_y) <= 3:
                _set_y(PlotAxisLabel(", ".join([format_label(y) for y in self.custom_y])))
            else:
                _set_y(PlotAxisLabel("Multiple columns..."))

    def _set_c(self):
        self.c_data = None
        self.c_name = None
        self.c_mapped_name = None  # Track the actual variable name when mapping file_prefix
        if self.plot_type == "Site Analysis":
            # Use variable dropdown to control color for Site Analysis
            if self.variable != "None" and self.variable and hasattr(self, "_site_metadata"):
                # Special handling for "filter" - binary color based on filter match
                if self.variable == "filter":
                    interaction_filters = self.interaction_filters
                    data_filters = self.data_filters

                    if interaction_filters or data_filters:
                        # Color sites based on whether they match ALL filters (interaction AND data)
                        c_values = []
                        for site_meta in self._site_metadata:
                            matches = True

                            # Check interaction filters if present
                            if interaction_filters:
                                site_interactions = site_meta.get("interactions", {})
                                matches = matches and self._check_interaction_filter(
                                    site_interactions, interaction_filters
                                )

                            # Check data filters if present
                            # Note: site_meta already contains "site_number" from metadata creation
                            if data_filters and matches:
                                for filter_config in data_filters:
                                    if not self._check_site_data_filter(site_meta, filter_config):
                                        matches = False
                                        break

                            c_values.append(1 if matches else 0)
                        self.c_data = np.array(c_values)
                        self.c_name = self.variable
                    else:
                        # No filters selected, cannot color by filter
                        self.c_data = None
                        self.c_name = None
                    return

                # Extract coloring data from site metadata
                c_values = []
                for site_meta in self._site_metadata:
                    if self.variable in site_meta:
                        c_values.append(site_meta[self.variable])
                    else:
                        c_values.append(None)

                # Special handling for file_prefix when we have a summary file
                if self.variable == "file_prefix" and self.summary_df is not None:
                    # Map file prefixes to summary file values
                    mapped_values = []
                    if hasattr(self.parent(), "xyz_files"):
                        xyz_files = self.parent().xyz_files
                        # Create mapping from file_prefix to index
                        prefix_to_index = {
                            xyz_file.stem: idx for idx, xyz_file in enumerate(xyz_files)
                        }

                        # Find first varying column in summary
                        varying_cols = []
                        for col in self.summary_df.columns:
                            if self.summary_df[col].nunique() > 1 and pd.api.types.is_numeric_dtype(
                                self.summary_df[col]
                            ):
                                varying_cols.append(col)

                        if varying_cols:
                            varying_col = varying_cols[0]
                            self.c_mapped_name = varying_col

                            for site_meta in self._site_metadata:
                                file_prefix = site_meta.get("file_prefix")
                                if file_prefix in prefix_to_index:
                                    idx = prefix_to_index[file_prefix]
                                    if idx < len(self.summary_df):
                                        mapped_values.append(self.summary_df.iloc[idx][varying_col])
                                    else:
                                        mapped_values.append(0)
                                else:
                                    mapped_values.append(0)

                    if mapped_values:
                        self.c_data = np.array(mapped_values)
                    else:
                        # Fallback to factorized file_prefix
                        self.c_data = pd.factorize(c_values)[0]
                    self.c_name = self.variable
                else:
                    # Regular variable
                    self.c_data = np.array(c_values)
                    self.c_name = self.variable
        elif self.plot_type == "Heatmap":
            # For heatmap, use custom_c as the value, default to Frame Index
            if self.custom_c and self.custom_c != "None":
                self.c_data = self.df[self.custom_c]
                self.c_name = self.custom_c
            elif "Frame Index" in self.df.columns:
                self.c_data = self.df["Frame Index"]
                self.c_name = "Frame Index"
                self.custom_c = "Frame Index"
        elif self.plot_type == "4D":
            # custom_c holds the Z axis column — must NOT be cleared here.
            # variable drives the colour (4th) dimension.
            if (
                self.variable != "None"
                and self.variable
                and self.df is not None
                and self.variable in self.df.columns
            ):
                self.c_data = self.df[self.variable]
                self.c_name = self.variable
        elif self.plot_type != "Custom":
            self.custom_c = "None"
            if self.variable != "None" and self.variable:
                self.c_data = self.df[self.variable]
                self.c_name = self.variable
        elif self.plot_type == "Custom":
            self.variable = "None"
            if self.custom_c != "None" and self.custom_c:
                self.c_data = self.df[self.custom_c]
                self.c_name = self.custom_c

        if self.c_data is None:
            return
        # if c_data is not numerical
        if self.c_data.dtype.kind not in "biufc":
            self.c_data = pd.factorize(self.c_data)[0]

    def _set_c_label(self):
        variable = self.variable
        if self.plot_type == "Custom":
            variable = self.custom_c
        elif self.plot_type == "Heatmap":
            self.c_label = (
                PlotAxisLabel.from_column(self.c_name) if self.c_name else PlotAxisLabel()
            )
            return

        self.c_label = PlotAxisLabel()
        if not variable or variable == "None":
            return
        if variable == "file_prefix":
            if hasattr(self, "c_mapped_name") and self.c_mapped_name:
                self.c_label = PlotAxisLabel.from_column(self.c_mapped_name)
            else:
                self.c_label = PlotAxisLabel("File / Dataset")
        elif variable == "filter":
            self.c_label = PlotAxisLabel("Filter Match")
        else:
            self.c_label = PlotAxisLabel.from_column(variable)

    def plot(self):
        self.plot_objects = {}
        logger.debug("Plotting called!")
        logger.debug(
            "Plot Called!\nType: %s, Permutation: %s, Variable: %s, \nCustom X: %s, Custom Y: %s, Custom C: %s, \nGrid: %s, Legend: %s, Point Size: %s, ",
            self.plot_type,
            self.permutation,
            self.variable,
            self.custom_x,
            self.custom_y,
            self.custom_c,
            self.grid,
            self.show_legend,
            self.point_size,
        )
        self.figure.clear()
        if self.plot_type == "4D":
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            self.ax = self.figure.add_subplot(111, projection="3d")
        else:
            self.ax = self.figure.add_subplot(111)
        self.ax.clear()
        self.canvas.draw()

        self.ax.set_xlabel(str(self.custom_xlabel or self.x_label))
        self.ax.set_ylabel(str(self.custom_ylabel or self.y_label))
        self.ax.set_title(self.custom_title if self.custom_title else self.title)

        if self.grid:
            self.ax.grid(True)
        else:
            self.ax.grid(False)

        if self.plot_type != "4D":
            self.annot = self.ax.annotate(
                "",
                xy=(-1, -1),
                xytext=(-20, 20),
                textcoords="offset points",
                ha="center",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
                zorder=20,
            )
            self.annot.set_visible(False)
        else:
            self.annot = None

        # Hierarchical clustering — renders its own figure layout
        if self.cluster_mat:
            self._plot_hierarchical_clustering()
            self.canvas.draw()
            return

        # Handle 4D plot type — 3D scatter with colour dimension
        if self.plot_type == "4D":
            if self.x_data is None or self.y_data is None or self.z_data is None:
                logger.warning("4D mode requires X, Y, and Z axis data to be selected.")
                self.canvas.draw()
                return
            self._plot_4d()
            # Apply axis inversions for 4D
            if self.checkbox_invert_x.isChecked():
                self.ax.invert_xaxis()
            if self.checkbox_invert_y.isChecked():
                self.ax.invert_yaxis()
            if self.checkbox_invert_z.isChecked():
                self.ax.invert_zaxis()
            self.canvas.draw()
            return

        # Handle Heatmap plot type separately
        if self.plot_type == "Heatmap":
            if self.x_data is None or self.y_data is None or self.c_data is None:
                logger.warning("No data for heatmap plotting!")
                return
            self._plot_heatmap()
        else:
            if self.x_data is None or self.y_data is None:
                logger.warning("No data for plotting!")
                return

            markers = [
                "o",  # Circle
                "^",  # Triangle up
                "s",  # Square
                "D",  # Diamond
                "v",  # Triangle down
                "p",  # Pentagon
                "*",  # Star
                "h",  # Hexagon
                "+",  # Plus
                "x",  # Cross
            ]

            # Plot the data
            # Check if y_data has ndim attribute (numpy array or pandas Series/DataFrame)
            if hasattr(self.y_data, "ndim"):
                if self.y_data.ndim == 1:
                    # 1D y_data
                    self._plot(x=self.x_data, y=self.y_data, c=self.c_data)
                elif self.y_data.ndim == 2 and self.y_data.shape[1] > 1:
                    # y_data with multiple columns - only for DataFrame-based plots
                    if self.df is not None:
                        if self.plot_type == "Growth Rates" and self.gr_grouped:
                            self._plot_gr_grouped()
                            self._set_legend() if self.show_legend else None
                        else:
                            use_variable = bool(self.variable and self.variable != "None")
                            line = (
                                True
                                if (self.plot_type == "Growth Rates" and not use_variable)
                                else False
                            )
                            for i, y in enumerate(self.y_data):
                                # Plot original data
                                show_original = (
                                    self.plot_type == "Growth Rates"
                                    and self.smoothing_legend_mode
                                    in ["Show Both Original and Processed", "Show Original Only"]
                                )

                                if show_original or y not in self.smoothing_configs:
                                    self._plot(
                                        x=self.x_data,
                                        y=self._ensure_pd_series(self.df[y]),
                                        c=self.c_data,
                                        add_line=line,
                                        label=y
                                        if y not in self.smoothing_configs
                                        else f"{y} (original)",
                                        marker=markers[i % len(markers)],
                                    )

                                # Plot processed (smoothed/interpolated/extrapolated) data
                                if self.plot_type == "Growth Rates" and y in self.smoothing_configs:
                                    show_processed = self.smoothing_legend_mode in [
                                        "Show Both Original and Processed",
                                        "Show Processed Only",
                                    ]

                                    if show_processed:
                                        x_processed, y_processed = self._apply_smoothing(
                                            self.x_data,
                                            self._ensure_pd_series(self.df[y]),
                                            self.smoothing_configs[y],
                                        )

                                        self._plot(
                                            x=x_processed,
                                            y=y_processed,
                                            c=None,  # Don't use color for processed data
                                            add_line=True,  # Always use line for processed data
                                            label=f"{y} (processed)",
                                            marker=markers[i % len(markers)],
                                        )

                            self._set_legend() if self.show_legend else None
            else:
                # Fallback for data without ndim (shouldn't happen, but handle gracefully)
                self._plot(x=self.x_data, y=self.y_data, c=self.c_data)

        if self.plot_type == "Zingg":
            self.ax.axhline(y=0.66, color="black", linestyle="--")
            self.ax.axvline(x=0.66, color="black", linestyle="--")
            self.ax.set_xlim(0, 1.0)
            self.ax.set_ylim(0, 1.0)

        # Add colorbar after plotting all data
        if self.c_data is not None and self.plot_objects:
            # Take the frist scatter object to attach the colorbar
            scatter = list(self.plot_objects.values())[0].scatter
            self.cbar = self.figure.colorbar(scatter)
            self.cbar.set_label(str(self.custom_cbar_label or self.c_label))
            self.cbar.ax.set_zorder(-1)

            # Set discrete ticks for binary filter variable
            if self.variable == "filter":
                self.cbar.set_ticks([0, 1])
                self.cbar.set_ticklabels(["Unfiltered", "Filtered"])

        # Apply 2D axis inversions (all non-4D modes)
        if self.checkbox_invert_x.isChecked():
            self.ax.invert_xaxis()
        if self.checkbox_invert_y.isChecked():
            self.ax.invert_yaxis()

        self.canvas.draw()

    # def _plot(
    #     self, x, y, c=None, cmap="plasma", add_line=False, label=None, marker="o"
    # ):
    #     cmap = None if c is None else cmap
    #     label = y.name if label is None else label
    #     line = None
    #     logger.info("X SIZE: %s    Y SIZE: %s", self.x_data.size, self.y_data.size)
    #     scatter = self.ax.scatter(
    #         x=x,
    #         y=y,
    #         c=c,
    #         cmap=cmap,
    #         s=self.point_size,
    #         label=label if self.plot_type != "Growth Rates" else None,
    #         marker=marker,
    #     )
    #     if add_line:
    #         (line,) = self.ax.plot(x, y, label=label)

    #     plot_object = self.plot_obj_tuple(scatter=scatter, line=line, trendline=None)
    #     self.plot_objects[label] = plot_object

    def _plot(self, x, y, c=None, cmap="plasma", add_line=False, label=None, marker="o"):
        # Handle label: use y.name for pandas Series, or plot_type for numpy arrays
        if label is None:
            if hasattr(y, "name"):
                label = y.name
            else:
                label = self.plot_type if self.plot_type else "data"

        if self.covmat:
            self._plot_covmat()
        elif self.plot_type == "Heatmap":
            self._plot_heatmap()
        else:
            self._plot_scatter(x, y, c, cmap, add_line, label, marker)

    def _plot_gr_grouped(self):
        """Plot grouped Growth Rates as mean ± std error bars, one series per direction."""
        markers = ["o", "^", "s", "D", "v", "p", "*", "h", "+", "x"]
        use_variable = (
            self.c_data is not None
            and self.variable
            and self.variable != "None"
            and self.variable in self.df.columns
        )

        if use_variable:
            x_col = self.gr_x_col if self.gr_x_col in self.df.columns else "Supersaturation"
            var_by_group = self.df.groupby(x_col)[self.variable].mean()
            c_vals = var_by_group.reindex(self.x_data).to_numpy()
            vmin, vmax = np.nanmin(c_vals), np.nanmax(c_vals)
        else:
            c_vals = None

        for i, direction in enumerate(self.directions):
            y_mean = self.y_data[direction].to_numpy()
            y_err = self.gr_y_err[direction].to_numpy() if self.gr_y_err is not None else None

            if use_variable:
                # Draw error caps in neutral grey so the colormap drives colour
                if y_err is not None:
                    self.ax.errorbar(
                        self.x_data,
                        y_mean,
                        yerr=y_err,
                        fmt="none",
                        ecolor="grey",
                        capsize=4,
                        alpha=0.5,
                    )
                sc = self.ax.scatter(
                    self.x_data,
                    y_mean,
                    c=c_vals,
                    cmap=self.cmap,
                    vmin=vmin,
                    vmax=vmax,
                    marker=markers[i % len(markers)],
                    s=self.point_size,
                    label=direction,
                    zorder=5,
                )
                self.plot_objects[direction] = self.plot_obj_tuple(
                    scatter=sc, line=None, trendline=None
                )
            else:
                self.ax.errorbar(
                    self.x_data,
                    y_mean,
                    yerr=y_err,
                    fmt=markers[i % len(markers)],
                    linestyle="-",
                    capsize=4,
                    label=direction,
                )

    def _plot_scatter(self, x, y, c, cmap, add_line, label, marker):
        # Use universal colour scheme if color data is present
        if c is not None:
            # Use binary colormap for filter variable (colorblind-safe: orange for unfiltered, blue for filtered)
            if self.variable == "filter":
                cmap = matplotlib.colors.ListedColormap(["#E69F00", "#56B4E9"])
            else:
                cmap = self.cmap
        else:
            cmap = None

        # Prepare scatter arguments
        scatter_kwargs = {
            "x": x,
            "y": y,
            "c": c,
            "cmap": cmap,
            "s": self.point_size,
            "label": label
            if (self.plot_type != "Growth Rates" or bool(self.variable and self.variable != "None"))
            else None,
            "marker": marker,
        }

        # Set vmin and vmax for binary filter coloring
        if c is not None and self.variable == "filter":
            scatter_kwargs["vmin"] = 0
            scatter_kwargs["vmax"] = 1

        scatter = self.ax.scatter(**scatter_kwargs)
        line = None
        if add_line:
            (line,) = self.ax.plot(x, y, label=label)

        if self.zingg:
            # Zingg specific modifications: dashed lines at 2/3 on both axes
            self.ax.axhline(y=2 / 3, color="gray", linestyle="--", linewidth=1)
            self.ax.axvline(x=2 / 3, color="gray", linestyle="--", linewidth=1)
            # Set x and y limits for the Zingg plot
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)

        plot_object = self.plot_obj_tuple(scatter=scatter, line=line, trendline=None)
        self.plot_objects[label] = plot_object

    def _update_axes_for_clustering(self):
        """When hierarchical clustering is enabled: lock x to interactions, restrict y to metrics."""
        if self.df is None:
            return
        interaction_cols = [c for c in self.interaction_columns if c != "None"]
        non_interaction_cols = [c for c in self.df.columns if c not in self.interaction_columns]
        self.custom_plot_widget.set_x_locked(interaction_cols)
        self.custom_plot_widget.yAxisListWidget.clear()
        self.custom_plot_widget.yAxisListWidget.addItems(non_interaction_cols)
        self.custom_plot_widget.y_axis_label.setText("Y-axis metrics (check multiple)")

    def _restore_axes_from_clustering(self):
        """Restore normal x/y axis widgets after disabling hierarchical clustering."""
        if self.df is None:
            return
        self.custom_plot_widget.restore_x(list(self.df.columns))
        self.custom_plot_widget.yAxisListWidget.clear()
        self.custom_plot_widget.yAxisListWidget.addItems(self.df.columns)
        self.custom_plot_widget.y_axis_label.setText("Y-axis (check multiple)")

    def _plot_hierarchical_clustering(self):
        """Plot a dendrogram clustering interactions by their correlation profiles across selected metrics."""

        if self.df is None:
            logger.warning("Hierarchical clustering requires DataFrame data")
            return

        interaction_cols = [
            c for c in self.interaction_columns if c != "None" and c in self.df.columns
        ]
        metric_cols = [c for c in self.custom_y if c in self.df.columns]

        if len(interaction_cols) < 2:
            logger.warning("Need at least 2 interaction columns for hierarchical clustering")
            QMessageBox.warning(
                self,
                "Hierarchical Clustering",
                "At least 2 interaction columns are required for hierarchical clustering.\n\n"
                "No interaction columns were detected in the current dataset.",
            )
            return
        if not metric_cols:
            logger.warning("No metric columns selected for hierarchical clustering")
            QMessageBox.warning(
                self,
                "Hierarchical Clustering",
                "Please select at least one metric in the Y-axis list to cluster by.",
            )
            return

        # Build profile matrix: rows = interactions, cols = metrics (Pearson r values)
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_rows = []
        for interaction in interaction_cols:
            row = []
            for metric in metric_cols:
                try:
                    r = numeric_df[interaction].corr(numeric_df[metric])
                except Exception:
                    r = np.nan
                row.append(0.0 if (r is None or np.isnan(r)) else r)
            corr_rows.append(row)

        data = np.array(corr_rows)

        linkage = hierarchy.linkage(pdist(data, metric="euclidean"), method="ward")

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        hierarchy.dendrogram(
            linkage,
            labels=interaction_cols,
            orientation="top",
            ax=self.ax,
            leaf_rotation=45,
        )

        title = (
            self.custom_title if self.custom_title else "Hierarchical Clustering of Interactions"
        )
        self.ax.set_title(title)
        self.ax.set_ylabel("Distance (Ward)")
        self.ax.set_xlabel(f"Interactions  [clustered by: {', '.join(metric_cols)}]")
        self.figure.tight_layout()

    def _plot_covmat(self):
        # Correlation matrix only works with DataFrame-based data
        if self.df is None:
            logger.warning("Cannot plot correlation matrix without DataFrame data")
            return

        # Calculate correlation matrix for the interactions
        vals = self.df[self.custom_y]
        try:
            corr_matrix = vals.corr()
            logger.debug("Correlation matrix:\n%s", corr_matrix)
        except ValueError as ve:
            logger.error("Make sure a valid data set is provided, Encountered: %s", ve)
            return

        # Clear existing plot
        self.ax.clear()

        # Plot correlation matrix as a heatmap
        im = self.ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto")

        # Add colorbar if it doesn't exist
        if self.cbar is None:
            self.cbar = self.figure.colorbar(im, ax=self.ax)
            self.cbar.set_label("Correlation")
        else:
            self.cbar.update_normal(im)

        # Set ticks and labels
        tick_positions = np.arange(len(self.custom_y))
        self.ax.set_xticks(tick_positions)
        self.ax.set_yticks(tick_positions)
        self.ax.set_xticklabels(self.custom_y, rotation=45, ha="right")
        self.ax.set_yticklabels(self.custom_y)

        # Add text annotations
        for i in range(len(self.custom_y)):
            for j in range(len(self.custom_y)):
                self.ax.text(
                    j,
                    i,
                    f"{corr_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        # Set title
        self.ax.set_title("Correlation Matrix")

        # Adjust layout to prevent clipping of tick-labels
        self.figure.tight_layout()

        return im

    def _plot_heatmap(self):
        """Plot a 2D grid heatmap with x, y axes and c as the color value."""
        if self.x_data is None or self.y_data is None or self.c_data is None:
            logger.warning("Insufficient data for heatmap!")
            return

        # Clear existing plot
        self.ax.clear()

        # Create a DataFrame for pivoting
        heatmap_df = pd.DataFrame({"x": self.x_data, "y": self.y_data, "value": self.c_data})

        # Create pivot table for heatmap
        try:
            pivot_table = heatmap_df.pivot_table(
                values="value",
                index="y",
                columns="x",
                aggfunc="mean",  # Use mean if there are duplicate x,y pairs
            )
        except Exception as e:
            logger.error("Failed to create pivot table for heatmap: %s", e)
            return

        # Plot heatmap using imshow
        im = self.ax.imshow(pivot_table, cmap=self.cmap, aspect="auto", origin="lower")

        # Set ticks and labels
        x_ticks = np.arange(len(pivot_table.columns))
        y_ticks = np.arange(len(pivot_table.index))

        # Show every nth tick to avoid crowding
        n_xticks = min(10, len(x_ticks))
        n_yticks = min(10, len(y_ticks))
        x_tick_indices = np.linspace(0, len(x_ticks) - 1, n_xticks, dtype=int)
        y_tick_indices = np.linspace(0, len(y_ticks) - 1, n_yticks, dtype=int)

        self.ax.set_xticks(x_tick_indices)
        self.ax.set_yticks(y_tick_indices)
        self.ax.set_xticklabels(
            [f"{pivot_table.columns[i]:.2f}" for i in x_tick_indices], rotation=45, ha="right"
        )
        self.ax.set_yticklabels([f"{pivot_table.index[i]:.2f}" for i in y_tick_indices])

        # Add colorbar
        effective_cbar_label = str(self.custom_cbar_label or self.c_label) or "Value"
        if self.cbar is None:
            self.cbar = self.figure.colorbar(im, ax=self.ax)
            self.cbar.set_label(effective_cbar_label)
        else:
            self.cbar.update_normal(im)
            self.cbar.set_label(effective_cbar_label)

        # Set labels and title
        self.ax.set_xlabel(str(self.custom_xlabel or self.x_label))
        self.ax.set_ylabel(str(self.custom_ylabel or self.y_label))
        self.ax.set_title(self.custom_title if self.custom_title else self.title)

        if self.grid:
            self.ax.grid(True, color="white", linewidth=0.5)

        # Adjust layout
        self.figure.tight_layout()

        return im

    def _plot_4d(self):
        """Plot a 3D scatter coloured by a 4th dimension (variables_combobox)."""
        x = np.asarray(self.x_data, dtype=float)
        y = np.asarray(self.y_data, dtype=float)
        z = np.asarray(self.z_data, dtype=float)

        # Colour by 4th variable (from variables_combobox) or fall back to Z values
        c_col = self.variable if self.variable and self.variable != "None" else None
        if c_col and self.df is not None and c_col in self.df.columns:
            c = np.asarray(self.df[c_col], dtype=float)
            c_label = c_col
        else:
            c = z
            c_label = self.custom_c if self.custom_c else "Z"

        # Mask NaN rows
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(c)
        x, y, z, c = x[mask], y[mask], z[mask], c[mask]

        cmap = self.cmap_combobox.currentText() if hasattr(self, "cmap_combobox") else "viridis"

        sc = self.ax.scatter(x, y, z, c=c, cmap=cmap, s=self.point_size, depthshade=True)
        self.figure.colorbar(sc, ax=self.ax, label=c_label, shrink=0.6)

        x_label = self.custom_x or "X"
        y_label = self.custom_y[0] if self.custom_y else "Y"
        z_label = self.custom_c or "Z"
        self.ax.set_xlabel(str(self.custom_xlabel or x_label))
        self.ax.set_ylabel(str(self.custom_ylabel or y_label))
        self.ax.set_zlabel(z_label)
        self.ax.set_title(self.custom_title if self.custom_title else "4D Plot")

        self.ax.view_init(
            elev=self.slider_elev.value(),
            azim=self.slider_azim.value(),
        )

        if self.grid:
            self.ax.grid(True)

        # tight_layout is unreliable with 3D axes — suppress the warning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout()
            except Exception:
                pass

    def _set_legend(self):
        # Create a legend
        legend = self.ax.legend()

        for legend_line in legend.get_lines():
            legend_line.set_picker(5)

        legend.figure.canvas.mpl_connect("pick_event", self.on_legend_click)

    def _handle_plot_checkboxes(self, checkbox):
        if checkbox.isChecked():
            if checkbox == self.checkbox_zingg:
                self.checkbox_corr_mat.blockSignals(True)
                self.checkbox_cluster_mat.blockSignals(True)
                self.checkbox_corr_mat.setChecked(False)
                self.checkbox_cluster_mat.setChecked(False)
                self.checkbox_corr_mat.blockSignals(False)
                self.checkbox_cluster_mat.blockSignals(False)
            elif checkbox == self.checkbox_corr_mat:
                self.checkbox_zingg.blockSignals(True)
                self.checkbox_cluster_mat.blockSignals(True)
                self.checkbox_zingg.setChecked(False)
                self.checkbox_cluster_mat.setChecked(False)
                self.checkbox_zingg.blockSignals(False)
                self.checkbox_cluster_mat.blockSignals(False)
            elif checkbox == self.checkbox_cluster_mat:
                self.checkbox_zingg.blockSignals(True)
                self.checkbox_corr_mat.blockSignals(True)
                self.checkbox_zingg.setChecked(False)
                self.checkbox_corr_mat.setChecked(False)
                self.checkbox_zingg.blockSignals(False)
                self.checkbox_corr_mat.blockSignals(False)
                self._update_axes_for_clustering()
        else:
            if checkbox == self.checkbox_cluster_mat:
                self._restore_axes_from_clustering()
        self.trigger_plot()

    def update_annot(self, scatter, colour_data, column_name, ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        x, y = pos

        _sim_id = "N/A"

        # Handle Site Analysis plots differently
        if self.plot_type == "Site Analysis" and hasattr(self, "_site_metadata"):
            if ind["ind"][0] < len(self._site_metadata):
                site_meta = self._site_metadata[ind["ind"][0]]
                site_num = site_meta["site_number"]
                coordination = site_meta["coordination"]
                energy = site_meta["energy"]
                occupation_status = "Grown" if site_meta["occupation"] else "Ungrown"
                file_prefix = site_meta.get("file_prefix", "N/A")

                # Get current plotting mode, time point info, and permutation
                plotting_mode = self.time_series_widget.get_plotting_mode()
                param_name, param_value, time_index = (
                    self.time_series_widget.get_current_time_point()
                )
                permutation = self.permutation

                # Common information for all permutations
                base_info = f"File: {file_prefix}\nSite Number: {site_num}\n"

                # Permutation-specific information
                if permutation == 0:
                    # Events/Population vs Energy
                    if plotting_mode == "Total Events":
                        current_value = site_meta.get("total_events", 0)
                        text = (
                            base_info + f"Total Events: {current_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Total Population":
                        current_value = site_meta.get("total_population", 0)
                        text = (
                            base_info + f"Total Population: {current_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Events per Step":
                        total_value = site_meta.get("total_events", "N/A")
                        current_value = abs(x)
                        text = (
                            base_info + f"Events (t={time_index}): {current_value:.1f}\n"
                            f"Total Events: {total_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Population per Step":
                        total_value = site_meta.get("total_population", "N/A")
                        current_value = abs(x)
                        text = (
                            base_info + f"Population (t={time_index}): {current_value:.1f}\n"
                            f"Total Population: {total_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )

                elif permutation == 1:
                    # Sites vs Events/Population
                    if plotting_mode == "Total Events":
                        current_value = site_meta.get("total_events", 0)
                        text = (
                            base_info + f"Total Events: {current_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Total Population":
                        current_value = site_meta.get("total_population", 0)
                        text = (
                            base_info + f"Total Population: {current_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Events per Step":
                        total_value = site_meta.get("total_events", "N/A")
                        current_value = abs(y)
                        text = (
                            base_info + f"Events (t={time_index}): {current_value:.1f}\n"
                            f"Total Events: {total_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    elif plotting_mode == "Population per Step":
                        total_value = site_meta.get("total_population", "N/A")
                        current_value = abs(y)
                        text = (
                            base_info + f"Population (t={time_index}): {current_value:.1f}\n"
                            f"Total Population: {total_value}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )

                elif permutation == 2:
                    # Events vs Population
                    events_value = abs(x)
                    population_value = abs(y)
                    if plotting_mode in ["Total Events", "Total Population"]:
                        text = (
                            base_info + f"Total Events: {events_value:.1f}\n"
                            f"Total Population: {population_value:.1f}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )
                    else:
                        total_events = site_meta.get("total_events", "N/A")
                        total_population = site_meta.get("total_population", "N/A")
                        text = (
                            base_info + f"Events (t={time_index}): {events_value:.1f}\n"
                            f"Population (t={time_index}): {population_value:.1f}\n"
                            f"Total Events: {total_events}\n"
                            f"Total Population: {total_population}\n"
                            f"Energy: {energy:.2f}\n"
                            f"Coordination: {coordination}\n"
                            f"Status: {occupation_status}"
                        )

                self.annot.set_text(text)
                self.annot.get_bbox_patch().set_alpha(0.4)
            return

        if not (self.df is not None and ind["ind"][0] < len(self.df)):
            return

        row_data = self.df.iloc[ind["ind"][0]]
        # print(self.df.columns)

        _sim_id = "N/A"
        # Determine the correct index based on column presence
        if "solvent" in self.df.columns:
            index_key = "solvent"
        elif "Solvent" in self.df.columns:
            index_key = "Solvent"
        else:
            index_key = "Simulation Number"

        # Attempt to handle the simulation ID based on the index_key
        try:
            _sim_id = int(row_data[index_key] - 1)
        except KeyError:
            # Fallback to the first column if index_key is not found
            _sim_id = row_data.iloc[0]
        except TypeError:
            # Handle cases where subtraction is not possible due to data type issues
            _sim_id = row_data[index_key]

        # Check if colour_data is available
        if colour_data is not None:
            color_val = colour_data[ind["ind"][0]]
            text = f"Index: {_sim_id}\n x: {x:.2f}\n y: {y:.2f}\n {column_name}: {color_val:.2f}"
        else:
            text = f"Index: {_sim_id}\n x: {x:.2f}\n y: {y:.2f}"

        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_alpha(0.4)

    def on_hover(self, event):
        vis = False
        if self.annot:
            vis = self.annot.get_visible()
        try:
            if event.inaxes != self.ax:
                return

            cont = False

            # Function to handle hover event for a scatter plot
            def handle_hover(scatter):
                nonlocal cont
                cont, ind = scatter.contains(event)
                if cont:
                    self.update_annot(
                        scatter,
                        self.c_data,
                        self.c_name,
                        ind,
                    )
                    if self.annot is not None:
                        self.annot.set_visible(True)
                    self.figure.canvas.draw_idle()

            for plot in self.plot_objects.values():
                handle_hover(scatter=plot.scatter)

            # Hide annotation if no scatter plot contains the event
            if not cont and vis and self.annot is not None:
                self.annot.set_visible(False)
                self.figure.canvas.draw_idle()

        except NameError:
            pass

    def on_legend_click(self, event):
        legend_line = event.artist
        p = self.plot_objects[legend_line.get_label()]
        vis = not p.line.get_visible()
        p.line.set_visible(vis)
        p.scatter.set_visible(vis)
        legend_line.set_alpha(1.0 if vis else 0.2)
        self.figure.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Track if we clicked on any point
        clicked_on_point = False

        def handle_scatter(scatter):
            nonlocal clicked_on_point
            # Check if scatter plot exists and handle click event
            if scatter is not None:
                is_contained, ind = scatter.contains(event)
                if is_contained:
                    clicked_on_point = True
                    self.handle_click(scatter, self.c_data, self.c_name, ind)

        for plot in self.plot_objects.values():
            handle_scatter(scatter=plot.scatter)

        # If we didn't click on any point, handle whitespace click (deselection)
        if not clicked_on_point:
            self.handle_whitespace_click()

    def handle_click(self, scatter, _colour_data, _column_name, ind):
        """Handle click event on scatter plot points."""
        # index of the clicked point
        point_index = ind["ind"][0]

        # Extracting the x and y data of the clicked point
        x, y = scatter.get_offsets()[point_index]

        # Handle Site Analysis plot clicks
        if self.plot_type == "Site Analysis" and hasattr(self, "_site_metadata"):
            if point_index < len(self._site_metadata):
                site_meta = self._site_metadata[point_index]
                site_num = site_meta["site_number"]
                file_prefix = site_meta.get("file_prefix", "N/A")
                logger.info(f"Clicked on Site Analysis point: file={file_prefix}, site={site_num}")

                # Emit signal to highlight this site in the visualization
                if self.signals and hasattr(self.signals, "highlight_site"):
                    self.signals.highlight_site.emit(int(site_num))
                return

        # Access the row in the self.df for non-site-analysis plots
        if self.df is not None and point_index < len(self.df):
            row_data = self.df.iloc[point_index]

            try:
                _sim_id = int(row_data["Simulation Number"] - 1)
            except KeyError:
                _sim_id = None
            if self.signals:
                self.signals.sim_id.emit(_sim_id)
            logger.info(f"Clicked on row {point_index}: {row_data}")
        else:
            logger.debug(f"Clicked on point {point_index} with coordinates (x={x}, y={y})")

    def handle_whitespace_click(self):
        """Handle click event on whitespace (not on any data point).

        This deselects any currently selected site type from a previous point click.
        """
        logger.info("Clicked on whitespace - deselecting site")

        # For Site Analysis mode, deselect the highlighted site
        if self.plot_type == "Site Analysis":
            if self.signals and hasattr(self.signals, "highlight_site"):
                # Emit signal with -1 or None to indicate deselection
                self.signals.highlight_site.emit(-1)
                logger.debug("Deselected site highlighting")
        else:
            # For other plot types, you could add similar deselection logic if needed
            logger.debug("Whitespace click in non-site-analysis mode")

    def _reset_figure_layout(self):
        """Reset the figure layout. Useful after switching between 4D (3D axes) and 2D modes."""
        self.figure.set_layout_engine(None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout()
            except Exception:
                pass
        self.canvas.draw()

    def toggle_trendline(self):
        logger.info(self.trendline_text)
        if not self.trendline:
            for i, (name, plot) in enumerate(self.plot_objects.items()):
                if plot.scatter is None:
                    return

                x = plot.scatter.get_offsets()[:, 0]
                y = plot.scatter.get_offsets()[:, 1]

                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                (trendline,) = self.ax.plot(x, p(x), "r--")
                self.plot_objects[name] = plot._replace(trendline=trendline)

                # Calculate the Pearson correlation coefficient
                r = np.corrcoef(x, y)[0, 1]

                equation_text = f"y = {z[0]:.2f}x + {z[1]:.2f}  [R={r:.2f}]"

                self.trendline_text.append(
                    self.ax.text(
                        0.05,
                        1 - (0.08 * i),
                        equation_text,
                        transform=self.ax.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                    )
                )
            self.trendline = True
            self.canvas.draw()
            self.button_add_trendline.setText("Remove Trendline")
        else:
            for name, plot in self.plot_objects.items():
                # Remove the trendline from the plot
                if plot.trendline:
                    plot.trendline.remove()
                    self.plot_objects[name] = plot._replace(trendline=None)

            # Remove the trendline text from the plot
            for text in self.trendline_text:
                text.remove()
            self.trendline_text = []

            self.canvas.draw()
            self.trendline = False
            self.button_add_trendline.setText("Add Trendline")

    def save(self):
        file_dialog = PlotSaveDialog(self.figure)
        if file_dialog.exec() == QDialog.Accepted:
            logger.debug("Closing PlotSaveDialog: accepted")
        else:
            logger.debug("Closing PlotSaveDialog: not accepted")

    def _ensure_pd_series(self, data):
        """
        Ensure the data is a Pandas Series.
        - If the input is a mutiple-column DataFrame, it's returned as is.
        - If the input is a single-column DataFrame, it's converted to a Series.
        - If the input is already a Series or 1D array, it's returned as is.
        """
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            return data.iloc[:, 0]
        return data

    def _apply_smoothing(self, x_data, y_data, config):
        """
        Apply smoothing, interpolation, and extrapolation to a data series.

        Args:
            x_data: X-axis data
            y_data: Y-axis data
            config: Configuration dictionary from smoothing dialog

        Returns:
            Tuple of (x_processed, y_processed)
        """
        try:
            x_proc, y_proc = process_series(x_data, y_data, config)
            return x_proc, y_proc
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}, returning original data")
            return np.array(x_data), np.array(y_data)


def main():
    app = QApplication(sys.argv)

    # Create a sample CSV data
    # csv_data = pd.DataFrame(
    #     {"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10], "c": ["A", "B", "A", "B", "A"]}
    # )
    csv_data = "path.csv"

    # Create an instance of PlottingDialog
    dialog = PlottingDialog(csv_data)

    # Show the dialog
    dialog.show()
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

import dataclasses
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from natsort import natsorted
from PySide6 import QtWidgets
from PySide6.QtCore import QObject, QSignalBlocker, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

from ..analysis.aspect_ratios import AspectRatio
from ..analysis.cluster_analysis import ClusterAnalysis
from ..analysis.growth_rates import GrowthRate
from ..analysis.gui_threads import WorkerCheckpoint, WorkerCheckpointExpand, WorkerXYZ
from ..analysis.site_analysis import SiteAnalysis
from ..fileio.cg_checkpoint import Checkpoint
from ..fileio.find_data import (
    find_info,
    locate_checkpoint_files,
    locate_xyz_files,
)
from ..fileio.structure import Structure
from ..fileio.log_setup import get_log_file_path, setup_logging
from ..fileio.pdb_export import write_docking_pdb
from ..fileio.opendir import open_directory
from ..fileio.xyz_file import CrystalCloud, DockingData
from .crystal_info import CrystalInfo
from .dialogs import CrystalInfoWidget, PlottingDialog
from .dialogs.about import AboutCGDialog
from .dialogs.atom_mode_settings_dialog import AtomModeSettingsDialog
from .dialogs.axes_settings_dialog import AxesSettingsDialog
from .dialogs.color_legend_dialog import ColorLegendDialog
from .dialogs.directions_dialog import DirectionsDialog
from .dialogs.keyboard_shortcuts import KeyboardShortcutsDialog
from .dialogs.lattice_dialog import LatticeParametersDialog
from .dialogs.planes_dialog import PlanesDialog
from .dialogs.render_settings_dialog import RenderSettingsDialog
from .dialogs.settings import SettingsDialog
from .dialogs.site_highlight_dialog import SiteHighlightDialog
from .dialogs.unit_cell_viewer_dialog import UnitCellViewerDialog
from .load_ui import Ui_MainWindow
from .shortcuts_manager import ShortcutsManager
from .utils.crystallography import Crystallography
from .visualisation.openGL import VisualisationWidget
from .widgets import (
    PointInfoToolbar,
    SimulationVariablesWidget,
    TextFileViewer,
    VisualizationSettingsWidget,
)
from .animation.keyframe import AnimationTimeline, Keyframe
from .animation.timeline_widget import KeyframeTimelineWidget

log_dict = {"basic": "DEBUG", "console": "INFO"}
setup_logging(**log_dict)
logger = logging.getLogger("CA:GUI")
logger.critical("LOGGING AT %s", log_dict)


class GUIWorkerSignals(QObject):
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
    sim_id = Signal(int)
    highlight_site = Signal(int)  # Signal for highlighting a site in visualization
    error = Signal(tuple)
    result = Signal(object)
    location = Signal(object)
    progress = Signal(int)
    message = Signal(str)


class MainWindow(QMainWindow, Ui_MainWindow):
    status_timeout = 1000
    crystalInfoChanged = Signal(CrystalInfo)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.setupUi(self)
        self.update_statusbar("CrystalAspects v1.0")

        self.threadpool = QThreadPool()

        self.welcome_message()

        # movie timer
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.next_frame)
        self.frame = 0
        self.frame_list = []
        self.playingState = False
        self.playingLoop = True
        self.playIcon = QIcon(":/material_icons/material_icons/png/play-custom.png")
        self.pauseIcon = QIcon(":/material_icons/material_icons/png/pause-custom.png")

        self.frame_slider.valueChanged.connect(self.update_movie)
        self.frame_spinBox.valueChanged.connect(self.update_movie)
        self.playPauseButton.clicked.connect(self.play_movie)

        self.worker_signals = GUIWorkerSignals()
        self.aspectratio = AspectRatio(signals=self.worker_signals)
        self.clusteranalysis = ClusterAnalysis(signals=self.worker_signals)
        self.clusteranalysis.dialog.applyColourRequested.connect(self._handle_apply_cluster_colour)
        self.clusteranalysis.dialog.showDataRequested.connect(self._show_cluster_analysis_data)
        self._active_colour_mode: str = "none"
        self._active_colour_cmap: str = "plasma"
        self.growthrate = GrowthRate(signals=self.worker_signals)
        self.siteanalysis = SiteAnalysis(signals=self.worker_signals)
        self.worker_signals.location.connect(self.set_output_folder)
        self.worker_signals.result.connect(self.set_results)
        self.worker_signals.started.connect(self.set_progressbar)
        self.worker_signals.finished.connect(self.clear_progressbar)
        self.worker_signals.progress.connect(self.update_progressbar)
        self.worker_signals.message.connect(self.set_message)
        self.worker_signals.sim_id.connect(self.update_sim_id)
        self.worker_signals.highlight_site.connect(self.highlight_site_in_visualization)
        self.worker_signals.error.connect(self.show_worker_error)
        # Other self variables
        self.crystal = None  # current CrystalCloud; None in checkpoint-grid mode
        self.sim_num: int | None = None
        self.input_folder: Path | None = None
        self.output_folder: Path | None = None
        self.xyz_files: List[Path] = []
        self.selected_directions: list = []
        self.plotting_csv: Path | None = None
        self.summ_df = None
        self.simulation_variables_widget = None
        self.plotting_dialog = None

        self.cluster_labels_cache: dict = {}
        self.coord_cache: dict = {}

        # (vis_mode, is_atom_view) last seen — used to trigger data loads on change.
        self._prev_view_state: tuple[str, bool] | None = None
        self._docking_file_map: dict[Path, Path] = {}  # normal_xyz -> docking_xyz
        self._checkpoint_file_map: dict[Path, Path] = {}  # normal_xyz -> checkpoint_txt
        self._structure: Structure | None = None

        self.aboutDialog = None
        self.text_file_viewer = None
        self._colour_legend_dialog = None

        self.progressBar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.hide()

        self.movie_controls_frame.hide()

        self.settings_dialog = SettingsDialog(self)
        self.openglwidget = VisualisationWidget()

        self.crystalInfoWidget = CrystalInfoWidget(self)
        self.crystalInfoWidget.setEnabled(False)
        self.crystalInfo_groupBox.layout().addWidget(self.crystalInfoWidget)

        self.crystal_info = CrystalInfo()
        self.crystalInfoChanged.connect(self.crystalInfoWidget.update)
        self.gl_vLayout.addWidget(self.openglwidget, 0, 0)  # Row 0, Column 0

        # Add point info toolbar on right side of OpenGL widget
        self.pointInfoToolbar = PointInfoToolbar(self)
        self.pointInfoToolbar.set_collapsed(True)  # Start collapsed
        self.gl_vLayout.addWidget(self.pointInfoToolbar, 0, 1)  # Row 0, Column 1
        self.gl_vLayout.setColumnStretch(0, 1)  # OpenGL widget stretches
        self.gl_vLayout.setColumnStretch(1, 0)  # Toolbar doesn't stretch

        # Connect toolbar signals
        self.pointInfoToolbar.deleteSelectedRequested.connect(self.delete_selected_points)
        self.pointInfoToolbar.clearSelectionRequested.connect(self.clear_point_selection)
        self.pointInfoToolbar.unselectCurrentRequested.connect(self.unselect_point)
        self.pointInfoToolbar.exportXYZRequested.connect(self.export_xyz)

        # Connect OpenGL widget signals to toolbar
        self.openglwidget.pointHovered.connect(self.pointInfoToolbar.update_hover_info)
        self.openglwidget.selectionChanged.connect(self.pointInfoToolbar.update_selection_info)
        # Keep the Crystal Information point count in sync with what is actually rendered
        # (e.g. after the colour-legend filter hides some points).
        self.openglwidget.renderedCountChanged.connect(self._handle_rendered_count)
        self.pointInfoToolbar.set_point_data_fn(self.openglwidget._get_point_data)

        self.xyzFilenameListWidget.currentRowChanged.connect(self.setCurrentXYZIndex)
        self.xyz_spinBox.valueChanged.connect(self.setCurrentXYZIndex)

        self.saveframe_pushButton.hide()

        self.visualizationSettings = VisualizationSettingsWidget(parent=self)
        self.visualizationSettings.setEnabled(enabled=True)
        self.visualizationSettings.settingsChanged.connect(self.handleVisualizationSettingsChange)
        self.visualizationTab.layout().addWidget(self.visualizationSettings)
        self.fps = self.visualizationSettings.fps()

        # Wire point-size slider in VisualizationSettings ↔ OpenGL widget
        ps_widget = self.visualizationSettings.widgets["Point Size"]
        self.openglwidget.pointSizeChanged.connect(
            lambda v: (
                ps_widget.slider.blockSignals(True),
                ps_widget.setValue(float(v)),
                ps_widget.slider.blockSignals(False),
            )
        )

        # Create atom mode settings dialog (non-modal, shown via Crystallography menu)
        self.atom_mode_settings_dialog = AtomModeSettingsDialog(parent=self)
        self.atom_mode_settings_dialog.settingsChanged.connect(self._handle_atom_mode_settings)
        self.openglwidget.viewStateChanged.connect(self._on_view_state_changed)

        # Create unit cell viewer dialog (Tools menu)
        self.unit_cell_viewer_dialog = UnitCellViewerDialog(parent=self)

        # Create site highlighting dialog
        self.site_highlight_dialog = SiteHighlightDialog(parent=self)
        self.site_highlight_dialog.highlightsChanged.connect(self.handle_highlights_changed)
        self.site_highlight_dialog.clearHighlights.connect(self.handle_clear_highlights)

        # Create axes settings dialog
        self.axes_settings_dialog = AxesSettingsDialog(parent=self)
        self.axes_settings_dialog.settingsChanged.connect(self.handle_axes_settings_changed)

        # Create sphere material / lighting settings dialog
        self.render_settings_dialog = RenderSettingsDialog(parent=self)
        self.render_settings_dialog.settingsChanged.connect(
            self.openglwidget.set_render_settings
        )

        # Create directions and planes dialogs
        self.directions_dialog = DirectionsDialog(parent=self)
        self.directions_dialog.directionsChanged.connect(self.handle_directions_changed)
        self.directions_dialog.directionsCleared.connect(self.handle_directions_cleared)

        self.planes_dialog = PlanesDialog(parent=self)
        self.planes_dialog.planesChanged.connect(self.handle_planes_changed)
        self.planes_dialog.planesCleared.connect(self.handle_planes_cleared)
        self.planes_dialog.computePlaneFromSelection.connect(self._on_compute_plane_from_selection)
        self.planes_dialog.addDirectionRequested.connect(self._on_add_direction_from_plane)
        self.planes_dialog.planeSelected.connect(self._on_plane_selected_for_move_slider)
        self.planes_dialog.planeNormalSliderMoved.connect(self._on_plane_normal_slider_moved)
        self.directions_dialog.addPlaneRequested.connect(self._on_add_plane_from_direction)

        # Animation system — timeline embedded below the OpenGL widget
        self._animation_timeline = AnimationTimeline()
        self._timeline_dock = KeyframeTimelineWidget(self)
        self._timeline_dock.set_timeline(self._animation_timeline)
        self.gl_vLayout.addWidget(self._timeline_dock, 1, 0, 1, 2)
        self.gl_vLayout.setRowStretch(0, 1)  # OpenGL widget takes all spare height
        self.gl_vLayout.setRowStretch(1, 0)  # Timeline takes only what it needs
        self._timeline_dock.hide()
        self._keyframe_preview_active = False
        self._render_worker = None

        # Timeline signals
        self._timeline_dock.keyframeAddRequested.connect(self._add_keyframe)
        self._timeline_dock.previewRequested.connect(self._on_preview_tick)
        self._timeline_dock.previewStopped.connect(self._on_preview_stopped)
        self._timeline_dock.renderRequested.connect(self._open_render_dialog)

        self._setup_analysis_button_grid()
        self.setup_button_connections()
        self.setup_menubar_connections()
        self.setup_log_menu_actions()
        self.shortcuts_manager = ShortcutsManager(self.menuBar)
        self._keyboard_shortcuts_dialog = None
        self.setShowPlottingButtons(False)

    def _setup_analysis_button_grid(self):
        """Replace the 3-button HBoxLayout with a 2×2 QGridLayout and add Clusters button."""
        # Create the new Clusters button
        self.cluster_analysis_pushButton = QPushButton("Clusters")
        self.cluster_analysis_pushButton.setEnabled(False)
        self.cluster_analysis_pushButton.setToolTip("Perform cluster analysis (DBSCAN/OPTICS)")
        self.cluster_analysis_pushButton.setSizePolicy(self.aspect_ratio_pushButton.sizePolicy())

        # Find the HBoxLayout inside verticalLayout_2 and replace it with a 2×2 grid
        vbox = self.verticalLayout_2
        for i in range(vbox.count()):
            item = vbox.itemAt(i)
            if item and item.layout() is self.horizontalLayout_2:
                # Remove buttons from old HBox
                while self.horizontalLayout_2.count():
                    w = self.horizontalLayout_2.takeAt(0).widget()
                    if w:
                        self.horizontalLayout_2.removeWidget(w)
                # Remove the HBox item from vbox
                vbox.removeItem(item)
                # Insert 2×2 grid at the same position
                grid = QGridLayout()
                grid.addWidget(self.aspect_ratio_pushButton, 0, 0)
                grid.addWidget(self.growth_rate_pushButton, 0, 1)
                grid.addWidget(self.site_analysis_pushButton, 1, 0)
                grid.addWidget(self.cluster_analysis_pushButton, 1, 1)
                vbox.insertLayout(i, grid)
                break

    def setup_menubar_connections(self):
        # Remove permanently-disabled leftover from the auto-generated UI
        self.menuView.removeAction(self.actionSettings)

        self.actionImport.triggered.connect(lambda: self.import_and_visualise_xyz(folder=None))
        self.actionImport_CSV_for_Plotting.triggered.connect(self.browse_plot_csv)
        self.actionImportCSVClipboard.triggered.connect(
            lambda x: self.set_plotting(QtWidgets.QApplication.clipboard().text())
        )

        self.actionImport_Summary_File.triggered.connect(
            lambda: self.read_summary(summary_file=None)
        )

        self.actionInput_Directory.triggered.connect(
            lambda: (
                open_directory(path=self.input_folder) if self.input_folder is not None else None
            )
        )
        self.actionResults_Directory.triggered.connect(
            lambda: (
                open_directory(path=self.output_folder) if self.output_folder is not None else None
            )
        )
        self.actionRender.triggered.connect(self.openglwidget.saveRenderDialog)
        self.actionExportXYZ = QAction("Export XYZ", self)
        self.actionExportXYZ.setObjectName("actionExportXYZ")
        self.actionExportXYZ.setShortcut("Ctrl+Shift+E")
        self.actionExportXYZ.setToolTip("Export the current point cloud to an XYZ file")
        self.actionExportXYZ.triggered.connect(self.export_xyz)
        self.actionExportXYZ.setEnabled(False)  # Disabled until XYZ is loaded
        self.menuFile.addAction(self.actionExportXYZ)
        self.actionExportDockingPDB = QAction("Export Docking PDB", self)
        self.actionExportDockingPDB.setObjectName("actionExportDockingPDB")
        self.actionExportDockingPDB.setShortcut("Ctrl+Shift+D")
        self.actionExportDockingPDB.setToolTip("Export docking atom-mode structure to a PDB file")
        self.actionExportDockingPDB.triggered.connect(self.export_docking_pdb)
        self.actionExportDockingPDB.setEnabled(False)  # Enabled once docking data is loaded
        self.menuFile.addAction(self.actionExportDockingPDB)
        self.actionPlottingDialog.triggered.connect(self.replotting_called)

        self.actionAboutCGAspects.triggered.connect(self.showAboutDialog)

        # Add Site Highlighting action to View menu
        self.actionSiteHighlighting = QAction("Highlight Sites", self)
        self.actionSiteHighlighting.setObjectName("actionSiteHighlighting")
        self.actionSiteHighlighting.setShortcut("Ctrl+Shift+S")
        self.actionSiteHighlighting.triggered.connect(self.show_site_highlighting_dialog)
        self.menuView.addAction(self.actionSiteHighlighting)

        # Add Projection Mode Toggle action to View menu
        self.actionToggleProjection = QAction("Switch to Perspective Projection", self)
        self.actionToggleProjection.setObjectName("actionToggleProjection")
        self.actionToggleProjection.setShortcut("Ctrl+Shift+P")
        self.actionToggleProjection.setToolTip(
            "Toggle between Orthographic and Perspective projection"
        )
        self.actionToggleProjection.triggered.connect(self.toggle_projection_mode)
        self.menuView.addAction(self.actionToggleProjection)

        # Add Axes Settings action to View menu
        self.actionAxesSettings = QAction("Axes Settings", self)
        self.actionAxesSettings.setObjectName("actionAxesSettings")
        self.actionAxesSettings.setShortcut("Ctrl+Shift+A")
        self.actionAxesSettings.setToolTip("Configure axes rendering settings")
        self.actionAxesSettings.triggered.connect(self.show_axes_settings_dialog)
        self.menuView.addAction(self.actionAxesSettings)

        # Add Sphere & Lighting Settings action to View menu
        self.actionRenderSettings = QAction("Sphere && Lighting Settings", self)
        self.actionRenderSettings.setObjectName("actionRenderSettings")
        self.actionRenderSettings.setToolTip(
            "Configure sphere material, lighting and ambient occlusion"
        )
        self.actionRenderSettings.triggered.connect(self.show_render_settings_dialog)
        self.menuView.addAction(self.actionRenderSettings)

        # Add Toggle Point Info Sidebar action to View menu
        self.actionToggleSidebar = QAction("Toggle Point Info Panel", self)
        self.actionToggleSidebar.setObjectName("actionToggleSidebar")
        self.actionToggleSidebar.setShortcut("Ctrl+B")
        self.actionToggleSidebar.setToolTip("Show/hide the point info side panel")
        self.actionToggleSidebar.triggered.connect(self._toggle_point_info_sidebar)
        self.menuView.addAction(self.actionToggleSidebar)

        # Add Show Colour Legend action to View menu
        self.actionShowLegend = QAction("Show Colour Legend", self)
        self.actionShowLegend.setObjectName("actionShowLegend")
        self.actionShowLegend.setShortcut("Ctrl+Shift+C")
        self.actionShowLegend.setToolTip("Show the current colour legend")
        self.actionShowLegend.triggered.connect(self.show_colour_legend)
        self.menuView.addAction(self.actionShowLegend)

        # Add Show Mesh Edges toggle to View menu (only enabled in Convex Hull mode)
        self.actionShowMeshEdges = QAction("Show Mesh Edges", self)
        self.actionShowMeshEdges.setObjectName("actionShowMeshEdges")
        self.actionShowMeshEdges.setEnabled(False)
        self.actionShowMeshEdges.setToolTip(
            "Toggle wireframe edges on the convex hull mesh (Convex Hull mode only)"
        )
        self.actionShowMeshEdges.triggered.connect(self.openglwidget.toggle_mesh_edges)
        self.menuView.addAction(self.actionShowMeshEdges)

        # ── Viewport shortcuts (configurable via ShortcutsManager) ────────────
        from PySide6.QtWidgets import QMenu

        self.menuView.addSeparator()

        # Align View submenu
        menuAlignView = QMenu("Align View", self)
        for shortcut, method, label in [
            ("X", self.openglwidget.align_view_x, "Align to X Axis"),
            ("Y", self.openglwidget.align_view_y, "Align to Y Axis"),
            ("Z", self.openglwidget.align_view_z, "Align to Z Axis"),
            ("A", self.openglwidget.align_view_a, "Align to a Axis"),
            ("B", self.openglwidget.align_view_b, "Align to b Axis"),
            ("C", self.openglwidget.align_view_c, "Align to c Axis"),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(method)
            menuAlignView.addAction(act)
        self.menuView.addMenu(menuAlignView)

        # Rotation Lock submenu — single-axis, mutually exclusive, toggle off by re-pressing
        menuRotLock = QMenu("Rotation Lock", self)
        self._rotation_lock_actions: dict[str, QAction] = {}
        for shortcut, axis, label in [
            ("1", "x", "Lock to X Axis"),
            ("2", "y", "Lock to Y Axis"),
            ("3", "z", "Lock to Z Axis"),
        ]:
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.setCheckable(True)
            self._rotation_lock_actions[axis] = act
            menuRotLock.addAction(act)
        # Wire after all actions exist so cross-uncheck is safe
        for axis, act in self._rotation_lock_actions.items():

            def _on_lock_toggled(checked, a=axis):
                if checked:
                    for other, other_act in self._rotation_lock_actions.items():
                        if other != a:
                            other_act.setChecked(False)
                self.openglwidget.toggle_rotation_lock(a, checked)

            act.toggled.connect(_on_lock_toggled)
        self.menuView.addMenu(menuRotLock)

        self.actionToggleCameraMode = QAction("Toggle Camera / Object Mode", self)
        self.actionToggleCameraMode.setObjectName("actionToggleCameraMode")
        self.actionToggleCameraMode.setShortcut("Ctrl+K")
        self.actionToggleCameraMode.setShortcutContext(Qt.ApplicationShortcut)
        self.actionToggleCameraMode.setToolTip(
            "Switch between Camera orbit mode and Object rotation mode (Ctrl+K)"
        )
        self.actionToggleCameraMode.triggered.connect(self.openglwidget.toggle_interaction_mode)
        self.menuView.addAction(self.actionToggleCameraMode)

        self.menuView.addSeparator()

        actReset = QAction("Reset View", self)
        actReset.setObjectName("actionResetView")
        actReset.setShortcut("R")
        actReset.triggered.connect(self.openglwidget.reset_view)
        self.menuView.addAction(actReset)

        actRecentre = QAction("Recentre View", self)
        actRecentre.setObjectName("actionRecentreView")
        actRecentre.setShortcut("F")
        actRecentre.triggered.connect(self.openglwidget.recentre_view)
        self.menuView.addAction(actRecentre)

        actStore = QAction("Store View Orientation", self)
        actStore.setObjectName("actionStoreView")
        actStore.setShortcut("Shift+S")
        actStore.triggered.connect(self.openglwidget.store_view)
        self.menuView.addAction(actStore)

        # Visualisation Mode submenu — which data source is shown (Crystal = the
        # normal .XYZ output, Docking = docking-site file, Checkpoint = grid file).
        # Docking / Checkpoint entries are enabled per-simulation when files exist.
        menuVisMode = QMenu("Visualisation Mode", self)
        self.menuVisualisationMode = menuVisMode
        self._vis_mode_group = QActionGroup(self)
        self._vis_mode_group.setExclusive(True)
        self._vis_mode_actions: dict[str, QAction] = {}
        for mode, shortcut, tooltip in [
            ("Crystal", "Shift+C", "Show the crystal (.XYZ) output"),
            ("Docking", "Shift+D", "Show docking sites (requires docking file)"),
            ("Checkpoint", "Shift+H", "Show the checkpoint grid (requires checkpoint file)"),
        ]:
            act = QAction(mode, self)
            act.setObjectName(f"actionVisMode{mode}")
            act.setShortcut(shortcut)
            act.setToolTip(tooltip)
            act.setCheckable(True)
            act.setEnabled(mode == "Crystal")
            act.triggered.connect(lambda checked=False, m=mode: self._on_vis_mode_action(m))
            self._vis_mode_group.addAction(act)
            menuVisMode.addAction(act)
            self._vis_mode_actions[mode] = act
        self._vis_mode_actions["Crystal"].setChecked(True)
        self.menuView.addMenu(menuVisMode)

        # Atom vs centroid representation for the current mode (centroid default).
        self.actionAtomView = QAction("Atom View", self)
        self.actionAtomView.setObjectName("actionToggleAtomView")
        self.actionAtomView.setShortcut("Shift+V")
        self.actionAtomView.setCheckable(True)
        self.actionAtomView.setToolTip(
            "Show individual atoms instead of centroids for the current visualisation "
            "mode (requires structure file)"
        )
        self.actionAtomView.triggered.connect(self._on_atom_view_toggled)
        self.menuView.addAction(self.actionAtomView)

        actCheckpointMiddle = QAction("Toggle Checkpoint Middle Cells", self)
        actCheckpointMiddle.setObjectName("actionToggleCheckpointMiddle")
        actCheckpointMiddle.setShortcut("Shift+M")
        actCheckpointMiddle.setToolTip(
            "Show or hide the interior (middle) cells of the checkpoint grid "
            "(edges only by default; requires checkpoint file)"
        )
        actCheckpointMiddle.triggered.connect(self.toggle_checkpoint_middle)
        self.menuView.addAction(actCheckpointMiddle)

        menuPointSize = QMenu("Point Size", self)
        actIncrease = QAction("Increase", self)
        actIncrease.setObjectName("actionIncreasePointSize")
        actIncrease.setShortcut("Ctrl+=")
        ps_widget = self.visualizationSettings.widgets["Point Size"]
        actIncrease.triggered.connect(
            lambda: ps_widget.setValue(ps_widget.value + ps_widget.step)
        )
        menuPointSize.addAction(actIncrease)
        actDecrease = QAction("Decrease", self)
        actDecrease.setObjectName("actionDecreasePointSize")
        actDecrease.setShortcut("Ctrl+-")
        actDecrease.triggered.connect(
            lambda: ps_widget.setValue(ps_widget.value - ps_widget.step)
        )
        menuPointSize.addAction(actDecrease)
        self.menuView.addMenu(menuPointSize)

        menuBondRadius = QMenu("Bond Radius", self)
        menuBondRadius.setEnabled(False)
        self.menuBondRadius = menuBondRadius
        actIncreaseBond = QAction("Increase", self)
        actIncreaseBond.setObjectName("actionIncreaseBondRadius")
        actIncreaseBond.setShortcut("Ctrl+Shift+=")
        actIncreaseBond.triggered.connect(self.openglwidget.increase_bond_radius)
        menuBondRadius.addAction(actIncreaseBond)
        actDecreaseBond = QAction("Decrease", self)
        actDecreaseBond.setObjectName("actionDecreaseBondRadius")
        actDecreaseBond.setShortcut("Ctrl+Shift+-")
        actDecreaseBond.triggered.connect(self.openglwidget.decrease_bond_radius)
        menuBondRadius.addAction(actDecreaseBond)
        self.menuView.addMenu(menuBondRadius)

        # Create Crystallography menu

        self.menuCrystallography = QMenu("Crystallography", self)
        self.menuBar.addAction(self.menuCrystallography.menuAction())

        self.actionAddDirections = QAction("Add Directions", self)
        self.actionAddDirections.setObjectName("actionAddDirections")
        self.actionAddDirections.setShortcut("Ctrl+Shift+D")
        self.actionAddDirections.setToolTip("Add crystallographic directions to the visualization")
        self.actionAddDirections.triggered.connect(self.show_directions_dialog)
        self.menuCrystallography.addAction(self.actionAddDirections)

        self.actionAddPlanes = QAction("Add Planes", self)
        self.actionAddPlanes.setObjectName("actionAddPlanes")
        self.actionAddPlanes.setShortcut("Ctrl+Shift+L")
        self.actionAddPlanes.setToolTip("Add crystallographic planes to the visualization")
        self.actionAddPlanes.triggered.connect(self.show_planes_dialog)
        self.menuCrystallography.addAction(self.actionAddPlanes)

        self.menuCrystallography.addSeparator()

        self.actionAtomModeSettings = QAction("Atom Mode Settings…", self)
        self.actionAtomModeSettings.setObjectName("actionAtomModeSettings")
        self.actionAtomModeSettings.setShortcut("Ctrl+Shift+M")
        self.actionAtomModeSettings.setToolTip(
            "Adjust per-element colours, atom sizes, and bond radius for Atom view"
        )
        self.actionAtomModeSettings.setEnabled(False)  # enabled only in Atom mode
        self.actionAtomModeSettings.triggered.connect(self.show_atom_mode_settings)
        self.menuCrystallography.addAction(self.actionAtomModeSettings)

        self.menuCrystallography.addSeparator()

        self.actionUnitCellViewer = QAction("Unit Cell Viewer", self)
        self.actionUnitCellViewer.setToolTip(
            "View the unit cell, molecule templates, and crystal net connections"
        )
        self.actionUnitCellViewer.triggered.connect(self.show_unit_cell_viewer)
        self.menuCrystallography.addAction(self.actionUnitCellViewer)

        # Tools menu
        self.menuTools = QMenu("Tools", self)
        self.menuBar.addAction(self.menuTools.menuAction())

        self.actionThreadMonitor = QAction("Active Threads", self)
        self.actionThreadMonitor.setToolTip("Show active background thread pools")
        self.actionThreadMonitor.triggered.connect(self.show_thread_monitor)
        self.menuTools.addAction(self.actionThreadMonitor)

        # Animation menu
        self.menuAnimation = QMenu("Animation", self)
        self.menuBar.addAction(self.menuAnimation.menuAction())

        self.actionToggleTimeline = QAction("Keyframe Timeline", self)
        self.actionToggleTimeline.setObjectName("actionToggleTimeline")
        self.actionToggleTimeline.setShortcut("Ctrl+T")
        self.actionToggleTimeline.setCheckable(True)
        self.actionToggleTimeline.setChecked(False)
        self.actionToggleTimeline.triggered.connect(self._toggle_timeline_dock)
        self.menuAnimation.addAction(self.actionToggleTimeline)

        self.actionAddKeyframe = QAction("Add Keyframe Here", self)
        self.actionAddKeyframe.setObjectName("actionAddKeyframe")
        self.actionAddKeyframe.setShortcut("K")
        self.actionAddKeyframe.triggered.connect(self._add_keyframe)
        self.menuAnimation.addAction(self.actionAddKeyframe)

        self.actionRenderAnimation = QAction("Render Animation…", self)
        self.actionRenderAnimation.setObjectName("actionRenderAnimation")
        self.actionRenderAnimation.triggered.connect(self._open_render_dialog)
        self.menuAnimation.addAction(self.actionRenderAnimation)

        self.menuAnimation.addSeparator()

        self.actionSaveAnimation = QAction("Save Animation…", self)
        self.actionSaveAnimation.setObjectName("actionSaveAnimation")
        self.actionSaveAnimation.triggered.connect(self._save_animation)
        self.menuAnimation.addAction(self.actionSaveAnimation)

        self.actionLoadAnimation = QAction("Load Animation…", self)
        self.actionLoadAnimation.setObjectName("actionLoadAnimation")
        self.actionLoadAnimation.triggered.connect(self._load_animation)
        self.menuAnimation.addAction(self.actionLoadAnimation)

        # Help menu
        self.menuHelp = QMenu("Help", self)
        self.menuBar.addAction(self.menuHelp.menuAction())

        self.actionKeyboardShortcuts = QAction("Keyboard Shortcuts", self)
        self.actionKeyboardShortcuts.setObjectName("actionKeyboardShortcuts")
        self.actionKeyboardShortcuts.setShortcut("Ctrl+/")
        self.actionKeyboardShortcuts.setToolTip("View and customise keyboard shortcuts")
        self.actionKeyboardShortcuts.triggered.connect(self.show_keyboard_shortcuts_dialog)
        self.menuHelp.addAction(self.actionKeyboardShortcuts)

    def setup_log_menu_actions(self):
        # Create Open Log File action
        self.actionOpenLogFile = QAction("Open Log File", self)
        self.actionOpenLogFile.setObjectName("actionOpenLogFile")
        self.actionOpenLogFile.setShortcut("Ctrl+L")
        self.actionOpenLogFile.setToolTip("Open the application log file")
        self.actionOpenLogFile.triggered.connect(self.open_log_file)

        # Create Clear Log File action
        self.actionClearLogFile = QAction("Clear Log File", self)
        self.actionClearLogFile.setObjectName("actionClearLogFile")
        self.actionClearLogFile.setToolTip("Clear the application log file")
        self.actionClearLogFile.triggered.connect(self.clear_log_file)

        # Create Set Lattice Parameters action
        self.actionSetLatticeParameters = QAction("Set Lattice Parameters", self)
        self.actionSetLatticeParameters.setObjectName("actionSetLatticeParameters")
        self.actionSetLatticeParameters.setToolTip(
            "Set lattice parameters to convert axes to fractional coordinates"
        )
        self.actionSetLatticeParameters.triggered.connect(self.show_lattice_parameters_dialog)

        # Create Toggle Axes action (starts disabled until lattice params are set)
        self.actionToggleAxes = QAction("Switch to Fractional Axes", self)
        self.actionToggleAxes.setObjectName("actionToggleAxes")
        self.actionToggleAxes.setShortcut("Shift+A")
        self.actionToggleAxes.setToolTip("Toggle between Cartesian and fractional axes")
        self.actionToggleAxes.setEnabled(False)  # Disabled until lattice params are set
        self.actionToggleAxes.triggered.connect(self.toggle_axes)

        # Track current axes state
        self.current_axes_type = "cartesian"
        self.crystallography = None

        # Add log actions to View menu
        self.menuView.addSeparator()
        self.menuView.addAction(self.actionOpenLogFile)
        self.menuView.addAction(self.actionClearLogFile)

        # Add lattice/axes actions to Crystallography menu
        self.menuCrystallography.addSeparator()
        self.menuCrystallography.addAction(self.actionSetLatticeParameters)
        self.menuCrystallography.addAction(self.actionToggleAxes)

    def show_colour_legend(self):
        if self._colour_legend_dialog is None:
            self._colour_legend_dialog = ColorLegendDialog(parent=self)
            self.openglwidget.legendChanged.connect(self._colour_legend_dialog.update_legend)
            self._colour_legend_dialog.colorOverrideRequested.connect(
                self._handle_legend_color_override
            )
            self._colour_legend_dialog.filterChanged.connect(
                self.openglwidget.set_legend_filter
            )
            info = self.openglwidget.get_legend_info()
            if info is not None:
                self._colour_legend_dialog.update_legend(info)

        if self._colour_legend_dialog.isVisible():
            self._colour_legend_dialog.raise_()
            self._colour_legend_dialog.activateWindow()
        else:
            self._colour_legend_dialog.show()

    def _handle_rendered_count(self, count: int, label: str):
        """Update the Crystal Information count to match the rendered (filtered) view."""
        self.crystal_info.pointCount = count if count else None
        self.crystal_info.countLabel = label
        self.crystalInfoChanged.emit(self.crystal_info)

    def _handle_legend_color_override(self, mode: str, key, color):
        """Route legend colour-pick / reset signals to the GL widget."""
        gl = self.openglwidget
        if mode == "reset_all":
            gl.reset_legend_colors()
        elif mode == "atom":
            # key is an element symbol string
            rgb = tuple(color) if color is not None else None
            gl.set_legend_element_color(key, rgb)
        elif mode == "docking_shell":
            # key is the shell_id integer (comes as str from signal — convert)
            shell_id = gl._shell_id(key) if isinstance(key, str) else int(key)
            if shell_id is not None:
                rgb = tuple(color) if color is not None else None
                gl.set_legend_shell_color(shell_id, rgb)

    def showAboutDialog(self):
        if self.aboutDialog is None:
            self.aboutDialog = AboutCGDialog(self)

        if self.aboutDialog.isVisible():
            self.aboutDialog.raise_()
            self.aboutDialog.activateWindow()
        else:
            self.aboutDialog.show()

    def show_keyboard_shortcuts_dialog(self):
        if self._keyboard_shortcuts_dialog is None:
            self._keyboard_shortcuts_dialog = KeyboardShortcutsDialog(
                manager=self.shortcuts_manager, parent=self
            )
        self._keyboard_shortcuts_dialog.show()
        self._keyboard_shortcuts_dialog.raise_()
        self._keyboard_shortcuts_dialog.activateWindow()

    def show_unit_cell_viewer(self):
        """Open the Unit Cell / Net Viewer dialog (Tools menu)."""
        self.unit_cell_viewer_dialog.set_crystallography(self.crystallography)
        self.unit_cell_viewer_dialog.set_structure(self._structure, self.crystallography)
        self.unit_cell_viewer_dialog.show()
        self.unit_cell_viewer_dialog.raise_()

    def show_thread_monitor(self):
        from .dialogs.thread_monitor_dialog import ThreadMonitorDialog

        pools = {
            "Main": self.threadpool,
            "AspectRatio": self.aspectratio.threadpool,
            "GrowthRate": self.growthrate.threadpool,
            "SiteAnalysis": self.siteanalysis.threadpool,
            "Clusters": self.clusteranalysis.threadpool,
        }
        worker_sources = {
            "AspectRatio": lambda: self.aspectratio.worker,
            "GrowthRate": lambda: self.growthrate.worker,
            "SiteAnalysis": lambda: self.siteanalysis.worker,
            "Clusters": lambda: self.clusteranalysis.worker,
        }
        dlg = ThreadMonitorDialog(pools, worker_sources=worker_sources, parent=self)
        dlg.exec()

    def show_lattice_parameters_dialog(self):
        """Show dialog to enter lattice parameters for fractional axes."""
        # Pass current cell if available to pre-fill the dialog
        current_cell = self.crystallography.cell if self.crystallography else None
        dialog = LatticeParametersDialog(self, cell=current_cell)

        if dialog.exec():
            cell = dialog.get_cell()
            if cell is not None:
                # Create and store Crystallography object from the cell
                self.crystallography = Crystallography(cell)

                # Enable the toggle axes action
                self.actionToggleAxes.setEnabled(True)

                # Switch to fractional axes
                self.current_axes_type = "fractional"
                self.openglwidget.set_fractional_axes(self.crystallography)
                self.openglwidget.apply_coord_scale(self.crystallography)
                self.actionToggleAxes.setText("Switch to Cartesian Axes")

                # Update crystallography dialogs
                self.directions_dialog.set_crystallography(self.crystallography)
                self.planes_dialog.set_crystallography(self.crystallography)
                self.unit_cell_viewer_dialog.set_crystallography(self.crystallography)

                self.log_message(
                    f"Axes converted to fractional coordinates using lattice parameters: "
                    f"a={cell.a:.4f}, b={cell.b:.4f}, c={cell.c:.4f}, "
                    f"α={cell.alpha:.3f}°, β={cell.beta:.3f}°, γ={cell.gamma:.3f}°",
                    "info",
                )

    def toggle_axes(self):
        """Toggle between Cartesian and fractional axes."""
        if self.crystallography is None:
            self.log_message("No lattice parameters set. Please set them first.", "warning")
            return

        if self.current_axes_type == "cartesian":
            # Switch to fractional
            self.openglwidget.set_fractional_axes(self.crystallography)
            self.current_axes_type = "fractional"
            self.actionToggleAxes.setText("Switch to Cartesian Axes")
            self.log_message("Axes switched to fractional coordinates", "info")
        else:
            # Switch to Cartesian
            self.openglwidget.set_cartesian_axes()
            self.current_axes_type = "cartesian"
            self.actionToggleAxes.setText("Switch to Fractional Axes")
            self.log_message("Axes switched to Cartesian coordinates", "info")

    def toggle_projection_mode(self):
        """Toggle between Orthographic and Perspective projection."""
        current_mode = self.openglwidget.camera.projectionMode()

        if current_mode == "Orthographic":
            # Switch to Perspective
            self.openglwidget.camera.setProjectionMode("Perspective")
            self.actionToggleProjection.setText("Switch to Orthographic Projection")
            self.log_message("Projection mode switched to Perspective", "info")
        else:
            # Switch to Orthographic
            self.openglwidget.camera.setProjectionMode("Orthographic")
            self.actionToggleProjection.setText("Switch to Perspective Projection")
            self.log_message("Projection mode switched to Orthographic", "info")

        # Update the visualization
        self.openglwidget.update()

    def setup_button_connections(self):
        self.importPlotDataPushButton.clicked.connect(self.browse_plot_csv)
        self.import_pushButton.clicked.connect(lambda: self.import_and_visualise_xyz(folder=None))
        self.batch_lineEdit.returnPressed.connect(
            lambda: self.import_and_visualise_xyz(folder=self.batch_lineEdit.text())
        )
        self.view_results_pushButton.clicked.connect(
            lambda: open_directory(path=self.output_folder)
        )

        self.aspect_ratio_pushButton.clicked.connect(self.calculate_aspect_ratio)
        self.growth_rate_pushButton.clicked.connect(self.calculate_growth_rates)
        self.site_analysis_pushButton.clicked.connect(self.calculate_site_analysis)
        self.cluster_analysis_pushButton.clicked.connect(self.calculate_clusters)

        self.plot_lineEdit.textChanged.connect(self.set_plotting)
        self.plot_lineEdit.returnPressed.connect(self.replotting_called)
        self.plot_pushButton.clicked.connect(self.replotting_called)

    def set_progressbar(self):
        self.set_message("Started Calculations...")
        self.progressBar.setValue(0)
        self.progressBar.show()

    def clear_progressbar(self):
        self.set_message("Calculations Completed!")
        if self.progressBar is not None:
            self.progressBar.hide()

    def update_progressbar(self, value):
        if self.progressBar is not None:
            self.progressBar.setValue(value)

    def update_sim_id(self, value):
        if value is None:
            return
        if self.input_folder is not None:
            self.setCurrentXYZIndex(value=value)

    def highlight_site_in_visualization(self, site_number):
        """Highlight a specific site in the 3D visualization (from plot click)."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            # Check if a crystal is loaded before trying to highlight
            if self.openglwidget._visual_data is None:
                logger.debug(
                    f"Cannot highlight site {site_number} - no crystal loaded in visualizer"
                )
                return
            # Highlight just this one site
            self.openglwidget.highlight_sites([(site_number, None)])
            logger.info(f"Highlighted site {site_number} in visualization")

    def _apply_cluster_colours(self):
        """Colour the current simulation's particles by cluster label."""
        if not self.cluster_labels_cache or self.sim_num is None:
            return
        if self.sim_num >= len(self.xyz_files):
            return
        xyz_path = str(self.xyz_files[self.sim_num])
        labels = self.cluster_labels_cache.get(xyz_path)
        if labels is None:
            logger.debug("No cluster labels for current simulation: %s", xyz_path)
            return

        import matplotlib.cm as _cm

        tab10 = _cm.get_cmap("tab10")
        unique_labels = sorted(set(int(l) for l in labels))
        groups = []
        for lbl in unique_labels:
            indices = set(int(i) for i in np.where(labels == lbl)[0])
            if lbl == -1:
                colour = [0.5, 0.5, 0.5]  # grey for noise
            else:
                rgba = tab10(lbl % 10)
                colour = list(rgba[:3])
            groups.append((indices, colour))
        self.openglwidget.clear_colour_override()
        self.openglwidget.highlight_sites(groups)
        logger.info("Applied cluster colours for %s (%d clusters)", xyz_path, len(unique_labels))

    def handle_highlights_changed(self, highlight_data):
        """Handle site highlighting changes from the dialog.

        Args:
            highlight_data: List of (site_numbers, color) tuples, with last item being ("background", color or None)
        """
        if not hasattr(self, "openglwidget") or self.openglwidget is None:
            return

        if not highlight_data:
            self.openglwidget.clear_highlighted_sites()
            return

        # Extract background color (last item)
        bg_color = None
        highlight_groups = []

        for item in highlight_data:
            if len(item) == 2:
                sites, color = item
                if sites == "background":
                    # Handle None case - keep bg_color as None to use existing coloring
                    if color is not None:
                        bg_color = [color.redF(), color.greenF(), color.blueF()]
                    else:
                        bg_color = None
                else:
                    # Convert QColor to RGB array [0.0-1.0]
                    rgb_color = [color.redF(), color.greenF(), color.blueF()]
                    highlight_groups.append((sites, rgb_color))

        # Apply highlights with background color
        self.openglwidget.highlight_sites(highlight_groups, background_color=bg_color)
        total_sites = sum(len(sites) for sites, _ in highlight_groups)
        logger.info(
            f"Applied {len(highlight_groups)} highlight group(s) with {total_sites} total sites"
        )

    def handle_clear_highlights(self):
        """Handle clearing site highlights from the visualization settings widget."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            self.openglwidget.clear_highlighted_sites()
            logger.info("Cleared all site highlights")

    def delete_selected_points(self):
        """Delete the currently selected points in the visualization."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            count = self.openglwidget.delete_selected_points()
            if count > 0:
                self.log_message(f"Deleted {count} point(s)", "info")

    def clear_point_selection(self):
        """Clear the current point selection."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            self.openglwidget.clear_selection()
            self.log_message("Selection cleared", "info")

    def unselect_point(self, index):
        """Remove a single point from the current selection."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            self.openglwidget.select_point(index, toggle=True)

    def export_xyz(self):
        """Export the current point cloud to an XYZ file."""
        if hasattr(self, "openglwidget") and self.openglwidget is not None:
            self.openglwidget.exportXYZDialog()

    def export_docking_pdb(self):
        """Export the current docking atom-mode structure to a PDB file."""
        gl = self.openglwidget
        if gl is None or gl._docking_data is None or gl._docking_data.empty:
            QMessageBox.warning(self, "Export Docking PDB", "No docking data loaded.")
            return
        if not gl._mol_cart_templates:
            QMessageBox.warning(
                self,
                "Export Docking PDB",
                "No molecular templates available — load a structure file first.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Docking PDB", "", "PDB Files (*.pdb);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".pdb"):
            path += ".pdb"
        try:
            out = write_docking_pdb(
                filepath=path,
                docking_data=gl._docking_data,
                mol_cart_templates=gl._mol_cart_templates,
                a_axis=gl._a_axis(),
                crystallography=gl._mol_crystallography,
            )
            self.log_message(f"Docking PDB exported: {out}", "info")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export Docking PDB", str(exc))

    def set_message(self, msg):
        self.log_message(message=msg, log_level="info", gui=True)

    def show_worker_error(self, error_tuple):
        _, exc, message = error_tuple
        self.clear_progressbar()
        QMessageBox.critical(self, type(exc).__name__, message)

    def show_settings(self):
        self.settings_dialog.show()
        self.settings_dialog.raise_()

    def show_site_highlighting_dialog(self):
        """Show the site highlighting dialog."""
        self.site_highlight_dialog.show()
        self.site_highlight_dialog.raise_()

    def _handle_apply_cluster_colour(self, mode: str, cmap: str):
        self._active_colour_mode = mode
        self._active_colour_cmap = cmap
        if mode == "cluster":
            self._apply_cluster_colours()
        elif mode == "coord":
            self._apply_coord_colours(cmap)
        else:
            self.openglwidget.clear_highlighted_sites()
            self.openglwidget.clear_colour_override()

    def _apply_coord_colours(self, cmap_name: str = "plasma"):
        if not self.coord_cache or self.sim_num is None:
            return
        if self.sim_num >= len(self.xyz_files):
            return
        xyz_path = str(self.xyz_files[self.sim_num])
        coord_numbers = self.coord_cache.get(xyz_path)
        if coord_numbers is None or len(coord_numbers) == 0:
            logger.warning("No coordination numbers cached for %s", xyz_path)
            return
        from matplotlib import colormaps as _cms
        lo, hi = int(coord_numbers.min()), int(coord_numbers.max())
        t = (
            np.zeros(len(coord_numbers), dtype=np.float32)
            if hi == lo
            else (coord_numbers - lo).astype(np.float32) / (hi - lo)
        )
        colours = _cms[cmap_name](t)[:, :3].astype(np.float32)
        self.openglwidget.set_colour_override(colours)
        logger.info("Coord-number colours applied for %s (range [%d, %d])", xyz_path, lo, hi)

    def _show_cluster_analysis_data(self):
        """Show a popup with coord/cluster debug data for the current file."""
        if self.sim_num is None or self.sim_num >= len(self.xyz_files):
            QMessageBox.information(self, "Analysis Data", "No simulation loaded.")
            return

        xyz_path = str(self.xyz_files[self.sim_num])
        filename = self.xyz_files[self.sim_num].name
        lines = [f"<b>File:</b> {filename}", f"<b>Full path:</b> {xyz_path}", ""]

        # Rendered point count
        rendered_n = None
        _vd = self.openglwidget._visual_data
        if _vd is not None:
            rendered_n = _vd.n_centroids
            lines.append(f"<b>Rendered points (frame 0):</b> {rendered_n:,}")
        else:
            lines.append("<b>Rendered points:</b> (no data loaded)")

        lines.append("")

        # Coordination numbers
        coord_numbers = self.coord_cache.get(xyz_path)
        if coord_numbers is None or len(coord_numbers) == 0:
            lines.append("<b>Coordination numbers:</b> ❌ not in cache")
            lines.append(
                "<i>Run analysis with 'Current file only' to populate the cache for this file.</i>"
            )
        else:
            n_cached = len(coord_numbers)
            match = rendered_n is not None and n_cached == rendered_n
            match_str = "✅ matches rendered" if match else f"⚠️ MISMATCH — cached={n_cached:,}, rendered={rendered_n:,} (colour override will be skipped!)"
            lines.append(f"<b>Coord numbers cached:</b> {n_cached:,} points — {match_str}")
            lo, hi = int(coord_numbers.min()), int(coord_numbers.max())
            mean = float(coord_numbers.mean())
            std = float(coord_numbers.std())
            lines.append(f"  min={lo}, max={hi}, mean={mean:.2f}, std={std:.2f}")

            # Value distribution
            unique, counts = np.unique(coord_numbers, return_counts=True)
            lines.append("  <b>Distribution:</b>")
            for val, cnt in zip(unique, counts):
                bar = "█" * min(40, round(40 * cnt / n_cached))
                lines.append(f"    coord={val:3d}: {cnt:6,}  {bar}  ({100*cnt/n_cached:.1f}%)")

        lines.append("")

        # Cluster labels
        labels = self.cluster_labels_cache.get(xyz_path)
        if labels is None or len(labels) == 0:
            lines.append("<b>Cluster labels:</b> ❌ not in cache")
        else:
            n_cached = len(labels)
            match = rendered_n is not None and n_cached == rendered_n
            match_str = "✅ matches rendered" if match else f"⚠️ MISMATCH — cached={n_cached:,}, rendered={rendered_n:,}"
            lines.append(f"<b>Cluster labels cached:</b> {n_cached:,} points — {match_str}")
            n_noise = int((labels == -1).sum())
            unique_cl = np.unique(labels[labels >= 0])
            lines.append(f"  Clusters: {len(unique_cl)}, noise points: {n_noise:,} ({100*n_noise/n_cached:.1f}%)")
            if len(unique_cl) > 0:
                sizes = np.array([(labels == c).sum() for c in unique_cl])
                lines.append(f"  Cluster sizes — min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")
                if len(unique_cl) <= 20:
                    lines.append("  <b>Per-cluster counts:</b>")
                    for cl_id, sz in zip(unique_cl, sizes):
                        lines.append(f"    cluster {cl_id}: {sz:,} points")

        html = "<pre style='font-family:monospace'>" + "<br>".join(lines) + "</pre>"

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Analysis Data — {filename}")
        dlg.resize(600, 500)
        v = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setHtml(html)
        v.addWidget(browser)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.accept)
        v.addWidget(btns)
        dlg.exec()

    def show_axes_settings_dialog(self):
        """Show the axes settings dialog."""
        self.axes_settings_dialog.show()
        self.axes_settings_dialog.raise_()

    def show_render_settings_dialog(self):
        """Show the sphere material / lighting settings dialog."""
        self.render_settings_dialog.show()
        self.render_settings_dialog.raise_()

    def handle_axes_settings_changed(self, settings):
        """Handle changes to axes settings."""
        if hasattr(self.openglwidget, "axes_renderer"):
            self.openglwidget.axes_renderer.update_settings(settings)
            self.openglwidget.update()

    def _toggle_point_info_sidebar(self):
        """Toggle the point info sidebar panel."""
        self.pointInfoToolbar.set_collapsed(not self.pointInfoToolbar.is_collapsed())

    def _get_point_cloud_max_extent(self):
        """Compute the max extent (half-range) of the point cloud."""
        vd = self.openglwidget._visual_data
        if vd is not None and vd.n_centroids > 0:
            extents = vd.centroids.max(axis=0) - vd.centroids.min(axis=0)
            return float(extents.max()) / 2.0
        return None

    def show_directions_dialog(self):
        """Show the directions dialog."""
        self.directions_dialog.set_crystallography(self.crystallography)
        self.directions_dialog.set_point_cloud_extent(self._get_point_cloud_max_extent())
        self.directions_dialog.show()
        self.directions_dialog.raise_()

    def show_planes_dialog(self):
        """Show the planes dialog."""
        self.planes_dialog.set_crystallography(self.crystallography)
        self.planes_dialog.set_point_cloud_extent(self._get_point_cloud_max_extent())
        self.planes_dialog.show()
        self.planes_dialog.raise_()

    def handle_directions_changed(self, directions):
        """Handle changes to crystallographic directions."""
        if hasattr(self.openglwidget, "set_directions"):
            max_extent = self._get_point_cloud_max_extent() or 1.0
            self.openglwidget.set_directions(directions, self.crystallography, max_extent)

    def handle_directions_cleared(self):
        """Handle clearing of all directions."""
        if hasattr(self.openglwidget, "set_directions"):
            self.openglwidget.set_directions([], None)

    def handle_planes_changed(self, planes):
        """Handle changes to crystallographic planes."""
        if hasattr(self.openglwidget, "set_planes"):
            self.openglwidget.set_planes(planes, self.crystallography)

    def handle_planes_cleared(self):
        """Handle clearing of all planes."""
        if hasattr(self.openglwidget, "set_planes"):
            self.openglwidget.set_planes([], None)

    def _on_add_direction_from_plane(self, direction_dict):
        """Route a plane's indices to the directions dialog as a new direction."""
        self.directions_dialog.add_external(direction_dict)
        self.directions_dialog.show()
        self.directions_dialog.raise_()

    def _on_add_plane_from_direction(self, plane_dict):
        """Route a direction's indices to the planes dialog as a new plane."""
        self.planes_dialog.add_external(plane_dict)
        self.planes_dialog.show()
        self.planes_dialog.raise_()

    def _on_compute_plane_from_selection(self):
        """Compute a best-fit plane from the currently selected points and populate the planes dialog."""
        if not hasattr(self, "openglwidget") or self.openglwidget is None:
            return
        selected_indices = self.openglwidget.get_selected_points()
        if len(selected_indices) < 3:
            QMessageBox.warning(
                self,
                "Not Enough Points",
                "Please select at least 3 points in the 3D view using Shift+Click.",
            )
            return
        xyz = self.openglwidget.xyz
        if xyz is None:
            return
        coords = xyz[list(selected_indices)]
        centroid = coords.mean(axis=0)
        _, _, Vt = np.linalg.svd(coords - centroid)
        normal = Vt[-1]  # eigenvector for smallest singular value = plane normal

        self.planes_dialog.populate_from_plane(normal, centroid, self.crystallography)

    # ------------------------------------------------------------------ #
    # Move plane along its normal (inline slider in PlanesDialog)         #
    # ------------------------------------------------------------------ #

    def _on_plane_selected_for_move_slider(self, row):
        """Configure the inline move-along-normal slider when a plane is selected."""
        planes = self.planes_dialog._planes
        if row < 0 or row >= len(planes):
            return
        plane = planes[row]
        xyz = self.openglwidget.xyz
        if xyz is None:
            return
        normal = np.array(plane.normal, dtype=np.float64)
        if plane.fractional and self.crystallography is not None:
            normal = self.crystallography.miller_to_cart_normal(normal)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-10:
            return
        normal /= n_len
        proj = xyz @ normal
        d_min, d_max = float(proj.min()), float(proj.max())
        current_d = float(np.dot(normal, np.array(plane.origin, dtype=np.float64)))
        self.planes_dialog.configure_move_slider(d_min, d_max, current_d)
        self._plane_move_state = {
            "row": row,
            "normal": normal,
            "d_min": d_min,
            "d_max": d_max,
            "steps": 2000,
        }

    def _on_plane_normal_slider_moved(self, row, val):
        """Apply inline slider position to the plane origin."""
        state = getattr(self, "_plane_move_state", None)
        if state is None or state["row"] != row:
            return
        d_min, d_max, steps = state["d_min"], state["d_max"], state["steps"]
        slider_range = d_max - d_min if d_max > d_min else 1.0
        d = d_min + (val / steps) * slider_range
        new_origin = tuple(float(v) for v in (state["normal"] * d))
        self.planes_dialog.update_plane_origin_from_dialog(row, new_origin)

    def welcome_message(self):
        self.log_message(
            "############################################", log_level="info", gui=False
        )
        self.log_message(
            "####        CrystalAspects v1.00        ####", log_level="info", gui=False
        )
        self.log_message(
            "############################################", log_level="info", gui=False
        )
        self.log_message(
            "     The CrystalGrower Data Analysis Program",
            log_level="info",
            gui=False,
        )

    def log_message(self, message: str, log_level: str, gui: bool = False):
        message = str(message)
        log_level_method = getattr(logger, log_level.lower(), logger.debug)

        if gui or log_level in ["info", "warning"]:
            # Update the status bar with the message
            self.update_statusbar(message)

        # Log the message with given level
        log_level_method(message)

    def set_output_folder(self, value):
        self.output_folder = Path(value)
        self.actionResults_Directory.setEnabled(True)
        self.view_results_pushButton.setEnabled(True)

    def import_and_visualise_xyz(self, folder=None):
        if folder is not None and folder != "":
            folder = Path(folder)
            if not folder.is_dir():
                return

        imported = self.import_xyz(folder=folder)
        if imported:
            self.set_visualiser()

    def import_xyz(self, folder=None):
        """Import XYZ file(s) by first opening the folder
        and then opening them via an OpenGL widget"""

        # Initialize or clear the list of XYZ files
        self.xyz_files = []

        # Read the .XYZ files from the selected folder
        if folder is None:
            folder = QFileDialog.getExistingDirectory(
                None, "Select Folder that contains the Crystal Outputs (.XYZ)"
            )
        if folder == "":
            self.log_message("No folder selected", "debug")
            return False
        if self.input_folder == folder:
            self.log_message("Same path as current location!", "info")
            return False

        self.log_message("Reading Images...", "info")
        result = locate_xyz_files(folder)

        if result is None:
            # No XYZ files — try checkpoint files as a point-cloud fallback
            checkpoint_fallback = locate_checkpoint_files(folder)
            if not checkpoint_fallback:
                self.log_message(
                    "No XYZ or checkpoint files found in the selected folder.", "warning"
                )
                return False
            self.log_message(
                f"No XYZ files found. Falling back to {len(checkpoint_fallback)} "
                "checkpoint file(s) as point clouds.",
                "warning",
            )
            xyz_files = checkpoint_fallback
            docking_xyz_files = []
        else:
            xyz_files, docking_xyz_files = result

        self.xyz_files = xyz_files

        # Both crystal and docking files share a common base stem before their
        # trailing _SUFFIX (e.g. _CGAspects, _sim, _docking, etc.).
        # Strip the last _segment from each stem to match them up.
        def _sim_stem(p):
            parts = p.stem.split("_")
            return "_".join(parts[:-1]) if len(parts) > 1 else p.stem

        docking_by_stem = {_sim_stem(p): p for p in docking_xyz_files}
        self._docking_file_map = {
            xyz_path: docking_by_stem[_sim_stem(xyz_path)]
            for xyz_path in xyz_files
            if _sim_stem(xyz_path) in docking_by_stem
        }
        if self._docking_file_map:
            self.log_message(f"Found {len(self._docking_file_map)} docking file(s)", "info")

        checkpoint_files = natsorted(
            f for f in Path(folder).rglob("*_checkpoint.txt") if f.stat().st_size > 0
        )
        checkpoint_by_stem = {_sim_stem(p): p for p in checkpoint_files}
        self._checkpoint_file_map = {
            xyz_path: checkpoint_by_stem[_sim_stem(xyz_path)]
            for xyz_path in xyz_files
            if _sim_stem(xyz_path) in checkpoint_by_stem
        }
        if self._checkpoint_file_map:
            self.log_message(f"Found {len(self._checkpoint_file_map)} checkpoint file(s)", "info")
        self.input_folder = folder
        self.output_folder = None
        self.actionResults_Directory.setEnabled(False)
        self.view_results_pushButton.setEnabled(False)
        self.batch_lineEdit.setText(str(self.input_folder))
        self.actionInput_Directory.setEnabled(True)
        self.log_message(f"Input path set to: {self.input_folder}", "info")
        self.log_message(f"Initial XYZ list: {xyz_files}", "debug")

        if folder:
            self.xyz_files = natsorted(xyz_files)

        return True

    def set_visualiser(self):
        self.actionImport_Summary_File.setEnabled(False)
        self.summ_df = None
        if self.simulation_variables_widget is not None:
            self.simulation_variables_widget.deleteLater()
            self.simulation_variables_widget = None

        if self.xyz_files is None:
            self.log_message("No XYZ files to visualise", "warning")
            return
        n_xyz = len(self.xyz_files)
        if n_xyz == 0:
            self.log_message(f"{n_xyz} XYZ files found to set to self!", "warning")
        if n_xyz > 0:
            self.set_batch_type()
            self.init_opengl()

            # Checkpoint-only folders: load the point cloud asynchronously so
            # the UI stays responsive during the (potentially slow) parse.
            if Path(str(self.xyz_files[0])).stem.endswith("_checkpoint"):
                if self._structure is None or self.crystallography is None:
                    self.log_message(
                        "Load a structure file to visualise checkpoint data.",
                        "warning",
                    )
                    self.openglwidget.showNoDataOverlay()
                else:
                    # No real xyz crystal — render the checkpoint grid directly so
                    # site-analysis colour modes (coordination, energy …) are available.
                    self.sim_num = 0
                    self._update_vis_mode_availability()
                    self.openglwidget.set_visualisation_mode("Checkpoint")
                self.aspect_ratio_pushButton.setEnabled(True)
                self.variablesTabWidget.setCurrentIndex(0)
                self.actionImport_Summary_File.setEnabled(True)
                return

            crystal_found = False
            try:
                for i in range(n_xyz):
                    self.crystal = self.get_crystal(i)
                    if self.crystal is not None and not self.crystal.empty:
                        self.setCurrentXYZIndex(i)
                        self.init_crystal()
                        crystal_found = True
                        break
            except Exception as e:
                self.log_message(f"Error initialising crystal visualisation: {e}", "error")

            if not crystal_found:
                self.openglwidget.showNoDataOverlay()

            self.aspect_ratio_pushButton.setEnabled(True)
            self.variablesTabWidget.setCurrentIndex(0)
            self.actionImport_Summary_File.setEnabled(True)

    def init_opengl(self):
        tot_sims = len(self.xyz_files)
        self.openglwidget.pass_XYZ_list([str(path) for path in self.xyz_files])
        self.sim_num = 0
        self.openglwidget.sim_num = -1  # Force fresh load for new folder
        self.openglwidget.get_XYZ_from_list(0)
        self.xyzFilenameListWidget.clear()
        self.xyzFilenameListWidget.addItems([x.name for x in self.xyz_files])

        self.xyz_spinBox.setMinimum(0)
        self.xyz_spinBox.setMaximum(tot_sims - 1)

        self.xyzFilenameListWidget.setEnabled(True)
        self.xyz_id_label.setEnabled(True)
        self.xyz_spinBox.setEnabled(True)

        self.log_message(f"{len(self.xyz_files)} XYZ files set to visualiser!", "info")

    def init_crystal(self):
        self.movie_controls_frame.hide()
        logger.debug("Initializing crystal!")
        self.openglwidget.pass_XYZ(self.crystal.get_raw_frame_coords(0))

        if len(self.crystal) > 1:
            self.playingState = False
            self.frame_timer.stop()
            self.movie_controls_frame.show()
            self.frame_list = list(range(1, len(self.crystal) + 1))
            self.frame = 0
            logger.debug("Frames: %s", self.frame_list)

            num_frames = len(self.frame_list)

            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(num_frames - 1)

            self.frame_spinBox.setMinimum(0)
            self.frame_spinBox.setMaximum(num_frames - 1)
            self.frameMaxLabel.setText(f"{num_frames - 1}")

        try:
            self.openglwidget.initGeometry()
            self.actionRender.setEnabled(True)
            self.actionExportXYZ.setEnabled(True)
        except AttributeError as e:
            logger.warning("Initialising XYZ: No Crystal Data Found! %s", e)

    def get_crystal(self, index):
        folder = self.input_folder
        if 0 <= index < len(self.xyz_files):
            file_name = self.xyz_files[index]
            full_file_path = os.path.join(folder, file_name)

            self.set_progressbar()

            def prog(val, tot):
                self.update_progressbar(100.0 * val / tot)

            if Path(str(full_file_path)).stem.endswith("_checkpoint"):
                if self._structure is None or self.crystallography is None:
                    self.log_message(
                        "Checkpoint point cloud requires a structure file — "
                        "load one via the Structure menu.",
                        "warning",
                    )
                    self.crystal = CrystalCloud(filepath=Path(str(full_file_path)))
                else:
                    self.crystal = CrystalCloud.from_checkpoint(
                        full_file_path,
                        self._structure.n_tiles,
                        self.crystallography,
                    )
            else:
                self.crystal = CrystalCloud.from_file(full_file_path, progress_callback=prog)

            self.clear_progressbar()

            return self.crystal

    def update_XYZ_info(self):
        vd = self.openglwidget._visual_data
        if vd is None or vd.n_centroids == 0:
            self.crystal_info.aspectRatio1 = None
            self.crystal_info.aspectRatio2 = None
            self.crystal_info.shapeClass = "N/A"
            self.crystal_info.surfaceAreaVolumeRatio = None
            self.crystal_info.surfaceArea = None
            self.crystal_info.volume = None
            self.crystal_info.pointCount = None
            self.crystalInfoChanged.emit(self.crystal_info)
            return

        if self.openglwidget.is_atom_view and vd.templates:
            self.crystal_info.pointCount = vd.n_atoms or vd.n_centroids
            self.crystal_info.countLabel = "Atoms"
        else:
            self.crystal_info.pointCount = vd.n_centroids
            self.crystal_info.countLabel = "Points"

        worker_xyz = WorkerXYZ(vd.centroids)
        worker_xyz.signals.result.connect(self.insert_info)
        worker_xyz.signals.message.connect(self.update_statusbar)
        self.threadpool.start(worker_xyz)

    def update_movie(self, frame):
        if frame != self.frame:
            self.update_frame(frame)
            # block to prevent double updates
            with QSignalBlocker(self.frame_slider):
                self.frame_slider.setValue(frame)
            with QSignalBlocker(self.xyz_spinBox):
                self.frame_spinBox.setValue(frame)

    def next_frame(self):
        num_frames = len(self.frame_list)

        if self.frame < num_frames:
            self.update_frame(self.frame)
            self.frame_slider.setValue(self.frame)
            self.frame_spinBox.setValue(self.frame)
            self.frame += 1

            if self.playingLoop and self.frame >= num_frames:
                self.frame = 0
        else:
            self.frame = 0
            self.frame_timer.stop()
            self.frame_slider.setValue(self.frame)
            self.frame_spinBox.setValue(self.frame)

    def play_movie(self):
        if not self.frame_list:
            return

        # Stop any active keyframe preview first
        if self._keyframe_preview_active:
            self._timeline_dock.stop_preview()

        if self.playingState:
            # pause playing
            self.playPauseButton.setIcon(self.playIcon)
            self.frame_timer.stop()
            self.playingState = False
        else:
            # play movie
            self.playPauseButton.setIcon(self.pauseIcon)
            # make sure we don't play from after the end
            if self.frame >= len(self.frame_list):
                self.frame = 0

            self.frame_timer.start(1000 // self.fps)
            self.playingState = True

    def close_application(self):
        self.log_message("Closing Application", "info")
        self.close()

    def browse(self):
        try:
            # Attempt to get the directory from the file dialog
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select Folder",
                "./",
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
            )

            # Check if the folder selection was canceled or empty and handle appropriately
            if folder:
                self.batch_lineEdit.clear()
                self.batch_lineEdit.setText(str(folder))
            else:
                # Handle the case where no folder was selected
                self.log_message(
                    "Folder selection was canceled or no folder was selected.",
                    "warning",
                )

        # Note: Bare Exception
        except Exception as e:
            self.log_message(f"An error occurred: {e}", "error")

    def set_batch_type(self):
        folder = self.input_folder
        # Initially disable buttons that depend on the data
        self.growth_rate_pushButton.setEnabled(False)
        self.site_analysis_pushButton.setEnabled(False)

        if not Path(folder).is_dir():
            QMessageBox.warning(None, "Directory Error", f"{folder} is not a valid directory.")
            return

        self.input_folder = folder
        information = find_info(folder)

        # Auto-load structure file for fractional axes and molecular data if available
        if information.structure_file:
            self._structure = Structure.from_file(information.structure_file)

            if self._structure.cell is not None:
                cell = self._structure.cell
                self.crystallography = Crystallography(cell)
                self.actionToggleAxes.setEnabled(True)
                self.current_axes_type = "fractional"
                self.openglwidget.set_fractional_axes(self.crystallography)
                self.actionToggleAxes.setText("Switch to Cartesian Axes")
                # Update crystallography dialogs
                self.directions_dialog.set_crystallography(self.crystallography)
                self.planes_dialog.set_crystallography(self.crystallography)
                self.unit_cell_viewer_dialog.set_crystallography(self.crystallography)

                self.log_message(
                    f"Auto-loaded lattice parameters: a={cell.a:.2f} b={cell.b:.2f} c={cell.c:.2f}",
                    "info",
                )

            if self._structure.templates and self.crystallography is not None:
                self.openglwidget.set_molecular_data(
                    self._structure.templates, self.crystallography
                )
                self.unit_cell_viewer_dialog.set_structure(self._structure, self.crystallography)
                self.log_message(
                    f"Loaded {len(self._structure.templates)} molecule template(s) — "
                    "press Shift+V to switch to Atom view",
                    "info",
                )

        # Enable buttons and set data based on available information
        if information.size_files and information.directions:
            self.growth_rate_pushButton.setEnabled(True)
            self.growthrate.set_folder(folder=folder)
            self.growthrate.set_information(information=information)
            self.growthrate.set_xyz_files(xyz_files=self.xyz_files)
        else:
            if not information.size_files:
                logger.warning("No size files were found in the directory!")

        # Set Aspect Ratio information
        self.aspectratio.set_folder(folder=folder)
        self.aspectratio.set_information(information=information)
        self.aspectratio.set_xyz_files(xyz_files=self.xyz_files)

        # Set Cluster Analysis information
        if self.xyz_files:
            self.cluster_analysis_pushButton.setEnabled(True)
            self.clusteranalysis.set_folder(folder=folder)
            self.clusteranalysis.set_information(information=information)
            self.clusteranalysis.set_xyz_files(xyz_files=self.xyz_files)

        # Enable and set Site Analysis information if relevant files are found
        if information.crystallisation_files or information.population_files:
            self.site_analysis_pushButton.setEnabled(True)
            self.siteanalysis.set_folder(folder=folder)
            self.siteanalysis.set_information(information=information)
            self.siteanalysis.set_xyz_files(xyz_files=self.xyz_files)
            self.siteanalysis.set_site_files(
                crystallisation_files=information.crystallisation_files,
                population_files=information.population_files,
                count_files=information.count_files,
            )
            logger.info(
                f"Found {len(information.crystallisation_files)} crystallisation files and "
                f"{len(information.population_files)} population files"
            )
        else:
            logger.info("No crystallisation events or population files found for site analysis")

        if not information.directions:
            QMessageBox.warning(
                None,
                "Data Incomplete",
                "No crystallographic direction data found in the simulation parameters output.\n"
                "Please make sure this data is available if Crystal Directional Analysis (CDA) is required.",
            )

        if information.summary_file:
            self.read_summary(summary_file=information.summary_file)
        else:
            QMessageBox.warning(None, "Data Incomplete", "Summary file not found.")

        self.input_folder = Path(folder)

    def calculate_aspect_ratio(self):
        self.aspectratio.calculate_aspect_ratio()

    def calculate_growth_rates(self):
        self.growthrate.calculate_growth_rates()

    def calculate_site_analysis(self):
        self.siteanalysis.calculate_site_analysis()

    def calculate_clusters(self):
        if self.sim_num is not None and 0 <= self.sim_num < len(self.xyz_files):
            self.clusteranalysis.set_current_file(self.xyz_files[self.sim_num])
        else:
            self.clusteranalysis.set_current_file(None)
        # Site-analysis metadata (coordination/energy) + structure context feed the
        # radial profile; coordination/energy are looked up per point by site number.
        self.clusteranalysis.set_site_metadata(self._resolve_site_metadata())
        n_tiles = self._structure.n_tiles if self._structure is not None else None
        self.clusteranalysis.set_checkpoint_context(self.crystallography, n_tiles)
        self.clusteranalysis.calculate_clusters()

    def setShowPlottingButtons(self, state=True):
        self.actionPlottingDialog.setEnabled(state)
        self.importPlotDataPushButton.setVisible(not state)
        self.plot_lineEdit.setVisible(state)
        self.plot_pushButton.setVisible(state)

    def browse_plot_csv(self):
        try:
            # Attempt to get the directory from the file dialog
            plotting_csv = QFileDialog.getOpenFileName(
                self,
                "Select CSV File",
                "./",
                "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)",
            )[0]

            # Check if the folder selection was canceled or empty and handle appropriately
            if plotting_csv:
                plotting_csv = Path(plotting_csv)
                if plotting_csv.is_file():
                    self.plot_lineEdit.setText(str(plotting_csv))
            else:
                # Handle the case where no folder was selected
                self.log_message("File selection was canceled or no file was selected.", "debug")

        # Note: Bare Exception
        except Exception as e:
            self.log_message(f"An error occurred: {e}", "error")

    def set_plotting(self, value):
        with QSignalBlocker(self.plot_lineEdit):
            self.plot_lineEdit.setText(value)

        valid_file = Path(value).is_file()
        self.setShowPlottingButtons(valid_file)
        if valid_file:
            self.plotting_csv = Path(value)
            self.plot_pushButton.setEnabled(True)
            self.log_message(f"Plotting CSV set to {self.plotting_csv}", "info")
        else:
            self.plot_pushButton.setEnabled(False)
            self.plotting_csv = None
            self.log_message("Plotting CSV set to None", "debug")

    def set_results(self, value):
        logger.debug(f"set_results called with value: {value}")

        # Always sync cluster caches first (needed before auto-apply colour below)
        if self.clusteranalysis.labels_cache:
            self.cluster_labels_cache = dict(self.clusteranalysis.labels_cache)
            self.coord_cache = dict(self.clusteranalysis.coord_cache)
            logger.info("Cluster labels cache updated (%d files)", len(self.cluster_labels_cache))

        # Single-file cluster analysis: caches updated, auto-apply colour, no plot
        if value.csv is None:
            opts = self.clusteranalysis.options
            if opts and opts.colour_mode != "none":
                self._handle_apply_cluster_colour(opts.colour_mode, opts.colour_cmap)
            return

        self.plot_lineEdit.setText(str(value.csv))
        self.log_message(f"Accepting incoming result to GUI {value}", "debug")
        if value.selected:
            self.selected_directions = value.selected
            self.log_message(f"Selected Directions set to: {self.selected_directions}", "debug")
        if value.folder:
            self.output_folder = value.folder
            self.log_message(f"Output folder updated: [{self.output_folder}]", "debug")

        # Site-analysis output → refresh checkpoint colour maps (coordination, energy …)
        if value.csv and str(value.csv).endswith("site_analysis_data.json"):
            self._refresh_site_metadata()

        logger.debug("About to call replotting_called()")
        self.replotting_called()
        logger.debug("Returned from replotting_called()")

    def replotting_called(self):
        if self.plotting_csv:
            self.log_message(f"Plotting file: {self.plotting_csv}", "info")

            if self.plotting_dialog is None:
                self.plotting_dialog = PlottingDialog(
                    csv=self.plotting_csv,
                    signals=self.worker_signals,
                    parent=self,
                    summary_df=self.summ_df,
                )
                self.plotting_dialog.trigger_plot()
            else:
                self.plotting_dialog.setCSV(self.plotting_csv)
                self.plotting_dialog.trigger_plot()

            self.plotting_dialog.show()

    # Read Summary
    def read_summary(self, summary_file=None):
        if not self.xyz_files:
            self.log_message(
                "XYZ files needs to be loaded first to use summary file information",
                "warning",
            )
            return

        self.log_message("Reading Summary file...", "info")
        if not summary_file:
            summary_file = QFileDialog.getOpenFileName(None, "Read Summary File")
            summary_file = Path(summary_file[0])

        if not summary_file:
            self.log_message("Summary file not set!", "warning")
            return

        # Select summary file and read in as a Dataframe
        self.log_message(f"Summary File Found at: {summary_file}", "debug")
        self.summ_df = pd.read_csv(summary_file, encoding="utf-8", encoding_errors="replace")

        # Propagate the summary file path to all analysis objects so workers pick it up
        for analysis_obj in (
            self.aspectratio,
            self.growthrate,
            self.clusteranalysis,
            self.siteanalysis,
        ):
            if analysis_obj.information is not None:
                analysis_obj.information = analysis_obj.information._replace(
                    summary_file=summary_file
                )
        if list(self.summ_df.columns)[-1].startswith("Unnamed"):
            self.summ_df = self.summ_df.iloc[:, 1:-1]
        else:
            self.summ_df = self.summ_df.iloc[:, 1:]
        self.log_message(f"Summary data succesfully read! [SHAPE {self.summ_df.shape}]", "info")

        column_names = list(self.summ_df)
        self.log_message(f"Summary Column Name: {column_names}", "debug")
        # Create dictionary to store the change in variables (tile/interaction energies)
        var_dict = defaultdict(list)

        # Records the variable values from summary file
        for column in column_names:
            for index, row in self.summ_df.iterrows():
                if row[str(column)] not in var_dict[column]:
                    var_dict[column].append(row[str(column)])

        layout = self.simulationVariablesWidget.layout()

        widget = SimulationVariablesWidget(var_dict, parent=self)

        if self.simulation_variables_widget is not None:
            layout.replaceWidget(self.simulation_variables_widget, widget)
            self.simulation_variables_widget.deleteLater()
            self.simulation_variables_widget = widget

        else:
            self.simulation_variables_widget = widget
            layout.addWidget(self.simulation_variables_widget)

        self.simulation_variables_widget.variableCombinationChanged.connect(self.summary_change)

        self.statusBar().showMessage("Complete: Summary file read in!")

    def summary_change(self):
        values = self.simulation_variables_widget.currentValues()
        self.log_message(f"Looking for: {values}", log_level="debug")

        mask = (self.summ_df == values).all(axis=1)
        filtered_df = self.summ_df[mask]
        if filtered_df.empty:
            return

        if len(filtered_df.index) > 1:
            self.log_message("Set of values have selected more than one row/simulation", "warning")
            return

        # self.update_variables(values=values)

        selected_index = filtered_df.index[0]
        self.setCurrentXYZIndex(value=selected_index)

    def update_variables(self, values):
        if self.simulation_variables_widget is not None:
            self.simulation_variables_widget.setValues(values)
        else:
            logger.error("Simulation variables widget is None")

    def setCurrentXYZIndex(self, value):
        self.sim_num = value
        self.openglwidget.get_XYZ_from_list(value=value)
        if self._active_colour_mode == "cluster" and self.cluster_labels_cache:
            self._apply_cluster_colours()
        elif self._active_colour_mode == "coord" and self.coord_cache:
            self._apply_coord_colours(self._active_colour_cmap)
        if self.openglwidget.crystal is not None:
            self.crystal = self.openglwidget.crystal
        self.movie_controls_frame.hide()

        if self.crystal is not None and self.crystal.empty:
            self.update_XYZ_info(None)
            return

        if self.crystal is not None and len(self.crystal) > 1:
            self.movie_controls_frame.show()
            self.frame_list = list(range(1, len(self.crystal) + 1))
            num_frames = len(self.frame_list)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(num_frames - 1)
            self.frame_spinBox.setMinimum(0)
            self.frame_spinBox.setMaximum(num_frames - 1)
            self.frameMaxLabel.setText(f"{num_frames - 1}")

        # block to prevent double updates
        with QSignalBlocker(self.xyzFilenameListWidget):
            self.xyzFilenameListWidget.setCurrentRow(value)
        with QSignalBlocker(self.xyz_spinBox):
            self.xyz_spinBox.setValue(value)

        self.update_XYZ_info()

        self._update_vis_mode_availability()
        self._update_docking_for_current_xyz()
        self._update_checkpoint_for_current_xyz()
        self.updateVisualizationSettings()

        if self.summ_df is not None:
            var_values = self.summ_df.iloc[value, :].values
            self.update_variables(values=var_values)

    def _update_docking_for_current_xyz(self):
        """Reload docking data for the new XYZ if Docking mode is active."""
        if self.openglwidget.vis_mode == "Docking":
            self._load_docking_for_current_xyz()

    def _update_checkpoint_for_current_xyz(self):
        """Reload checkpoint data for the new XYZ if Checkpoint mode is active."""
        if self.openglwidget.vis_mode == "Checkpoint":
            self._load_checkpoint_for_current_xyz()

    def _load_docking_for_current_xyz(self):
        """Load docking data for the current XYZ into the GL widget (for Docking style)."""
        if self.sim_num is None or not self.xyz_files:
            return
        current_path = self.xyz_files[self.sim_num]
        docking_path = self._docking_file_map.get(current_path)
        if docking_path is None:
            self.log_message("No docking file for this simulation", "warning")
            return
        try:
            docking_data = DockingData.from_file(docking_path)
            self.openglwidget.set_docking_data(docking_data)
            self.actionExportDockingPDB.setEnabled(True)
            self.log_message(f"Docking site loaded: {docking_path.name}", "info")
        except (OSError, ValueError) as exc:
            self.log_message(f"Failed to load docking file: {exc}", "error")

    def _load_checkpoint_for_current_xyz(self):
        """Dispatch a background worker to load the checkpoint for the current simulation."""
        if self.sim_num is None or not self.xyz_files:
            return
        current_path = self.xyz_files[self.sim_num]
        checkpoint_path = self._checkpoint_file_map.get(current_path)
        if checkpoint_path is None:
            self.log_message("No checkpoint file for this simulation", "warning")
            return
        if self._structure is None:
            self.log_message(
                "No structure file found — cannot read n_tiles for checkpoint", "warning"
            )
            return
        worker = WorkerCheckpoint(
            checkpoint_path,
            self._structure.n_tiles,
            self.crystallography,
            build_visual_data=True,
        )
        worker.signals.result.connect(self._on_checkpoint_loaded_for_view)
        worker.signals.progress.connect(self.update_progressbar)
        worker.signals.error.connect(
            lambda err: self.log_message(f"Failed to load checkpoint: {err[1]}", "error")
        )
        self.set_progressbar()
        # Override the generic "Started Calculations…" banner: this is a file parse,
        # not an analysis run, so the status should say so.
        self.set_message(f"Loading checkpoint {checkpoint_path.name}…")
        worker.signals.finished.connect(self.clear_progressbar)
        self.threadpool.start(worker)

    def _on_checkpoint_loaded_for_view(self, checkpoint):
        """Receive a loaded Checkpoint and push it to the GL widget (grid/view mode)."""
        self.openglwidget.set_checkpoint(checkpoint)
        self._refresh_site_metadata()
        self.log_message(
            f"Checkpoint loaded: {checkpoint.filepath.name} ({checkpoint.n_filled:,} filled cells)",
            "info",
        )

    def toggle_checkpoint_middle(self):
        """Show/hide the checkpoint's interior (middle) cells.

        Switching middle cells *on* the first time expands the whole grid, which is
        far heavier than the edges-only load, so it runs on a worker thread and the
        view updates when it finishes. Switching off (or on again, once cached) is an
        instant swap handled directly by the GL widget.
        """
        widget = self.openglwidget
        if not widget.has_checkpoint:
            self.log_message("No checkpoint loaded to toggle middle cells", "warning")
            return
        show_middle = not widget._checkpoint.show_middle
        needs_build = widget.request_show_middle(show_middle)
        if not needs_build:
            self.log_message(
                f"Checkpoint middle cells {'shown' if show_middle else 'hidden'}", "info"
            )
            return
        # First time showing middle cells: expand the full grid off the GUI thread.
        self.log_message("Expanding checkpoint interior cells…", "info")
        cryst = widget._checkpoint.crystallography or self.crystallography
        worker = WorkerCheckpointExpand(widget._checkpoint, cryst, include_middle=True)
        worker.signals.result.connect(widget.apply_checkpoint_full_vd)
        worker.signals.progress.connect(self.update_progressbar)
        worker.signals.result.connect(
            lambda _: self.log_message("Checkpoint middle cells shown", "info")
        )
        worker.signals.error.connect(
            lambda err: self.log_message(f"Failed to expand checkpoint: {err[1]}", "error")
        )
        self.set_progressbar()
        # Override the generic "Started Calculations…" banner (see checkpoint load).
        self.set_message("Expanding checkpoint interior cells…")
        worker.signals.finished.connect(self.clear_progressbar)
        self.threadpool.start(worker)

    def _find_site_analysis_json(self) -> Path | None:
        """Locate a saved site_analysis_data.json (output folder, then input tree)."""
        candidates: list[Path] = []
        if self.output_folder:
            candidates.append(Path(self.output_folder) / "site_analysis_data.json")
        of = getattr(self.siteanalysis, "output_folder", None)
        if of:
            candidates.append(Path(of) / "site_analysis_data.json")
        if self.input_folder:
            candidates.extend(Path(self.input_folder).rglob("site_analysis_data.json"))
        for candidate in candidates:
            if candidate and candidate.is_file():
                return candidate
        return None

    def _resolve_site_metadata(self) -> dict[str, dict[int, float]]:
        """Build site→value colour maps: in-session parsed data first, else saved JSON."""
        from ..analysis.site_parser import build_site_metadata_maps, load_site_metadata_maps

        parsed = getattr(self.siteanalysis, "parsed_data", None)
        if parsed:
            try:
                return build_site_metadata_maps(parsed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to build site metadata from in-session data: %s", exc)

        json_path = self._find_site_analysis_json()
        if json_path is not None:
            try:
                return load_site_metadata_maps(json_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load site metadata from %s: %s", json_path, exc)

        return {}

    def _refresh_site_metadata(self):
        """Resolve site-analysis colour maps, push them to the GL widget, and
        repopulate the Color By combo so new modes appear for the active style."""
        maps = self._resolve_site_metadata()
        self.openglwidget.set_site_metadata(maps)
        if any(maps.values()):
            available = [k for k, v in maps.items() if v]
            logger.info("Site colour metadata available: %s", ", ".join(available))

        # Refresh Color By options for the current view state (the new fields only
        # show in Checkpoint mode, but this is harmless otherwise).
        self._update_color_by_options_for_current_state()

    def _dispatch_checkpoint_as_crystal(self, index):
        """Start a worker to load checkpoint file *index* as a point-cloud CrystalCloud."""
        checkpoint_path = self.xyz_files[index]
        worker = WorkerCheckpoint(
            checkpoint_path,
            self._structure.n_tiles,
            self.crystallography,
        )
        worker.signals.result.connect(self._on_checkpoint_loaded_as_crystal)
        worker.signals.progress.connect(self.update_progressbar)
        worker.signals.error.connect(
            lambda err: self.log_message(f"Checkpoint point-cloud error: {err[1]}", "error")
        )
        self.set_progressbar()
        worker.signals.finished.connect(self.clear_progressbar)
        self.threadpool.start(worker)

    def _on_checkpoint_loaded_as_crystal(self, checkpoint):
        """Convert a loaded Checkpoint to a CrystalCloud and initialise the visualiser."""
        import numpy as np
        from ..fileio.xyz_file import CrystalCloud, Frames, Frame

        coords = checkpoint.to_cartesian().astype(np.float32)
        if coords.size:
            coords -= coords.mean(axis=0)

        # VisualData.from_xyz expects (N, 7) with mol_type in col 0 and xyz in cols 3:6.
        n = len(coords)
        raw = np.zeros((n, 7), dtype=np.float32)
        raw[:, 0] = 1           # mol_type
        raw[:, 3:6] = coords

        frames = Frames([Frame(raw=raw, comment="checkpoint")])
        xyz = CrystalCloud.normalise_verts(coords.copy()) if coords.size else coords
        self.crystal = CrystalCloud(filepath=checkpoint.filepath, frames=frames, xyz=xyz)

        if not self.crystal.empty:
            self.sim_num = 0
            self.init_crystal()
        else:
            self.openglwidget.showNoDataOverlay()

    def updateVisualizationSettings(self):
        pass

    def handleVisualizationSettingsChange(self):
        settings = self.visualizationSettings.settings()
        self.openglwidget.updateSettings(**settings)

        fps = self.visualizationSettings.fps()
        if self.fps != fps:
            self.frame_timer.start(1000 // self.fps)
            self.fps = fps

    def _update_color_by_options_for_current_state(self):
        """Update the Color By combo options to match the active mode / atom view."""
        opts, default = self.openglwidget._color_by_options_for_state()
        options = list(opts)
        current_val = self.visualizationSettings.settings().get("Color By", "")
        effective_default = current_val if current_val in options else default
        self.visualizationSettings.setColorByOptions(options, effective_default)

    def show_atom_mode_settings(self):
        """Open the Atom Mode Settings dialog, populated with current elements."""
        elements = self.openglwidget.get_visible_elements()
        if not elements:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Atom Mode Settings",
                "No molecular data loaded yet.\n"
                "Load a simulation folder with a structure file and switch to Atom view first.",
            )
            return
        color_ov, radius_ov, bond_r = self.openglwidget.get_atom_overrides()
        bond_summary = self.openglwidget.get_bond_summary()
        self.atom_mode_settings_dialog.populate(elements, color_ov, radius_ov, bond_r, bond_summary)
        self.atom_mode_settings_dialog.show()
        self.atom_mode_settings_dialog.raise_()
        self.atom_mode_settings_dialog.activateWindow()

    def _handle_atom_mode_settings(self, color_overrides, radius_overrides, bond_radius):
        """Relay atom-mode override changes from the dialog to the OpenGL widget."""
        self.openglwidget.set_atom_overrides(color_overrides, radius_overrides, bond_radius)

    def _on_vis_mode_action(self, mode: str):
        """Handle a Visualisation Mode menu selection (Crystal / Docking / Checkpoint)."""
        changed = self.openglwidget.set_visualisation_mode(mode)
        # Re-sync checks in case the switch was refused or was a no-op.
        self._sync_view_menu_checks()
        if changed:
            self.openglwidget.recentre_view()

    def _on_atom_view_toggled(self, checked: bool):
        """Handle the Atom View menu toggle for the current visualisation mode."""
        self.openglwidget.set_atom_view(checked)
        # Re-sync in case the switch was refused (e.g. no structure file loaded).
        self._sync_view_menu_checks()

    def _sync_view_menu_checks(self):
        mode = self.openglwidget.vis_mode
        for m, act in self._vis_mode_actions.items():
            act.setChecked(m == mode)
        self.actionAtomView.setChecked(self.openglwidget.is_atom_view)

    def _update_vis_mode_availability(self):
        """Enable Docking/Checkpoint modes per current simulation; fall back if needed."""
        current_path = (
            self.xyz_files[self.sim_num]
            if self.xyz_files and self.sim_num is not None and self.sim_num < len(self.xyz_files)
            else None
        )
        checkpoint_only = current_path is not None and Path(str(current_path)).stem.endswith(
            "_checkpoint"
        )
        available = {
            "Crystal": current_path is not None and not checkpoint_only,
            "Docking": current_path in self._docking_file_map,
            "Checkpoint": checkpoint_only or current_path in self._checkpoint_file_map,
        }
        for mode, act in self._vis_mode_actions.items():
            act.setEnabled(available[mode])

        # If the active mode is no longer available, fall back to the first enabled one.
        if not available.get(self.openglwidget.vis_mode, False):
            for mode in ("Crystal", "Checkpoint", "Docking"):
                if available[mode]:
                    self.openglwidget.set_visualisation_mode(mode)
                    break
            self._sync_view_menu_checks()

    def _on_view_state_changed(self):
        """Keep menus, combos, and info panels in sync when the viewport's
        visualisation mode, atom/centroid toggle, or render option changes."""
        gl = self.openglwidget
        mode, atom_view = gl.vis_mode, gl.is_atom_view
        self._sync_view_menu_checks()

        # Sync the render-style combo (blocked so this doesn't loop back)
        style_widget = self.visualizationSettings.widgets.get("Style")
        if style_widget is not None:
            style_widget.comboBox.blockSignals(True)
            style_widget.setValue(gl.render_option)
            style_widget.comboBox.blockSignals(False)

        self.actionAtomModeSettings.setEnabled(atom_view)
        self.menuBondRadius.setEnabled(atom_view)

        is_convex_hull = not atom_view and gl.render_option == "Convex Hull"
        self.actionShowMeshEdges.setEnabled(is_convex_hull)
        if not is_convex_hull and gl.show_mesh_edges:
            gl.show_mesh_edges = False
            gl.update()

        self._update_color_by_options_for_current_state()

        # Load mode data only when the (mode, atom view) pair actually changes —
        # render-option changes must not re-trigger file loads.
        state = (mode, atom_view)
        if state != self._prev_view_state:
            self._prev_view_state = state
            if mode == "Docking":
                self._load_docking_for_current_xyz()
            elif mode == "Checkpoint":
                self._load_checkpoint_for_current_xyz()

        # Update crystal info count when switching between atom and centroid views
        vd = gl._visual_data
        if vd is not None and vd.n_centroids > 0:
            if atom_view and vd.templates:
                self.crystal_info.pointCount = vd.n_atoms or vd.n_centroids
                self.crystal_info.countLabel = "Atoms"
            else:
                self.crystal_info.pointCount = vd.n_centroids
                self.crystal_info.countLabel = "Points"
            self.crystalInfoChanged.emit(self.crystal_info)

        # If switching into an atom view and the dialog is already open, refresh it
        if atom_view and self.atom_mode_settings_dialog.isVisible():
            elements = self.openglwidget.get_visible_elements()
            color_ov, radius_ov, bond_r = self.openglwidget.get_atom_overrides()
            bond_summary = self.openglwidget.get_bond_summary()
            self.atom_mode_settings_dialog.populate(
                elements, color_ov, radius_ov, bond_r, bond_summary
            )

    # Utility function to clear a layout of all its widgets
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def insert_info(self, result):
        self.log_message("Inserting data to GUI!", log_level="debug", gui=True)
        if result is None:
            self.crystal_info.aspectRatio1 = None
            self.crystal_info.aspectRatio2 = None
            self.crystal_info.shapeClass = "N/A"
            self.crystal_info.surfaceAreaVolumeRatio = None
            self.crystal_info.surfaceArea = None
            self.crystal_info.volume = None
        else:
            self.crystal_info.aspectRatio1 = result.aspect1
            self.crystal_info.aspectRatio2 = result.aspect2
            self.crystal_info.shapeClass = result.shape
            self.crystal_info.surfaceAreaVolumeRatio = result.surface_area_to_volume_ratio
            self.crystal_info.surfaceArea = result.surface_area
            self.crystal_info.volume = result.volume

        self.crystalInfoChanged.emit(self.crystal_info)

    def update_frame(self, frame):
        self.frame = frame
        self.openglwidget.pass_XYZ(self.crystal.get_raw_frame_coords(frame))
        self.update_XYZ_info()
        try:
            self.openglwidget.initGeometry()
        except AttributeError:
            logger.warning("Updating XYZ: No Crystal Data Found!")
        except IndexError:
            logger.warning("Frame %s has no point data, skipping.", frame)
            return False
        return True

    # ------------------------------------------------------------------
    # Animation / keyframe methods
    # ------------------------------------------------------------------

    def _toggle_timeline_dock(self, checked: bool):
        self._timeline_dock.setVisible(checked)

    def _add_keyframe(self):
        """Capture the current viewport state as a keyframe."""
        snap = self.openglwidget.snapshot()

        # Capture style / colour settings
        settings = self.visualizationSettings.settings()
        single_color_q = settings.get("Single Color")
        if single_color_q is not None and hasattr(single_color_q, "getRgbF"):
            sc = single_color_q.getRgbF()  # (r, g, b, a) floats 0-1
        else:
            sc = (0.5, 0.5, 0.5, 1.0)

        snap = dataclasses.replace(
            snap,
            style=settings.get("Style", self.openglwidget.render_option),
            color_by=settings.get("Color By", self.openglwidget.color_by),
            colormap=settings.get("Color Map", self.openglwidget.colormap),
            single_color=sc,
            planes=list(self.openglwidget._raw_planes),
            directions=list(self.openglwidget._raw_directions),
        )

        tl = self._animation_timeline
        # Place at end of timeline + 1 second
        t = tl.keyframes[-1].time + 1.0 if tl.keyframes else 0.0
        data_frame = self.frame if self.frame_list else None
        kf = Keyframe(time=t, camera=snap, data_frame=data_frame)
        tl.add_keyframe(kf)
        self._timeline_dock.refresh()
        # Show the timeline panel if hidden
        if not self._timeline_dock.isVisible():
            self._timeline_dock.show()
            self.actionToggleTimeline.setChecked(True)

    def _apply_snapshot_view_state(self, snapshot) -> None:
        """Apply style, colour, planes, and directions from a snapshot to the viewport."""
        from PySide6.QtGui import QColor as _QColor

        sc = snapshot.single_color
        q_single = _QColor.fromRgbF(
            max(0.0, min(1.0, sc[0])),
            max(0.0, min(1.0, sc[1])),
            max(0.0, min(1.0, sc[2])),
            max(0.0, min(1.0, sc[3])),
        )
        # Apply style/colour settings first; camera restore below will overwrite any
        # side-effect camera rescaling from a style switch.
        self.openglwidget.updateSettings(
            **{
                "Style": snapshot.style,
                "Color By": snapshot.color_by,
                "Color Map": snapshot.colormap,
                "Single Color": q_single,
            }
        )
        # Restore camera (overwrites any scale side-effect from updateSettings)
        self.openglwidget.apply_camera_snapshot(snapshot)
        # Apply planes / directions
        self.openglwidget.set_planes(snapshot.planes, self.crystallography)
        self.openglwidget.set_directions(
            snapshot.directions,
            self.crystallography,
            self.openglwidget._directions_max_extent,
        )

    def _on_preview_tick(self, t: float):
        """Apply interpolated viewport state for preview at time t."""
        tl = self._animation_timeline
        if len(tl.keyframes) < 2:
            return
        self._keyframe_preview_active = True
        # Stop the data-frame playback timer to avoid conflict
        if self.playingState:
            self.frame_timer.stop()
            self.playingState = False
            self.playPauseButton.setIcon(self.playIcon)
        try:
            snapshot, data_frame = tl.get_state_at_time(t)
        except ValueError:
            return
        self._apply_snapshot_view_state(snapshot)
        if data_frame is not None and self.frame_list:
            frame_idx = max(0, min(data_frame, len(self.frame_list) - 1))
            if frame_idx != self.frame:
                self.update_frame(frame_idx)

    def _on_preview_stopped(self):
        self._keyframe_preview_active = False

    def _open_render_dialog(self):
        """Open the render animation settings dialog."""
        from .animation.render_dialog import RenderAnimationDialog

        if len(self._animation_timeline.keyframes) < 2:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "No Animation",
                "Add at least 2 keyframes before rendering.\n"
                "Use Animation → Add Keyframe Here (K) to capture the current view.",
            )
            return
        dlg = RenderAnimationDialog(
            timeline=self._animation_timeline,
            viewport_width=self.openglwidget.width(),
            viewport_height=self.openglwidget.height(),
            parent=self,
        )
        dlg.renderStarted.connect(self._on_render_started)
        dlg.exec()

    def _on_render_started(self, worker):
        """Bridge: connect render worker's frameRequested to the main-thread render slot."""
        self._timeline_dock.stop_preview()
        self._render_worker = worker
        worker.frameRequested.connect(self._on_render_frame_requested)

    def _on_render_frame_requested(self, frame_idx: int, snapshot, data_frame):
        """Main-thread slot: render one frame and return the QImage to the worker."""
        self._apply_snapshot_view_state(snapshot)
        if data_frame is not None and self.frame_list:
            frame_idx_clamped = max(0, min(data_frame, len(self.frame_list) - 1))
            self.update_frame(frame_idx_clamped)

        worker = self._render_worker
        backend = getattr(worker, "raytrace_backend", None) if worker else None
        if backend:
            img = self._raytrace_animation_frame(frame_idx, worker)
        else:
            img = self.openglwidget.render_animation_frame()
        if worker is not None:
            worker.frame_ready(img)

    def _raytrace_animation_frame(self, frame_idx: int, worker):
        """Render the current view for one animation frame via POV-Ray/Tachyon.

        Runs on the main thread (needs the live widget state); reuses a per-render
        scratch directory so scene files don't accumulate.
        """
        from PySide6.QtGui import QImage

        from .visualisation import raytrace_export as rt

        w, h = worker.resolution
        scene = rt.build_scene_from_widget(
            self.openglwidget, w, h, photoreal=worker.photoreal)
        if not len(scene.spheres) and not len(scene.cylinders):
            return QImage()

        scratch = getattr(worker, "_rt_scratch", None)
        if scratch is None:
            import tempfile
            scratch = Path(tempfile.mkdtemp(prefix="cga_raytrace_"))
            worker._rt_scratch = scratch

        suffix = rt.SCENE_SUFFIX[worker.raytrace_backend]
        scene_path = scratch / f"frame_{frame_idx:05d}{suffix}"
        image_path = scratch / f"frame_{frame_idx:05d}.png"
        try:
            rt.render_scene(scene, worker.raytrace_backend, scene_path, image_path)
        except Exception:
            logger.exception("Ray-traced frame %d failed", frame_idx)
            return QImage()
        return QImage(str(image_path))

    def _save_animation(self):
        """Save the current animation timeline to a JSON file."""
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation", "animation.json", "Animation (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._animation_timeline.to_dict(), f, indent=2)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_animation(self):
        """Load an animation timeline from a JSON file."""
        import json

        path, _ = QFileDialog.getOpenFileName(self, "Load Animation", "", "Animation (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._animation_timeline = AnimationTimeline.from_dict(data)
            self._timeline_dock.set_timeline(self._animation_timeline)
            self._timeline_dock.show()
            self.actionToggleTimeline.setChecked(True)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Load Error", str(exc))

    def close_opengl_widget(self):
        if self.current_viewer:
            # Remove the OpenGL widget from its parent layout
            self.viewer_container_layout.removeWidget(self.current_viewer)

            # Delete the widget from memory
            self.current_viewer.deleteLater()
            self.current_viewer = None  # Reset the active viewer

    def update_statusbar(self, status):
        self.statusBar().showMessage(status, self.status_timeout)

    def thread_finished(self):
        self.log_message("THREAD COMPLETED!", "info")

    def open_log_file(self):
        """Open the log file in the text file viewer widget."""
        log_file = get_log_file_path()
        if log_file.exists():
            try:
                # Create or show the text file viewer
                if self.text_file_viewer is None:
                    self.text_file_viewer = TextFileViewer(
                        file_path=log_file, parent=self, auto_refresh=False, refresh_interval=2000
                    )
                    # Set window size
                    self.text_file_viewer.resize(800, 600)
                else:
                    # Update the file path if viewer already exists
                    self.text_file_viewer.set_file(log_file)

                # Show and raise the viewer window
                self.text_file_viewer.show()
                self.text_file_viewer.raise_()
                self.text_file_viewer.activateWindow()

                self.log_message(f"Opening log file: {log_file}", "info")
            except Exception as e:
                self.log_message(f"Failed to open log file: {e}", "error")
                QMessageBox.warning(
                    self,
                    "Error Opening Log File",
                    f"Could not open the log file:\n{log_file}\n\nError: {e}",
                )
        else:
            self.log_message("Log file does not exist yet", "warning")
            QMessageBox.information(
                self,
                "Log File Not Found",
                f"The log file has not been created yet:\n{log_file}\n\n"
                "It will be created automatically when the application logs messages.",
            )

    def clear_log_file(self):
        """Clear the contents of the log file after user confirmation."""
        log_file = get_log_file_path()

        # Confirm with user before clearing
        reply = QMessageBox.question(
            self,
            "Clear Log File",
            f"Are you sure you want to clear the log file?\n\n{log_file}\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            try:
                if log_file.exists():
                    # Open in write mode to truncate the file
                    with open(log_file, "w") as f:
                        pass
                    self.log_message("Log file cleared successfully", "info")
                    QMessageBox.information(
                        self, "Log File Cleared", "The log file has been cleared successfully."
                    )
                else:
                    self.log_message("Log file does not exist", "warning")
                    QMessageBox.information(
                        self, "Log File Not Found", "The log file does not exist yet."
                    )
            except Exception as e:
                self.log_message(f"Failed to clear log file: {e}", "error")
                QMessageBox.warning(
                    self,
                    "Error Clearing Log File",
                    f"Could not clear the log file:\n{log_file}\n\nError: {e}",
                )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                pass
        else:
            super().keyPressEvent(event)


def set_default_opengl_version(major, minor):
    from PySide6.QtGui import QSurfaceFormat

    format = QSurfaceFormat()
    format.setVersion(major, minor)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setOption(QSurfaceFormat.DebugContext)
    # format.setDepthBufferSize(24)
    # format.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(format)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-native-menubar",
        default=False,
        action="store_true",
        help="Don't use native menubar",
    )
    parser.add_argument(
        "--no-dpi-scaling",
        default=False,
        action="store_true",
        help="Disable High DPI scaling",
    )
    args = parser.parse_args()

    set_default_opengl_version(3, 3)
    # Setting taskbar icon permissions - windows
    appid = "CrystalGrower.CGAspects.0.8.0"
    import ctypes

    if hasattr(ctypes, "windll"):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

    # ############# Runs the application ############## #
    # sys.argv += ['--style', 'Material.Light']
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, args.no_dpi_scaling)

    QtWidgets.QApplication.setAttribute(Qt.AA_DontUseNativeMenuBar, args.no_native_menubar)

    QtWidgets.QApplication.setApplicationName("CGAspects")
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    mainwindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

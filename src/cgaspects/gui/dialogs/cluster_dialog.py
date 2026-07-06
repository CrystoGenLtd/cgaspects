from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ...utils.data_structures import cluster_options_tuple

_COLORMAPS = ["plasma", "viridis", "inferno", "coolwarm", "RdYlGn"]


class ClusterAnalysisDialog(QDialog):
    runRequested = Signal(object)           # emits cluster_options_tuple
    applyColourRequested = Signal(str, str) # (mode, cmap_name)
    showDataRequested = Signal()            # request debug data display

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Cluster Analysis")

        self._all_paths: list[Path] = []
        self._current_path: Path | None = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        warning_label = QLabel(
            "<b>⚠️ Beta Feature</b><br>"
            "Cluster analysis is intended for simulations with internal sites (not just surface "
            "particles). Clustering can be slow and may cause the application to become "
            "unresponsive when there are many points."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet(
            "QLabel { background-color: #fff3cd; color: #856404; "
            "border: 1px solid #ffc107; border-radius: 4px; padding: 6px; }"
        )
        layout.addWidget(warning_label)

        layout.addSpacing(6)

        # --- Analysis status (updated after each run) ---
        self._status_label = QLabel("No analysis run yet.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet(
            "QLabel { color: #555; font-style: italic; padding: 2px 0; }"
        )
        layout.addWidget(self._status_label)

        layout.addSpacing(6)

        # --- File scope ---
        scope_group = QGroupBox("Files to analyse")
        scope_layout = QVBoxLayout()
        self._scope_buttons = QButtonGroup(self)
        self._scope_current = QRadioButton("Current file only (default)")
        self._scope_all = QRadioButton("All files in folder")
        self._scope_current.setChecked(True)
        self._scope_buttons.addButton(self._scope_current, 0)
        self._scope_buttons.addButton(self._scope_all, 1)
        scope_layout.addWidget(self._scope_current)
        scope_layout.addWidget(self._scope_all)
        self._scope_file_label = QLabel()
        self._scope_file_label.setWordWrap(True)
        self._scope_file_label.setStyleSheet("QLabel { color: #444; font-size: 11px; }")
        scope_layout.addWidget(self._scope_file_label)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        layout.addSpacing(4)

        self.ratios_only_checkbox = QCheckBox("Ratios only (skip clustering)")
        self.ratios_only_checkbox.setToolTip(
            "Only compute particle-type count ratios without running a clustering algorithm."
        )
        self.ratios_only_checkbox.setContentsMargins(0, 4, 0, 4)
        layout.addWidget(self.ratios_only_checkbox)
        layout.addSpacing(6)

        # --- KDTree parameters ---
        params_group = QGroupBox("KDTree Connectivity Parameters")
        params_layout = QFormLayout()

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.1, 100.0)
        self.eps_spin.setSingleStep(0.5)
        self.eps_spin.setValue(3.0)
        self.eps_spin.setDecimals(1)
        self.eps_spin.setToolTip(
            "Maximum distance between two points to be considered neighbours. "
            "When 'Standardise coordinates' is enabled this is in standard-deviation units."
        )
        params_layout.addRow("Neighbour radius (ε):", self.eps_spin)

        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(1, 200)
        self.min_samples_spin.setValue(5)
        self.min_samples_spin.setToolTip(
            "Minimum number of points for a connected component to be kept as a cluster. "
            "Smaller components are labelled as noise (−1)."
        )
        params_layout.addRow("Minimum cluster size:", self.min_samples_spin)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(-1, 9999)
        self.frame_spin.setValue(-1)
        self.frame_spin.setToolTip(
            "Frame index to analyse per XYZ file. -1 = last frame, 0 = first frame.\n"
            "⚠ If the analysed frame differs from the displayed frame, point counts may\n"
            "mismatch and the colour override will be silently skipped."
        )
        params_layout.addRow("Frame index (−1 = last):", self.frame_spin)

        self.scale_checkbox = QCheckBox("Standardise coordinates (StandardScaler)")
        self.scale_checkbox.setToolTip(
            "Normalise coordinates to zero mean and unit variance before clustering. "
            "Changes the units of ε."
        )
        params_layout.addRow("", self.scale_checkbox)

        self.downsample_spin = QDoubleSpinBox()
        self.downsample_spin.setRange(0.01, 1.0)
        self.downsample_spin.setSingleStep(0.05)
        self.downsample_spin.setValue(1.0)
        self.downsample_spin.setDecimals(2)
        self.downsample_spin.setToolTip(
            "Fraction of particles to keep before clustering (1.0 = no downsampling). "
            "Coordination numbers are always computed on the full particle set."
        )
        params_layout.addRow("Downsample fraction:", self.downsample_spin)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        self._params_group = params_group

        # --- Radial profile (distance from origin, using site-analysis metadata) ---
        radial_group = QGroupBox("Radial Profile")
        radial_group.setCheckable(True)
        radial_group.setChecked(False)
        radial_group.setToolTip(
            "Write radial_analysis.csv: point density and the proportion of points "
            "with each coordination number / energy level versus distance from the "
            "origin. Coordination and energy are read from the site-analysis metadata."
        )
        radial_layout = QFormLayout()

        self.radial_bins_spin = QSpinBox()
        self.radial_bins_spin.setRange(2, 500)
        self.radial_bins_spin.setValue(30)
        self.radial_bins_spin.setToolTip("Number of radial shells (bins) from the origin.")
        radial_layout.addRow("Radial bins:", self.radial_bins_spin)

        self.radial_source_combo = QComboBox()
        self.radial_source_combo.addItems(["XYZ point cloud", "Checkpoint grid"])
        self.radial_source_combo.setToolTip(
            "Source of points and site numbers. Checkpoint mode expands the grid "
            "and needs a structure file loaded (skips KDTree clustering)."
        )
        radial_layout.addRow("Source:", self.radial_source_combo)

        self.radial_middle_checkbox = QCheckBox("Include interior (middle) cells")
        self.radial_middle_checkbox.setToolTip(
            "Checkpoint source only: expand the full grid (edges + interior) rather "
            "than just the surface strip-edge cells. Slower and much larger."
        )
        radial_layout.addRow("", self.radial_middle_checkbox)

        radial_group.setLayout(radial_layout)
        layout.addWidget(radial_group)
        self._radial_group = radial_group

        run_row = QHBoxLayout()
        run_row.addStretch()
        self._run_btn = QPushButton("Run Analysis")
        run_row.addWidget(self._run_btn)
        layout.addLayout(run_row)

        layout.addSpacing(6)

        # --- Colour after analysis ---
        vis_group = QGroupBox("Colour after analysis")
        vis_layout = QFormLayout()

        self._colour_buttons = QButtonGroup(self)
        self._colour_none = QRadioButton("None")
        self._colour_cluster = QRadioButton("Cluster membership")
        self._colour_coord = QRadioButton("Coordination number")
        self._colour_none.setChecked(True)
        self._colour_buttons.addButton(self._colour_none, 0)
        self._colour_buttons.addButton(self._colour_cluster, 1)
        self._colour_buttons.addButton(self._colour_coord, 2)
        vis_layout.addRow("", self._colour_none)
        vis_layout.addRow("", self._colour_cluster)
        vis_layout.addRow("", self._colour_coord)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setToolTip("Colormap for coordination-number colouring.")
        self._cmap_combo.setEnabled(False)
        vis_layout.addRow("Colormap:", self._cmap_combo)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)
        self._vis_group = vis_group

        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply Colour")
        self._show_data_btn = QPushButton("Show Analysis Data")
        self._show_data_btn.setToolTip(
            "Show coordination numbers and cluster labels for the current file — "
            "useful for diagnosing why colour override is not appearing."
        )
        self._close_btn = QPushButton("Close")
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._show_data_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._close_btn)
        layout.addLayout(btn_row)

        # Connections
        self.ratios_only_checkbox.toggled.connect(self._on_ratios_only_toggled)
        self._colour_coord.toggled.connect(self._cmap_combo.setEnabled)
        self._run_btn.clicked.connect(self._on_run)
        self._apply_btn.clicked.connect(self._on_apply)
        self._show_data_btn.clicked.connect(self.showDataRequested)
        self._close_btn.clicked.connect(self.hide)
        self._scope_current.toggled.connect(self._update_scope_label)

    # ------------------------------------------------------------------
    # Public API for mainwindow to populate context

    def set_file_context(self, current_path: Path | None, all_paths: list[Path]):
        """Call before show() so the dialog knows available files."""
        self._current_path = current_path
        self._all_paths = list(all_paths)
        self._update_scope_label()

    def update_analysis_status(self, labels_cache: dict, coord_cache: dict, all_paths: list[Path]):
        """Refresh the status label to show how many files have cached results."""
        n_total = len(all_paths)
        n_labels = sum(1 for p in all_paths if str(p) in labels_cache)
        n_coord = sum(1 for p in all_paths if str(p) in coord_cache)

        if n_labels == 0 and n_coord == 0:
            self._status_label.setText("No analysis run yet.")
            self._status_label.setStyleSheet("QLabel { color: #555; font-style: italic; padding: 2px 0; }")
        else:
            parts = []
            if n_labels:
                parts.append(f"cluster labels: {n_labels}/{n_total}")
            if n_coord:
                parts.append(f"coord numbers: {n_coord}/{n_total}")
            text = "Cached — " + ", ".join(parts) + " files."
            self._status_label.setText(text)
            self._status_label.setStyleSheet(
                "QLabel { color: #1a6b1a; font-style: normal; padding: 2px 0; }"
            )

    # ------------------------------------------------------------------

    def _update_scope_label(self):
        if self._scope_current.isChecked() and self._current_path is not None:
            self._scope_file_label.setText(f"→ {self._current_path.name}")
        elif self._all_paths:
            self._scope_file_label.setText(f"→ {len(self._all_paths)} files")
        else:
            self._scope_file_label.setText("")

    def _on_ratios_only_toggled(self, checked: bool):
        self._params_group.setEnabled(not checked)
        self.frame_spin.setEnabled(True)
        self._vis_group.setEnabled(not checked)

    def _on_run(self):
        self.runRequested.emit(self._build_options())

    def _on_apply(self):
        if self._colour_cluster.isChecked():
            mode = "cluster"
        elif self._colour_coord.isChecked():
            mode = "coord"
        else:
            mode = "none"
        self.applyColourRequested.emit(mode, self._cmap_combo.currentText())

    def _build_options(self) -> cluster_options_tuple:
        if self._colour_cluster.isChecked():
            colour_mode = "cluster"
        elif self._colour_coord.isChecked():
            colour_mode = "coord"
        else:
            colour_mode = "none"

        if self._scope_current.isChecked() and self._current_path is not None:
            files_to_analyse = [self._current_path]
        else:
            files_to_analyse = None  # means "all"

        radial_source = (
            "checkpoint"
            if self.radial_source_combo.currentText().startswith("Checkpoint")
            else "xyz"
        )

        return cluster_options_tuple(
            eps=self.eps_spin.value(),
            min_samples=self.min_samples_spin.value(),
            frame_index=self.frame_spin.value(),
            scale=self.scale_checkbox.isChecked(),
            downsample=self.downsample_spin.value(),
            ratios_only=self.ratios_only_checkbox.isChecked(),
            colour_mode=colour_mode,
            colour_cmap=self._cmap_combo.currentText(),
            files_to_analyse=files_to_analyse,
            radial=self._radial_group.isChecked(),
            radial_bins=self.radial_bins_spin.value(),
            radial_source=radial_source,
            radial_include_middle=self.radial_middle_checkbox.isChecked(),
        )

"""Appearance & export settings for the Unit Cell Viewer.

Non-modal dialog holding everything that only affects how the current scene
is *drawn* (connection style, sizes) plus material/lighting settings and
image/ray-trace export — kept out of the main viewer dialog so that dialog
stays focused on data (supercell, net, selection, and per-item visibility).
"""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
)

from .render_settings_dialog import RenderSettingsDialog

_BOND_RADIUS = 0.15


class UnitCellAppearanceDialog(QDialog):
    """Connection style, sizing, lighting and export controls for a UnitCellViewerWidget."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unit Cell Appearance & Export")
        self.setModal(False)
        self.setMinimumWidth(360)

        self._viewer = viewer
        self._render_settings_dialog: RenderSettingsDialog | None = None

        outer = QVBoxLayout(self)

        # --------------------------------------------------------- connections
        conn_group = QGroupBox("Connections")
        conn_form = QFormLayout(conn_group)

        self.conn_style_combo = QComboBox()
        self.conn_style_combo.addItems(["Lines", "Tubes (scaled by energy)"])
        self.conn_style_combo.setEnabled(False)
        self.conn_style_combo.currentIndexChanged.connect(
            lambda idx: viewer.set_conn_tubes(idx == 1)
        )
        conn_form.addRow("Style:", self.conn_style_combo)

        outer.addWidget(conn_group)

        # --------------------------------------------------------------- sizes
        size_group = QGroupBox("Sizes")
        size_form = QFormLayout(size_group)

        self.atom_radius_sb = QDoubleSpinBox()
        self.atom_radius_sb.setRange(0.1, 3.0)
        self.atom_radius_sb.setSingleStep(0.05)
        self.atom_radius_sb.setValue(1.0)
        self.atom_radius_sb.setDecimals(2)
        self.atom_radius_sb.setToolTip("Scale factor applied to all atom VdW radii")
        self.atom_radius_sb.valueChanged.connect(viewer.set_atom_radius_scale)
        size_form.addRow("Atom Radius Scale:", self.atom_radius_sb)

        self.bond_radius_sb = QDoubleSpinBox()
        self.bond_radius_sb.setRange(0.01, 1.0)
        self.bond_radius_sb.setSingleStep(0.01)
        self.bond_radius_sb.setValue(_BOND_RADIUS)
        self.bond_radius_sb.setDecimals(2)
        self.bond_radius_sb.setToolTip("Bond cylinder radius in Ångströms")
        self.bond_radius_sb.valueChanged.connect(viewer.set_bond_radius)
        size_form.addRow("Bond Radius (Å):", self.bond_radius_sb)

        self.conn_radius_sb = QDoubleSpinBox()
        self.conn_radius_sb.setRange(0.1, 5.0)
        self.conn_radius_sb.setSingleStep(0.1)
        self.conn_radius_sb.setValue(1.0)
        self.conn_radius_sb.setDecimals(2)
        self.conn_radius_sb.setToolTip(
            "Scale factor applied to connection lines/tubes (tube radii keep "
            "their relative energy scaling)"
        )
        self.conn_radius_sb.valueChanged.connect(viewer.set_conn_radius_scale)
        size_form.addRow("Connection Radius Scale:", self.conn_radius_sb)

        outer.addWidget(size_group)

        # ---------------------------------------------------- lighting button
        self.light_btn = QPushButton("Sphere && Lighting Settings…")
        self.light_btn.clicked.connect(self._open_render_settings)
        outer.addWidget(self.light_btn)

        # ------------------------------------------------------------- export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self.export_image_btn = QPushButton("Export Image (PNG)…")
        self.export_image_btn.clicked.connect(viewer.export_image_dialog)
        export_layout.addWidget(self.export_image_btn)

        self.export_raytrace_btn = QPushButton("Export Ray-Traced Image (POV-Ray/Tachyon)…")
        self.export_raytrace_btn.clicked.connect(viewer.export_raytrace_dialog)
        export_layout.addWidget(self.export_raytrace_btn)

        outer.addWidget(export_group)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        outer.addWidget(close_btn)
        outer.addStretch()

    # ------------------------------------------------------------------

    def _open_render_settings(self):
        if self._render_settings_dialog is None:
            self._render_settings_dialog = RenderSettingsDialog(parent=self)
            self._render_settings_dialog.settingsChanged.connect(
                self._viewer.set_render_settings
            )
        self._render_settings_dialog.show()
        self._render_settings_dialog.raise_()

    def set_available(self, has_connections: bool):
        """Enable/disable controls based on what data is currently loaded."""
        self.conn_style_combo.setEnabled(has_connections)

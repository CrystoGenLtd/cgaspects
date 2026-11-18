"""Appearance settings for the Topology Viewer.

Non-modal dialog holding everything that only affects how the current scene
is *drawn* (connection style, sizes, sphere material and lighting) — kept out
of the main viewer dialog so that dialog stays focused on data (supercell, net,
selection, and per-item visibility).
"""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)

from .render_settings_dialog import RenderSettingsPanel

_BOND_RADIUS = 0.15


class TopologyAppearanceDialog(QDialog):
    """Connection style, sizing, material and lighting controls for a TopologyViewerWidget."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topology Viewer Appearance")
        self.setModal(False)

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

        # ------------------------------------------------- material & lighting
        self.render_settings_panel = RenderSettingsPanel()
        self.render_settings_panel.settingsChanged.connect(viewer.set_render_settings)

        # ------------------------------------------------------------- buttons
        # Not auto-default, so Enter in a spin box cannot reset or close the dialog.
        reset_btn = QPushButton("Reset Sphere && Lighting")
        reset_btn.setAutoDefault(False)
        reset_btn.clicked.connect(self.render_settings_panel.reset_defaults)

        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)

        button_row = QHBoxLayout()
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        button_row.addWidget(close_btn)

        # -------------------------------------------------------------- layout
        scene_column = QVBoxLayout()
        scene_column.addWidget(conn_group)
        scene_column.addWidget(size_group)
        scene_column.addStretch()

        columns = QHBoxLayout()
        columns.addLayout(scene_column)
        columns.addWidget(self.render_settings_panel)

        outer = QVBoxLayout(self)
        outer.addLayout(columns)
        outer.addLayout(button_row)

    # ------------------------------------------------------------------

    def set_available(self, has_connections: bool):
        """Enable/disable controls based on what data is currently loaded."""
        self.conn_style_combo.setEnabled(has_connections)

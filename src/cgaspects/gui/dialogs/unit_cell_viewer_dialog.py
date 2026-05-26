"""Unit Cell / Crystal Net viewer dialog.

Opens as a standalone window from the Tools menu.  Shows:
  - The unit cell box (always, when crystallography is available).
  - Molecule templates from the structure file positioned at their
    fractional coordinates inside the cell (when a structure file was loaded).
  - Interaction connections from a CrystalGrower net file (when imported),
    drawn as coloured line segments between molecule centroids with the
    correct source and target molecules shown at each end.

Two view modes are available once a net file is loaded:
  - Unit Cell: unit cell box + molecule templates at their crystallographic positions.
  - Net / Connections: source molecule(s) + all neighbour molecules at translated
    positions + coloured connection lines between them.

All geometry is centred at the scene origin so rotation always happens at
the centre of mass, not the world origin.
"""

import logging
import re
from pathlib import Path
from typing import List

import numpy as np
from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QMatrix4x4, QPainter, QVector2D
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ...fileio.cg_net import CGNet, Molecule
from ...fileio.structure import Structure
from ..utils.crystallography import Crystallography
from ..visualisation.atom_renderer import AtomRenderer
from ..visualisation.bond_renderer import BondRenderer
from ..visualisation.camera import Camera
from ..visualisation.unit_cell_renderer import (
    _AXIS_COLORS,
    _EDGES,
    _FRAC_CORNERS,
    UnitCellRenderer,
)
from ..visualisation.visual_data import VisualData

logger = logging.getLogger("CGA:UnitCellViewer")

_BOND_RADIUS = 0.15

# Distinct colours for interaction shells (cycled when more shells than entries)
_SHELL_COLORS = np.array(
    [
        [0.90, 0.20, 0.20],  # red
        [0.20, 0.70, 0.95],  # cyan-blue
        [0.20, 0.85, 0.30],  # green
        [0.95, 0.60, 0.10],  # orange
        [0.70, 0.20, 0.90],  # purple
        [0.95, 0.90, 0.15],  # yellow
        [0.50, 0.85, 0.65],  # teal
        [0.90, 0.45, 0.70],  # pink
    ],
    dtype=np.float32,
)


def _parse_mol_type(label: str) -> int | None:
    """Extract the integer mol_type from a net-file label like '1A' → 1."""
    m = re.match(r"(\d+)", label)
    return int(m.group(1)) if m else None


def _parse_translation(molecule_info: str) -> tuple[int, int, int] | None:
    """Extract (tx, ty, tz) integers from a molecule_info string like '(1,0,-1)'."""
    nums = re.findall(r"-?\d+", molecule_info)
    if len(nums) >= 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    return None


# ---------------------------------------------------------------------------
# OpenGL viewer widget
# ---------------------------------------------------------------------------


class UnitCellViewerWidget(QOpenGLWidget):
    """Lightweight QOpenGLWidget that renders a unit cell and optional data."""

    itemTreeChanged = Signal()  # emitted when the available atoms/molecules/connections change

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self._crystallography: Crystallography | None = None
        self._templates: dict | None = None  # mol_type → precomputed template dict
        self._net_molecules: List[Molecule] | None = None

        # Mol types to fan connection lines from (empty = no connections shown)
        self._conn_source_types: set = set()

        # Scene centre used to offset all geometry so rotation is at centre of mass
        self._scene_centre: np.ndarray = np.zeros(3, dtype=np.float32)

        # Selection and visibility
        self._selected_ids: set = set()   # {("mol", type), ("atom", type, idx), ("conn", idx)}
        self._hidden_ids: set = set()     # same ID tuples; matched items are skipped in upload

        # Display toggles
        self._show_molecules = True
        self._show_connections = True
        self._show_atom_labels = False
        self._show_mol_labels = False

        # Label geometry cache (rebuilt on geometry upload)
        self._atom_label_data: list = []
        self._mol_label_data: list = []

        # Appearance parameters
        self._atom_radius_scale: float = 1.0
        self._bond_radius: float = _BOND_RADIUS

        # OpenGL resources (created in initializeGL)
        self._atom_renderer: AtomRenderer | None = None
        self._bond_renderer: BondRenderer | None = None
        self._cell_renderer: UnitCellRenderer | None = None
        self._conn_renderer: UnitCellRenderer | None = None
        self._initialized = False

        # Camera
        self._camera = Camera()
        self._last_mouse_pos = None
        self._aspect_ratio = 1.0
        self._view_fitted = False

        # Dirty flag — triggers geometry rebuild on next paint
        self._dirty = True

        self.setMinimumSize(520, 420)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ------------------------------------------------------------------
    # Qt OpenGL overrides
    # ------------------------------------------------------------------

    def initializeGL(self):
        gl = self.context().extraFunctions()
        self._atom_renderer = AtomRenderer(gl)
        self._bond_renderer = BondRenderer(gl)
        self._cell_renderer = UnitCellRenderer(gl)
        self._conn_renderer = UnitCellRenderer(gl)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glClearColor(0.12, 0.12, 0.12, 1.0)
        self._initialized = True
        self._dirty = True

    def resizeGL(self, w, h):
        self._aspect_ratio = w / max(h, 1)

    def paintGL(self):
        gl = self.context().extraFunctions()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self._initialized:
            return

        if self._dirty:
            self._upload_geometry()
            self._dirty = False

        uniforms = self._build_uniforms()

        # Cell box
        if self._cell_renderer.numberOfVertices() > 0:
            self._cell_renderer.bind()
            self._cell_renderer.setUniforms(**uniforms)
            self._cell_renderer.draw(gl)
            self._cell_renderer.release()

        # Atoms + bonds
        if self._show_molecules:
            if self._atom_renderer.numberOfInstances() > 0:
                self._atom_renderer.bind(gl)
                self._atom_renderer.setUniforms(**uniforms)
                self._atom_renderer.draw(gl)
                self._atom_renderer.release()
            if self._bond_renderer.numberOfInstances() > 0:
                self._bond_renderer.bind(gl)
                self._bond_renderer.setUniforms(**uniforms)
                self._bond_renderer.draw(gl)
                self._bond_renderer.release()

        # Interaction connections
        if self._show_connections and self._conn_renderer.numberOfVertices() > 0:
            self._conn_renderer.bind()
            self._conn_renderer.setUniforms(**uniforms)
            self._conn_renderer.draw(gl)
            self._conn_renderer.release()

        if self._show_atom_labels or self._show_mol_labels:
            self._draw_labels()

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_crystallography(self, crystallography: Crystallography | None):
        self._crystallography = crystallography
        self._view_fitted = False
        self._mark_dirty()

    def set_structure(self, structure: Structure | None):
        if structure is None or not structure.templates:
            self._templates = None
        elif self._crystallography is not None:
            self._templates = VisualData._build_cart_templates(
                structure.templates, self._crystallography
            )
        else:
            self._templates = None
        self._selected_ids = set()
        self._hidden_ids = set()
        self._view_fitted = False
        self._mark_dirty()
        self.itemTreeChanged.emit()

    def set_net_molecules(self, molecules: List[Molecule] | None):
        self._net_molecules = molecules
        self._selected_ids = set()
        self._hidden_ids = set()
        self._mark_dirty()
        self.itemTreeChanged.emit()

    def set_conn_source_types(self, mol_types: set):
        """Set which mol types fan out connection lines; triggers geometry rebuild."""
        self._conn_source_types = set(mol_types)
        self._view_fitted = False
        self._mark_dirty()

    def set_show_molecules(self, enabled: bool):
        self._show_molecules = enabled
        self.update()

    def set_show_connections(self, enabled: bool):
        self._show_connections = enabled
        self.update()

    def reset_view(self):
        self._view_fitted = False
        self._dirty = True
        self.update()

    # ------------------------------------------------------------------
    # Selection / visibility API (called from dialog's selection panel)
    # ------------------------------------------------------------------

    def set_selected_ids(self, ids: set):
        self._selected_ids = set(ids)
        self._mark_dirty()

    def hide_selected(self):
        self._hidden_ids = self._hidden_ids | self._selected_ids
        self._mark_dirty()

    def hide_unselected(self):
        self._hidden_ids = self.get_all_item_ids() - self._selected_ids
        self._mark_dirty()

    def show_all(self):
        if self._hidden_ids:
            self._hidden_ids = set()
            self._mark_dirty()

    def get_all_item_ids(self) -> set:
        """Return the complete set of item IDs currently in the scene."""
        ids: set = set()
        if self._templates:
            for mol_type, tmpl in self._templates.items():
                ids.add(("mol", mol_type))
                for i in range(len(tmpl["symbols"])):
                    ids.add(("atom", mol_type, i))
        if self._net_molecules:
            idx = 0
            for mol in self._net_molecules:
                for _intr in mol.interactions:
                    ids.add(("conn", idx))
                    idx += 1
        return ids

    def set_show_atom_labels(self, enabled: bool):
        self._show_atom_labels = enabled
        self.update()

    def set_show_mol_labels(self, enabled: bool):
        self._show_mol_labels = enabled
        self.update()

    def set_atom_radius_scale(self, scale: float):
        self._atom_radius_scale = max(0.1, float(scale))
        self._mark_dirty()

    def set_bond_radius(self, radius: float):
        self._bond_radius = max(0.01, float(radius))
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_dirty(self):
        self._dirty = True
        self.update()

    def _upload_geometry(self):
        """Rebuild and upload all GPU geometry from current data."""
        if self._crystallography is None:
            return

        scene_centre = self._crystallography.frac_to_cart(
            np.array([0.5, 0.5, 0.5], dtype=np.float64)
        ).astype(np.float32)
        self._scene_centre = scene_centre

        self._cell_renderer.set_lines(self._build_cell_lines(scene_centre))

        atom_blocks = []
        bond_blocks = []
        conn_vertices = []
        self._atom_label_data = []
        self._mol_label_data = []

        # Pre-compute which mol types / connection instances are highlighted by a
        # selected connection, so we can highlight both endpoint molecules.
        selected_src_types: set = set()
        selected_tgt_conn_idxs: set = set()
        if self._net_molecules and self._conn_source_types:
            ci = 0
            for mol in self._net_molecules:
                src_type = _parse_mol_type(mol.label)
                for _intr in mol.interactions:
                    if ("conn", ci) in self._selected_ids and src_type in self._conn_source_types:
                        if src_type is not None:
                            selected_src_types.add(src_type)
                        selected_tgt_conn_idxs.add(ci)
                    ci += 1

        # Track (mol_type, tx, ty, tz) instances already drawn to avoid Z-fighting
        # from duplicate molecules at overlapping positions.
        drawn_instances: set = set()

        # --- Unit cell molecules ---
        if self._templates:
            for mol_type, tmpl in self._templates.items():
                if ("mol", mol_type) in self._hidden_ids:
                    continue
                drawn_instances.add((mol_type, 0, 0, 0))

                n = len(tmpl["cart"])
                pos = tmpl["cart"] - scene_centre
                colors = tmpl["colors"]
                radii = tmpl["radii"]

                atom_hidden = np.array(
                    [("atom", mol_type, i) in self._hidden_ids for i in range(n)], dtype=bool
                )
                visible = ~atom_hidden
                if not visible.any():
                    continue

                mol_sel = (
                    ("mol", mol_type) in self._selected_ids
                    or mol_type in selected_src_types
                )
                sel = np.array(
                    [[1.0 if (mol_sel or ("atom", mol_type, i) in self._selected_ids) else 0.0]
                     for i in range(n)],
                    dtype=np.float32,
                )

                vis_pos = pos[visible]
                vis_radii = radii[visible]
                vis_colors = colors[visible]
                atom_blocks.append(np.hstack([
                    vis_pos, vis_colors, sel[visible],
                    (vis_radii * self._atom_radius_scale)[:, None],
                ]))

                for a1, a2 in tmpl["bonds"]:
                    if a1 < n and a2 < n and visible[a1] and visible[a2]:
                        p1, p2 = pos[a1], pos[a2]
                        mid = (p1 + p2) * 0.5
                        bond_blocks.append(np.concatenate([p1, mid, colors[a1], [self._bond_radius]]))
                        bond_blocks.append(np.concatenate([mid, p2, colors[a2], [self._bond_radius]]))

                for i, sym in enumerate(tmpl["symbols"]):
                    if visible[i]:
                        self._atom_label_data.append((pos[i].copy(), sym))
                self._mol_label_data.append(((tmpl["centroid"] - scene_centre).copy(), f"M{mol_type}"))

        # --- Connection neighbours and lines ---
        if self._net_molecules and self._conn_source_types and self._templates:
            unique_r = sorted(
                {intr.r for mol in self._net_molecules for intr in mol.interactions}
            )
            r_color = {r: _SHELL_COLORS[i % len(_SHELL_COLORS)] for i, r in enumerate(unique_r)}

            conn_idx = 0
            for mol in self._net_molecules:
                src_type = _parse_mol_type(mol.label)
                src_tmpl = self._templates.get(src_type) if src_type is not None else None

                if src_type not in self._conn_source_types or src_tmpl is None:
                    conn_idx += len(mol.interactions)
                    continue

                src_centroid = src_tmpl["centroid"] - scene_centre

                for intr in mol.interactions:
                    cid = ("conn", conn_idx)
                    conn_idx += 1

                    if cid in self._hidden_ids:
                        continue

                    tgt_type = _parse_mol_type(intr.mol_type)
                    tgt_tmpl = self._templates.get(tgt_type) if tgt_type is not None else None
                    if tgt_tmpl is None:
                        continue
                    trans = _parse_translation(intr.molecule_info)
                    if trans is None:
                        continue

                    trans_cart = self._crystallography.frac_to_cart(
                        np.array(trans, dtype=np.float64)
                    ).astype(np.float32)

                    tgt_pos = tgt_tmpl["cart"] + trans_cart - scene_centre
                    tgt_centroid = tgt_tmpl["centroid"] + trans_cart - scene_centre
                    tgt_colors = tgt_tmpl["colors"]
                    tgt_radii = tgt_tmpl["radii"]
                    nt = len(tgt_pos)

                    # Draw the neighbour molecule unless it coincides with an already-drawn instance
                    instance_key = (tgt_type, *trans)
                    if instance_key not in drawn_instances:
                        drawn_instances.add(instance_key)
                        tgt_sel = cid[1] in selected_tgt_conn_idxs
                        sel = np.full((nt, 1), 1.0 if tgt_sel else 0.0, dtype=np.float32)
                        atom_blocks.append(np.hstack([
                            tgt_pos, tgt_colors, sel,
                            (tgt_radii * self._atom_radius_scale)[:, None],
                        ]))
                        for a1, a2 in tgt_tmpl["bonds"]:
                            if a1 < nt and a2 < nt:
                                p1, p2 = tgt_pos[a1], tgt_pos[a2]
                                mid = (p1 + p2) * 0.5
                                bond_blocks.append(np.concatenate([p1, mid, tgt_colors[a1], [self._bond_radius]]))
                                bond_blocks.append(np.concatenate([mid, p2, tgt_colors[a2], [self._bond_radius]]))

                    color = r_color[intr.r]
                    if cid in self._selected_ids:
                        color = np.clip(color * 1.8, 0.0, 1.0)
                    conn_vertices.append(np.concatenate([src_centroid, color]))
                    conn_vertices.append(np.concatenate([tgt_centroid, color]))

        if atom_blocks:
            atom_arr = np.vstack(atom_blocks).astype(np.float32)
            self._atom_renderer.setPoints(atom_arr)
            if not self._view_fitted:
                self._camera.fitToObject(atom_arr[:, :3])
                self._view_fitted = True
        elif not self._view_fitted:
            cart = self._crystallography.frac_to_cart(_FRAC_CORNERS).astype(np.float32) - scene_centre
            self._camera.fitToObject(cart)
            self._view_fitted = True

        if bond_blocks:
            self._bond_renderer.setBonds(np.array(bond_blocks, dtype=np.float32))

        self._conn_renderer.set_lines(
            np.array(conn_vertices, dtype=np.float32).flatten() if conn_vertices
            else np.array([], dtype=np.float32)
        )

    def _build_cell_lines(self, offset: np.ndarray) -> np.ndarray:
        """Build unit cell edge vertex array offset so that cell centre is at origin."""
        cart = self._crystallography.frac_to_cart(_FRAC_CORNERS).astype(np.float32) - offset
        vertices = []
        for i0, i1, ax in _EDGES:
            color = _AXIS_COLORS[ax]
            vertices.append(np.concatenate([cart[i0], color]))
            vertices.append(np.concatenate([cart[i1], color]))
        return np.array(vertices, dtype=np.float32).flatten()

    def _project_to_screen(self, pos3d: np.ndarray) -> tuple | None:
        """Project a world-space position to screen pixel coordinates.

        Returns (x, y) or None if the point is behind the camera or off-screen.
        """
        mvp = self._camera.modelViewProjectionMatrix(self._aspect_ratio)
        m = np.array(mvp.copyDataTo(), dtype=np.float32).reshape(4, 4).T
        p = np.array([pos3d[0], pos3d[1], pos3d[2], 1.0], dtype=np.float32)
        clip = m @ p
        w = clip[3]
        if w <= 0.0:
            return None
        ndc_x = clip[0] / w
        ndc_y = clip[1] / w
        if abs(ndc_x) > 1.5 or abs(ndc_y) > 1.5:
            return None
        return (
            (ndc_x + 1.0) * 0.5 * self.width(),
            (1.0 - ndc_y) * 0.5 * self.height(),
        )

    def _draw_labels(self):
        """Draw atom symbol and/or molecule-type labels over the GL viewport."""
        painter = QPainter(self)
        painter.beginNativePainting()
        painter.endNativePainting()
        painter.setRenderHint(QPainter.Antialiasing)

        if self._show_atom_labels:
            painter.setFont(QFont("Arial", 9))
            painter.setPen(QColor(255, 255, 210))
            for pos3d, sym in self._atom_label_data:
                sp = self._project_to_screen(pos3d)
                if sp is not None:
                    painter.drawText(QPoint(int(sp[0]) + 4, int(sp[1]) - 4), sym)

        if self._show_mol_labels:
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QColor(255, 210, 60))
            for pos3d, text in self._mol_label_data:
                sp = self._project_to_screen(pos3d)
                if sp is not None:
                    painter.drawText(QPoint(int(sp[0]) + 4, int(sp[1]) - 4), text)

        painter.end()

    def _build_uniforms(self) -> dict:
        mvp = self._camera.modelViewProjectionMatrix(self._aspect_ratio)
        view = self._camera.viewMatrix()
        proj = self._camera.projectionMatrix(self._aspect_ratio)
        model_view = self._camera.modelViewMatrix()
        model = self._camera.modelMatrix()
        screen_size = QVector2D(self.width(), self.height())
        axes = QMatrix4x4()
        return {
            "u_modelMat": model,
            "u_modelRotMat": self._camera.modelRotationMatrix(),
            "u_viewMat": view,
            "u_modelViewProjectionMat": mvp,
            "u_modelViewMat": model_view,
            "u_projectionMat": proj,
            "u_pointSize": 6.0,
            "u_screenSize": screen_size,
            "u_scale": self._camera.scale,
            "u_lineScale": 2.0,
            "u_axesMat": axes,
        }

    # ------------------------------------------------------------------
    # Mouse / keyboard interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        self._last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        self._last_mouse_pos = None

    def mouseMoveEvent(self, event):
        if self._last_mouse_pos is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos
        if event.buttons() & Qt.LeftButton:
            self._camera.rotate_model(dx * 0.5, dy * 0.5)
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._camera.zoom(delta / 120.0)
        self.update()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class UnitCellViewerDialog(QDialog):
    """Tools → Unit Cell Viewer dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unit Cell Viewer")
        self.setMinimumSize(860, 580)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint
        )

        self._viewer = UnitCellViewerWidget()

        # --- Net file controls ---
        self._net_label = QLabel("No net file loaded")
        self._net_label.setWordWrap(True)

        self._import_btn = QPushButton("Import Net File…")
        self._import_btn.clicked.connect(self._on_import_net)

        self._clear_net_btn = QPushButton("Clear Net")
        self._clear_net_btn.setEnabled(False)
        self._clear_net_btn.clicked.connect(self._on_clear_net)

        # Checkboxes for selecting which unit-cell molecules to fan connections from;
        # populated dynamically after a net file is loaded.
        self._source_checkboxes: dict = {}  # mol_type → QCheckBox
        self._sources_group = QGroupBox("Show Connections From")
        self._sources_layout = QVBoxLayout(self._sources_group)
        self._sources_group.setVisible(False)

        # --- Display toggles ---
        self._show_mol_cb = QCheckBox("Show Molecules")
        self._show_mol_cb.setChecked(True)
        self._show_mol_cb.setEnabled(False)
        self._show_mol_cb.toggled.connect(self._viewer.set_show_molecules)

        self._show_conn_cb = QCheckBox("Show Connections")
        self._show_conn_cb.setChecked(True)
        self._show_conn_cb.setEnabled(False)
        self._show_conn_cb.toggled.connect(self._viewer.set_show_connections)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._viewer.reset_view)

        self._status_label = QLabel("Load a simulation folder with a structure file to view molecules.")
        self._status_label.setWordWrap(True)

        # --- Layout ---
        net_group = QGroupBox("Crystal Net")
        net_layout = QVBoxLayout(net_group)
        net_layout.addWidget(self._net_label)
        net_layout.addWidget(self._import_btn)
        net_layout.addWidget(self._clear_net_btn)
        net_layout.addWidget(self._sources_group)

        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.addWidget(self._show_mol_cb)
        display_layout.addWidget(self._show_conn_cb)
        display_layout.addWidget(self._reset_btn)

        # --- Selection panel ---
        sel_group = QGroupBox("Atoms / Molecules")
        sel_layout = QVBoxLayout(sel_group)

        self._sel_tree = QTreeWidget()
        self._sel_tree.setHeaderHidden(True)
        self._sel_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._sel_tree.setMinimumHeight(160)
        self._sel_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        sel_layout.addWidget(self._sel_tree)

        sel_btn_row = QHBoxLayout()
        self._hide_sel_btn = QPushButton("Hide")
        self._hide_sel_btn.setToolTip("Hide the selected items")
        self._hide_sel_btn.clicked.connect(self._on_hide_selected)
        self._isolate_btn = QPushButton("Isolate")
        self._isolate_btn.setToolTip("Hide everything except the selected items")
        self._isolate_btn.clicked.connect(self._on_hide_unselected)
        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.clicked.connect(self._on_show_all)
        sel_btn_row.addWidget(self._hide_sel_btn)
        sel_btn_row.addWidget(self._isolate_btn)
        sel_btn_row.addWidget(self._show_all_btn)
        sel_layout.addLayout(sel_btn_row)

        self._viewer.itemTreeChanged.connect(self._rebuild_item_list)

        # --- Appearance ---
        appearance_group = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_group)
        self._atom_radius_sb = QDoubleSpinBox()
        self._atom_radius_sb.setRange(0.1, 3.0)
        self._atom_radius_sb.setSingleStep(0.05)
        self._atom_radius_sb.setValue(1.0)
        self._atom_radius_sb.setDecimals(2)
        self._atom_radius_sb.setToolTip("Scale factor applied to all atom VdW radii")
        self._atom_radius_sb.valueChanged.connect(self._viewer.set_atom_radius_scale)
        appearance_form.addRow("Atom Radius Scale:", self._atom_radius_sb)
        self._bond_radius_sb = QDoubleSpinBox()
        self._bond_radius_sb.setRange(0.01, 1.0)
        self._bond_radius_sb.setSingleStep(0.01)
        self._bond_radius_sb.setValue(_BOND_RADIUS)
        self._bond_radius_sb.setDecimals(2)
        self._bond_radius_sb.setToolTip("Bond cylinder radius in Ångströms")
        self._bond_radius_sb.valueChanged.connect(self._viewer.set_bond_radius)
        appearance_form.addRow("Bond Radius (Å):", self._bond_radius_sb)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(net_group)
        ctrl_layout.addWidget(display_group)
        ctrl_layout.addWidget(sel_group)
        ctrl_layout.addWidget(appearance_group)
        ctrl_layout.addWidget(self._status_label)
        ctrl_layout.addStretch()

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(ctrl_layout, 0)
        main_layout.addWidget(self._viewer, 1)

    # ------------------------------------------------------------------
    # Public API (called from MainWindow)
    # ------------------------------------------------------------------

    def set_crystallography(self, crystallography: Crystallography | None):
        """Update the unit cell box.  Call whenever lattice parameters change."""
        self._viewer.set_crystallography(crystallography)

    def set_structure(self, structure: Structure | None, crystallography: Crystallography | None = None):
        """Update molecule templates.  Also accepts a fresh crystallography if needed."""
        if crystallography is not None:
            self._viewer.set_crystallography(crystallography)
        has_templates = bool(structure and structure.templates)
        self._viewer.set_structure(structure)
        self._show_mol_cb.setEnabled(has_templates)
        if has_templates:
            self._status_label.setText(
                f"Loaded {len(structure.templates)} molecule template(s)."
            )
        else:
            self._status_label.setText(
                "No molecule templates found — showing cell edges only."
            )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_import_net(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Crystal Net File", "", "Net Files (*.txt *.net);;All Files (*)"
        )
        if not path:
            return
        try:
            net = CGNet(path)
            net.parse()
        except Exception as exc:
            logger.exception("Failed to load net file: %s", exc)
            QMessageBox.warning(self, "Error", f"Failed to load net file:\n{exc}")
            return

        if not net.molecules:
            QMessageBox.information(
                self, "Empty Net", "The net file contained no molecule interactions."
            )
            return

        self._viewer.set_net_molecules(net.molecules)
        self._net_label.setText(Path(path).name)
        self._clear_net_btn.setEnabled(True)
        self._show_conn_cb.setEnabled(True)
        n_mol = len(net.molecules)
        n_int = sum(m.n_interactions for m in net.molecules)
        self._status_label.setText(
            f"Net: {n_mol} molecule type(s), {n_int} interactions."
        )
        self._populate_source_checkboxes(net.molecules)

    def _on_clear_net(self):
        self._viewer.set_net_molecules(None)
        self._viewer.set_conn_source_types(set())
        self._net_label.setText("No net file loaded")
        self._show_conn_cb.setEnabled(False)
        self._clear_net_btn.setEnabled(False)
        self._status_label.setText("Net cleared.")
        self._clear_source_checkboxes()

    def _populate_source_checkboxes(self, molecules):
        """Build one checkbox per unique source molecule type from the net."""
        self._clear_source_checkboxes()
        seen = set()
        for mol in molecules:
            mol_type = _parse_mol_type(mol.label)
            if mol_type is None or mol_type in seen:
                continue
            seen.add(mol_type)
            cb = QCheckBox(f"M{mol_type}  ({mol.label})")
            cb.setChecked(False)
            cb.toggled.connect(self._on_source_changed)
            self._source_checkboxes[mol_type] = cb
            self._sources_layout.addWidget(cb)
        self._sources_group.setVisible(bool(seen))

    def _clear_source_checkboxes(self):
        for cb in self._source_checkboxes.values():
            self._sources_layout.removeWidget(cb)
            cb.deleteLater()
        self._source_checkboxes.clear()
        self._sources_group.setVisible(False)

    def _on_source_changed(self):
        checked = {mt for mt, cb in self._source_checkboxes.items() if cb.isChecked()}
        self._viewer.set_conn_source_types(checked)

    # ------------------------------------------------------------------
    # Selection panel
    # ------------------------------------------------------------------

    def _rebuild_item_list(self):
        """Repopulate the tree from whatever is currently loaded in the viewer."""
        self._sel_tree.blockSignals(True)
        self._sel_tree.clear()

        templates = self._viewer._templates
        if templates:
            mol_root = QTreeWidgetItem(self._sel_tree, ["Molecules"])
            mol_root.setFlags(mol_root.flags() & ~Qt.ItemIsSelectable)
            for mol_type, tmpl in sorted(templates.items()):
                mol_item = QTreeWidgetItem(mol_root, [f"M{mol_type}"])
                mol_item.setData(0, Qt.UserRole, ("mol", mol_type))
                for i, sym in enumerate(tmpl["symbols"]):
                    atom_item = QTreeWidgetItem(mol_item, [f"{sym}  (atom {i})"])
                    atom_item.setData(0, Qt.UserRole, ("atom", mol_type, i))
                mol_item.setExpanded(True)
            mol_root.setExpanded(True)

        net_mols = self._viewer._net_molecules
        if net_mols:
            conn_root = QTreeWidgetItem(self._sel_tree, ["Connections"])
            conn_root.setFlags(conn_root.flags() & ~Qt.ItemIsSelectable)
            idx = 0
            for mol in net_mols:
                for intr in mol.interactions:
                    label = f"{mol.label} → {intr.mol_type}  (r={intr.r:.1f} Å)"
                    item = QTreeWidgetItem(conn_root, [label])
                    item.setData(0, Qt.UserRole, ("conn", idx))
                    idx += 1
            conn_root.setExpanded(True)

        self._sel_tree.blockSignals(False)
        self._update_tree_hidden_state()

    def _on_tree_selection_changed(self):
        ids: set = set()
        for item in self._sel_tree.selectedItems():
            item_id = item.data(0, Qt.UserRole)
            if item_id is not None:
                ids.add(item_id)
        self._viewer.set_selected_ids(ids)

    def _on_hide_selected(self):
        self._viewer.hide_selected()
        self._update_tree_hidden_state()

    def _on_hide_unselected(self):
        self._viewer.hide_unselected()
        self._update_tree_hidden_state()

    def _on_show_all(self):
        self._viewer.show_all()
        self._update_tree_hidden_state()

    def _update_tree_hidden_state(self):
        """Dim tree items that are currently hidden in the viewer."""
        hidden = self._viewer._hidden_ids
        dim = QBrush(QColor(100, 100, 100))
        normal = QBrush(QColor(220, 220, 220))

        def _apply(item: QTreeWidgetItem):
            item_id = item.data(0, Qt.UserRole)
            brush = dim if (item_id is not None and item_id in hidden) else normal
            item.setForeground(0, brush)
            for i in range(item.childCount()):
                _apply(item.child(i))

        root = self._sel_tree.invisibleRootItem()
        for i in range(root.childCount()):
            _apply(root.child(i))

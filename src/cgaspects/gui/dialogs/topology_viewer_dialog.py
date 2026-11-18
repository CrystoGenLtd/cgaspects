"""Topology Viewer dialog: the unit cell, crystal net and site-type environments.

Opens as a standalone window from the Crystallography menu.  The Unit Cell mode shows:
  - The unit cell box (or a user-grown supercell grid).
  - Molecule templates from the structure file positioned at their
    fractional coordinates inside each cell of the supercell.
  - The crystal net parsed from the structure file itself: each tile
    (molecule) lists its face neighbours with relative unit-cell offsets,
    drawn as coloured line segments between molecule centroids.
  - Optionally, a CrystoGen net file imported on top to annotate each
    connection with its interaction distance and energy.

Molecule instances (a tile in a specific cell of the supercell), individual
atoms, and individual connections can all be selected, hidden, or isolated
from the tree panel.  Connections are drawn only from molecules that are
visible and whose tile is ticked under "Show Connections From".

All geometry is centred at the supercell centre so rotation always happens
at the centre of mass, not the world origin.

The Site Types mode reads a checkpoint, lists the distinct neighbour environments
of a chosen site type by how often they occur, draws the selected environment and
exports it as a docking site.
"""

import dataclasses
import logging
from pathlib import Path

import numpy as np
from OpenGL.GL import GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST
from PySide6.QtCore import Qt, QPoint, QRect, QThreadPool, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QMatrix4x4,
    QPainter,
    QVector2D,
)
from PySide6.QtOpenGL import QOpenGLFramebufferObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...analysis.gui_threads import WorkerSiteEnvironmentBuild, WorkerSiteEnvironmentLoad
from ...analysis.site_environments import SiteEnvironment, SiteEnvironmentAnalysis
from ...fileio.cg_net import CGNet
from ...fileio.pdb_export import write_docking_pdb
from ...fileio.structure import Structure, TileConnection
from ..animation.keyframe import AnimationTimeline, Keyframe
from ..animation.timeline_widget import KeyframeTimelineWidget
from ..utils.crystallography import Crystallography
from ..visualisation.atom_renderer import AtomRenderer
from ..visualisation.bond_renderer import BondRenderer
from ..visualisation.camera import Camera
from ..visualisation.multicolor_sphere_renderer import (
    MAX_SECTIONS,
    MultiColorSphereRenderer,
)
from ..visualisation.shading import RenderSettings
from ..visualisation.unit_cell_renderer import (
    _AXIS_COLORS,
    _EDGES,
    _FRAC_CORNERS,
    UnitCellRenderer,
)
from ..visualisation.visual_data import VisualData
from .topology_appearance_dialog import TopologyAppearanceDialog

logger = logging.getLogger("CGA:TopologyViewer")

_BOND_RADIUS = 0.15
_MAX_SUPERCELL = 6
# Orbit step (dx, dy) per arrow key press, matching the Crystal Viewer.
_ARROW_KEY_ROTATION = {
    Qt.Key_Up: (0, -10),
    Qt.Key_Down: (0, 10),
    Qt.Key_Left: (-10, 0),
    Qt.Key_Right: (10, 0),
}

# Atoms whose positions round to the same value at this many decimals (Å) are
# treated as occupying the same lattice site and drawn as one sectioned sphere.
_OVERLAP_DECIMALS = 2

# Distinct colours for interaction groups (cycled when more groups than entries)
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

# Site Types mode colours, matching the main viewport's docking shell colours
_SITE_TARGET_COLOR = np.array([1.0, 0.35, 0.0], dtype=np.float32)
_SITE_FIRST_COLOR = np.array([0.0, 0.85, 0.2], dtype=np.float32)
_SITE_SECOND_COLOR = np.array([0.15, 0.45, 1.0], dtype=np.float32)
_SITE_EMPTY_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float32)
_SITE_EMPTY_RADIUS = 0.45
_SITE_CONTACT_RADIUS = 0.08
_SHARE_BAR_COLOR = QColor(255, 140, 40, 110)
_ENV_SHARE_COLUMN = 4

_MODE_SITE_TYPES = 1


# ---------------------------------------------------------------------------
# OpenGL viewer widget
# ---------------------------------------------------------------------------


class TopologyViewerWidget(QOpenGLWidget):
    """Lightweight QOpenGLWidget that renders a supercell of the crystal net.

    Item IDs used for selection and visibility:
      ("mol", tile, (i, j, k))  — one molecule instance in one supercell cell
      ("atom", tile, idx)       — atom idx of a tile, in all of its instances
      ("conn", tile, k)         — k-th connection of a tile, in all instances
    """

    itemTreeChanged = Signal()  # emitted when the available items change

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self._crystallography: Crystallography | None = None
        self._templates: dict | None = None  # tile → precomputed template dict
        self._connections: dict[int, list[TileConnection]] = {}
        # (tile, conn_idx) → {"r": float | None, "energy": float | str | None}
        self._conn_annotations: dict | None = None

        # Supercell extent along a, b, c
        self._supercell: tuple[int, int, int] = (1, 1, 1)

        # Tiles to fan connection lines from (empty = no connections shown)
        self._conn_source_tiles: set = set()

        # Scene centre used to offset all geometry so rotation is at centre of mass
        self._scene_centre: np.ndarray = np.zeros(3, dtype=np.float32)

        # Selection and visibility (item ID tuples, see class docstring)
        self._selected_ids: set = set()
        self._hidden_ids: set = set()

        # Display toggles
        self._show_cell = True
        self._show_molecules = True
        self._show_connections = True
        self._show_neighbours = False  # ghost molecules outside the supercell
        self._conn_tubes = False  # draw connections as energy-scaled tubes
        self._show_atom_labels = False
        self._show_mol_labels = False

        # Site Types mode: one neighbour environment replaces the supercell
        self._site_mode = False
        self._site_env: SiteEnvironment | None = None
        self._site_contacts: list = []
        self._site_show_first = True
        self._site_show_second = True
        self._site_show_empty = True
        self._site_show_contacts = True
        self._site_color_by_shell = True

        # Label geometry cache (rebuilt on geometry upload)
        self._atom_label_data: list = []
        self._mol_label_data: list = []

        # Appearance parameters
        self._atom_radius_scale: float = 1.0
        self._bond_radius: float = _BOND_RADIUS
        self._conn_radius_scale: float = 1.0
        self._render_settings = RenderSettings()

        # Compatibility surface for raytrace_export.build_scene_from_widget()
        # and the keyframe/video-render pipeline, both written against the
        # main VisualisationWidget's public attribute names.
        self.is_atom_view = True
        self.point_size = 6.0  # matches the fixed u_pointSize uniform below
        self.backgroundColor = QColor.fromRgbF(0.12, 0.12, 0.12)

        # OpenGL resources (created in initializeGL)
        self._atom_renderer: AtomRenderer | None = None
        self._multi_atom_renderer: MultiColorSphereRenderer | None = None
        self._bond_renderer: BondRenderer | None = None
        self._cell_renderer: UnitCellRenderer | None = None
        self._conn_renderer: UnitCellRenderer | None = None
        self._conn_tube_renderer: BondRenderer | None = None
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
        self._multi_atom_renderer = MultiColorSphereRenderer(gl)
        self._bond_renderer = BondRenderer(gl)
        self._cell_renderer = UnitCellRenderer(gl)
        self._conn_renderer = UnitCellRenderer(gl)
        self._conn_tube_renderer = BondRenderer(gl)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glClearColor(0.12, 0.12, 0.12, 1.0)
        self._initialized = True
        self._dirty = True

    def resizeGL(self, w, h):
        self._aspect_ratio = w / max(h, 1)

    # ------------------------------------------------------------------
    # Compatibility properties (raytrace_export.build_scene_from_widget,
    # the animation render worker) — mirror the main VisualisationWidget's
    # public attribute names so those modules work unmodified.
    # ------------------------------------------------------------------

    @property
    def camera(self):
        return self._camera

    @property
    def atom_renderer(self):
        return self._atom_renderer

    @property
    def bond_renderer(self):
        return self._bond_renderer

    @property
    def conn_tube_renderer(self):
        return self._conn_tube_renderer

    @property
    def render_settings(self):
        return self._render_settings

    def paintGL(self):
        gl = self.context().extraFunctions()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if not self._initialized:
            return

        if self._dirty:
            self._upload_geometry()
            self._dirty = False

        uniforms = self._build_uniforms()
        self._draw_scene(gl, uniforms)

        if self._show_atom_labels or self._show_mol_labels:
            self._draw_labels()

    def _draw_scene(self, gl, uniforms: dict):
        """Draw the cell box, molecules and connections. Shared by paintGL and
        offscreen FBO rendering (image/video export)."""
        # Cell box(es)
        if self._show_cell and self._cell_renderer.numberOfVertices() > 0:
            self._cell_renderer.bind()
            self._cell_renderer.setUniforms(**uniforms)
            self._cell_renderer.draw(gl)
            self._cell_renderer.release()

        # Atoms + bonds
        if self._show_molecules or self._site_mode:
            if self._atom_renderer.numberOfInstances() > 0:
                self._atom_renderer.bind(gl)
                self._atom_renderer.setUniforms(**uniforms)
                self._atom_renderer.draw(gl)
                self._atom_renderer.release()
            if self._multi_atom_renderer.numberOfInstances() > 0:
                self._multi_atom_renderer.bind(gl)
                self._multi_atom_renderer.setUniforms(**uniforms)
                self._multi_atom_renderer.draw(gl)
                self._multi_atom_renderer.release()
            if self._bond_renderer.numberOfInstances() > 0:
                self._bond_renderer.bind(gl)
                self._bond_renderer.setUniforms(**uniforms)
                self._bond_renderer.draw(gl)
                self._bond_renderer.release()

        # Interaction connections (lines or energy-scaled tubes)
        if self._site_show_contacts if self._site_mode else self._show_connections:
            if self._conn_tubes:
                if self._conn_tube_renderer.numberOfInstances() > 0:
                    self._conn_tube_renderer.bind(gl)
                    self._conn_tube_renderer.setUniforms(**uniforms)
                    self._conn_tube_renderer.draw(gl)
                    self._conn_tube_renderer.release()
            elif self._conn_renderer.numberOfVertices() > 0:
                self._conn_renderer.bind()
                self._conn_renderer.setUniforms(
                    **{**uniforms, "u_lineScale": 2.0 * self._conn_radius_scale}
                )
                self._conn_renderer.draw(gl)
                self._conn_renderer.release()

    # ------------------------------------------------------------------
    # Data setters
    # ------------------------------------------------------------------

    def set_crystallography(self, crystallography: Crystallography | None):
        self._crystallography = crystallography
        self._view_fitted = False
        self._mark_dirty()

    def set_structure(self, structure: Structure | None):
        if structure is None or not structure.templates or self._crystallography is None:
            self._templates = None
        else:
            self._templates = VisualData._build_cart_templates(
                structure.templates, self._crystallography
            )
        self._connections = dict(structure.connections) if structure is not None else {}
        self._conn_annotations = None
        self._selected_ids = set()
        self._hidden_ids = set()
        self._view_fitted = False
        self._mark_dirty()
        self.itemTreeChanged.emit()

    def set_supercell(self, na: int, nb: int, nc: int):
        supercell = (max(1, int(na)), max(1, int(nb)), max(1, int(nc)))
        if supercell == self._supercell:
            return
        self._supercell = supercell
        # Drop stale per-instance IDs that reference removed cells
        valid = set(self.cells())
        self._hidden_ids = {
            i for i in self._hidden_ids if i[0] != "mol" or i[2] in valid
        }
        self._selected_ids = {
            i for i in self._selected_ids if i[0] != "mol" or i[2] in valid
        }
        self._view_fitted = False
        self._mark_dirty()
        self.itemTreeChanged.emit()

    def set_conn_source_tiles(self, tiles: set):
        """Set which tiles fan out connection lines; triggers geometry rebuild."""
        self._conn_source_tiles = set(tiles)
        self._mark_dirty()

    def set_conn_annotations(self, annotations: dict | None):
        """Attach net-file r/energy annotations keyed by (tile, conn_idx)."""
        self._conn_annotations = annotations
        self._mark_dirty()

    def set_site_mode(self, enabled: bool):
        """Switch between the unit-cell scene and the site-environment scene."""
        self._site_mode = enabled
        self._view_fitted = False
        self._mark_dirty()

    def set_site_environment(self, env: SiteEnvironment | None, contacts: list):
        """Show one neighbour environment; ``None`` leaves the Site Types scene empty."""
        self._site_env = env
        self._site_contacts = contacts
        self._view_fitted = False
        self._mark_dirty()

    def set_site_display(
        self, first: bool, second: bool, empty: bool, contacts: bool, color_by_shell: bool
    ):
        """Choose which parts of the site environment are drawn, and their colouring."""
        self._site_show_first = first
        self._site_show_second = second
        self._site_show_empty = empty
        self._site_show_contacts = contacts
        self._site_color_by_shell = color_by_shell
        self._mark_dirty()

    def set_show_cell(self, enabled: bool):
        self._show_cell = enabled
        self.update()

    def set_show_molecules(self, enabled: bool):
        self._show_molecules = enabled
        self.update()

    def set_show_connections(self, enabled: bool):
        self._show_connections = enabled
        self.update()

    def set_show_neighbours(self, enabled: bool):
        self._show_neighbours = enabled
        self._mark_dirty()

    def set_conn_tubes(self, enabled: bool):
        self._conn_tubes = enabled
        self._mark_dirty()

    def set_render_settings(self, settings: RenderSettings):
        """Apply material / lighting / ambient-occlusion settings from the dialog."""
        self._render_settings = settings
        if self._atom_renderer is not None:
            self._atom_renderer.set_ao_enabled(settings.ao_enabled)
        if self._multi_atom_renderer is not None:
            self._multi_atom_renderer.set_ao_enabled(settings.ao_enabled)
        self.update()

    def reset_view(self):
        self._camera.resetOrientation()
        self._view_fitted = False
        self._dirty = True
        self.update()

    def cells(self) -> list[tuple[int, int, int]]:
        """All (i, j, k) cells of the current supercell."""
        na, nb, nc = self._supercell
        return [
            (i, j, k) for i in range(na) for j in range(nb) for k in range(nc)
        ]

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
        keep = set(self._selected_ids)
        sel_mol_tiles = {i[1] for i in self._selected_ids if i[0] == "mol"}
        sel_atom_tiles = {i[1] for i in self._selected_ids if i[0] == "atom"}

        # Isolating a connection keeps both endpoint molecules visible
        for item_id in self._selected_ids:
            if item_id[0] != "conn":
                continue
            tile, k = item_id[1], item_id[2]
            conns = self._connections.get(tile, [])
            if k >= len(conns):
                continue
            conn = conns[k]
            for cell in self.cells():
                keep.add(("mol", tile, cell))
                keep.add(
                    (
                        "mol",
                        conn.target,
                        (
                            cell[0] + conn.offset[0],
                            cell[1] + conn.offset[1],
                            cell[2] + conn.offset[2],
                        ),
                    )
                )

        # Isolating molecules keeps their connections visible
        for tile in sel_mol_tiles:
            keep |= {("conn", tile, k) for k in range(len(self._connections.get(tile, ())))}

        # Atom IDs are global per tile: hiding them would blank the kept
        # molecules too.  Only hide atoms within tiles where the user
        # explicitly selected individual atoms.
        if self._templates:
            for tile, tmpl in self._templates.items():
                if tile not in sel_atom_tiles:
                    keep |= {("atom", tile, i) for i in range(len(tmpl["symbols"]))}

        self._hidden_ids = self.get_all_item_ids() - keep
        self._mark_dirty()

    def show_all(self):
        if self._hidden_ids:
            self._hidden_ids = set()
            self._mark_dirty()

    def get_all_item_ids(self) -> set:
        """Return the complete set of item IDs currently in the scene."""
        ids: set = set()
        cells = self.cells()
        if self._templates:
            for tile, tmpl in self._templates.items():
                for cell in cells:
                    ids.add(("mol", tile, cell))
                for i in range(len(tmpl["symbols"])):
                    ids.add(("atom", tile, i))
        for tile, conns in self._connections.items():
            for k in range(len(conns)):
                ids.add(("conn", tile, k))
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

    def set_conn_radius_scale(self, scale: float):
        self._conn_radius_scale = max(0.1, float(scale))
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

    def _cell_cart(self, cell: tuple[int, int, int]) -> np.ndarray:
        return self._crystallography.frac_to_cart(
            np.array(cell, dtype=np.float64)
        ).astype(np.float32)

    def _supercell_centre(self) -> np.ndarray:
        na, nb, nc = self._supercell
        return self._crystallography.frac_to_cart(
            np.array([na * 0.5, nb * 0.5, nc * 0.5], dtype=np.float64)
        ).astype(np.float32)

    def _visible_geometry_centre(self, cells: list) -> np.ndarray:
        """Bounding-box centre of the currently visible molecule atoms.

        Used as the geometry offset so rotation pivots at the centre of what is
        actually shown.  Falls back to the supercell centre when no molecule is
        visible (e.g. cell-edges-only view).
        """
        if not self._templates:
            return self._supercell_centre()

        lo = None
        hi = None
        for cell in cells:
            cell_cart = self._cell_cart(cell)
            for tile, tmpl in self._templates.items():
                if ("mol", tile, cell) in self._hidden_ids:
                    continue
                vis = [
                    i
                    for i in range(len(tmpl["cart"]))
                    if ("atom", tile, i) not in self._hidden_ids
                ]
                if not vis:
                    continue
                pts = tmpl["cart"][vis] + cell_cart
                pmin, pmax = pts.min(axis=0), pts.max(axis=0)
                lo = pmin if lo is None else np.minimum(lo, pmin)
                hi = pmax if hi is None else np.maximum(hi, pmax)

        if lo is None:
            return self._supercell_centre()
        return ((lo + hi) * 0.5).astype(np.float32)

    def _conn_group_values(self) -> dict:
        """(tile, conn_idx) → value used to colour-group connections.

        Uses the net-file energy when annotated, then the annotated r,
        falling back to the centroid–centroid distance computed from the
        structure geometry (identical for all supercell instances).
        """
        values: dict = {}
        if not self._connections or not self._templates:
            return values
        for tile, conns in self._connections.items():
            src_tmpl = self._templates.get(tile)
            for k, conn in enumerate(conns):
                ann = (self._conn_annotations or {}).get((tile, k))
                value = None
                if ann is not None:
                    energy = ann.get("energy")
                    if isinstance(energy, (int, float)):
                        value = round(float(energy), 6)
                    elif ann.get("r") is not None:
                        value = round(float(ann["r"]), 3)
                if value is None and src_tmpl is not None:
                    tgt_tmpl = self._templates.get(conn.target)
                    if tgt_tmpl is not None:
                        d = (
                            tgt_tmpl["centroid"]
                            + self._cell_cart(conn.offset)
                            - src_tmpl["centroid"]
                        )
                        value = round(float(np.linalg.norm(d)), 3)
                values[(tile, k)] = value
        return values

    def _emit_molecule(
        self,
        tile: int,
        cell_cart: np.ndarray,
        selected: bool,
        atom_blocks: list,
        bond_blocks: list,
        color: np.ndarray | None = None,
    ):
        """Append one molecule instance's atoms/bonds/labels to the draw lists.

        ``color`` paints the whole molecule one colour instead of by element.
        """
        tmpl = self._templates[tile]
        n = len(tmpl["cart"])
        pos = tmpl["cart"] + cell_cart - self._scene_centre
        colors = tmpl["colors"] if color is None else np.tile(color, (n, 1))

        # Unit-cell hiding and selection do not apply to site environments
        hidden_ids = set() if self._site_mode else self._hidden_ids
        selected_ids = set() if self._site_mode else self._selected_ids
        visible = np.array(
            [("atom", tile, i) not in hidden_ids for i in range(n)], dtype=bool
        )
        if not visible.any():
            return

        sel = np.array(
            [
                [1.0 if (selected or ("atom", tile, i) in selected_ids) else 0.0]
                for i in range(n)
            ],
            dtype=np.float32,
        )

        atom_blocks.append(
            np.hstack(
                [
                    pos[visible],
                    colors[visible],
                    sel[visible],
                    (tmpl["radii"][visible] * self._atom_radius_scale)[:, None],
                ]
            )
        )

        for a1, a2 in tmpl["bonds"]:
            if a1 < n and a2 < n and visible[a1] and visible[a2]:
                p1, p2 = pos[a1], pos[a2]
                mid = (p1 + p2) * 0.5
                bond_blocks.append(np.concatenate([p1, mid, colors[a1], [self._bond_radius]]))
                bond_blocks.append(np.concatenate([mid, p2, colors[a2], [self._bond_radius]]))

        for i, sym in enumerate(tmpl["symbols"]):
            if visible[i]:
                self._atom_label_data.append((pos[i].copy(), sym))
        self._mol_label_data.append(
            ((tmpl["centroid"] + cell_cart - self._scene_centre).copy(), f"M{tile}")
        )

    def _upload_geometry(self):
        """Rebuild and upload all GPU geometry from current data."""
        if self._crystallography is None:
            return

        if self._site_mode:
            self._upload_site_scene()
            return

        cells = self.cells()
        cell_set = set(cells)

        # Centre all geometry on the currently visible molecules so rotation
        # pivots at their centre of geometry; this recomputes on every rebuild
        # (isolate / show-all / supercell expansion), moving the pivot with the
        # visible object.  Falls back to the supercell centre when nothing is
        # visible (cell-edges-only view).
        scene_centre = self._visible_geometry_centre(cells)
        self._scene_centre = scene_centre

        self._cell_renderer.set_lines(self._build_cell_lines(cells))

        atom_blocks: list = []
        bond_blocks: list = []
        self._atom_label_data = []
        self._mol_label_data = []

        conns_active = bool(
            self._connections and self._conn_source_tiles and self._templates
        )

        # Instances highlighted because one of their connections is selected
        highlight: set = set()
        if conns_active:
            sel_conns = {
                (i[1], i[2]) for i in self._selected_ids if i[0] == "conn"
            }
            for tile, k in sel_conns:
                conns = self._connections.get(tile)
                if not conns or k >= len(conns) or tile not in self._conn_source_tiles:
                    continue
                conn = conns[k]
                for cell in cells:
                    if ("mol", tile, cell) in self._hidden_ids:
                        continue
                    tgt_cell = (
                        cell[0] + conn.offset[0],
                        cell[1] + conn.offset[1],
                        cell[2] + conn.offset[2],
                    )
                    highlight.add((tile, cell))
                    highlight.add((conn.target, tgt_cell))

        # --- Supercell molecules ---
        drawn: set = set()  # (tile, cell) instances with geometry emitted
        if self._templates:
            for cell in cells:
                cell_cart = self._cell_cart(cell)
                for tile in self._templates:
                    if ("mol", tile, cell) in self._hidden_ids:
                        continue
                    drawn.add((tile, cell))
                    selected = (
                        ("mol", tile, cell) in self._selected_ids
                        or (tile, cell) in highlight
                    )
                    self._emit_molecule(tile, cell_cart, selected, atom_blocks, bond_blocks)

        # --- Connection lines (and ghost neighbours outside the supercell) ---
        conn_vertices: list = []
        if conns_active:
            group_values = self._conn_group_values()
            unique_vals = sorted({v for v in group_values.values() if v is not None})
            val_color = {
                v: _SHELL_COLORS[i % len(_SHELL_COLORS)]
                for i, v in enumerate(unique_vals)
            }

            # Deduplicate reciprocal connections (1→3 and 3→1 describe the
            # same segment) so selected lines are not z-fought by twins.
            segments: dict = {}  # key → [p1, p2, color, selected, energy]
            ghosts: dict = {}  # (tile, cell) → cell_cart

            for cell in cells:
                cell_cart = self._cell_cart(cell)
                for tile in sorted(self._connections):
                    if tile not in self._conn_source_tiles:
                        continue
                    if ("mol", tile, cell) in self._hidden_ids:
                        continue
                    src_tmpl = self._templates.get(tile)
                    if src_tmpl is None:
                        continue
                    src_centroid = src_tmpl["centroid"] + cell_cart - scene_centre

                    for k, conn in enumerate(self._connections[tile]):
                        cid = ("conn", tile, k)
                        if cid in self._hidden_ids:
                            continue
                        if conn.target == tile and conn.offset == (0, 0, 0):
                            continue  # degenerate self-loop
                        tgt_tmpl = self._templates.get(conn.target)
                        if tgt_tmpl is None:
                            continue
                        tgt_cell = (
                            cell[0] + conn.offset[0],
                            cell[1] + conn.offset[1],
                            cell[2] + conn.offset[2],
                        )
                        tgt_cart = self._cell_cart(tgt_cell)
                        tgt_centroid = tgt_tmpl["centroid"] + tgt_cart - scene_centre

                        selected = cid in self._selected_ids
                        key = tuple(sorted(((tile, cell), (conn.target, tgt_cell))))
                        seg = segments.get(key)
                        if seg is None:
                            color = val_color.get(
                                group_values.get((tile, k)), _SHELL_COLORS[0]
                            )
                            ann = (self._conn_annotations or {}).get((tile, k))
                            energy = ann.get("energy") if ann else None
                            if not isinstance(energy, (int, float)):
                                energy = None
                            segments[key] = [
                                src_centroid, tgt_centroid, color, selected, energy,
                            ]
                        elif selected:
                            seg[3] = True

                        # Ghost neighbour beyond the supercell boundary
                        if (
                            self._show_neighbours
                            and tgt_cell not in cell_set
                            and (conn.target, tgt_cell) not in drawn
                            and ("mol", conn.target, tgt_cell) not in self._hidden_ids
                        ):
                            ghosts[(conn.target, tgt_cell)] = tgt_cart

            for (tile, cell), cell_cart in ghosts.items():
                if (tile, cell) in drawn:
                    continue
                drawn.add((tile, cell))
                self._emit_molecule(
                    tile, cell_cart, (tile, cell) in highlight, atom_blocks, bond_blocks
                )

            # Tube radii scale with |interaction energy| when net energies are
            # loaded; segments without a numeric energy use the minimum radius.
            r_lo, r_hi = 0.06, 0.45
            magnitudes = [
                abs(s[4]) for s in segments.values() if s[4] is not None
            ]
            e_min = min(magnitudes) if magnitudes else 0.0
            e_span = (max(magnitudes) - e_min) if magnitudes else 0.0

            tube_blocks: list = []
            for p1, p2, color, selected, energy in segments.values():
                c = np.clip(color * 1.8, 0.0, 1.0) if selected else color
                conn_vertices.append(np.concatenate([p1, c]))
                conn_vertices.append(np.concatenate([p2, c]))
                if self._conn_tubes:
                    if energy is None:
                        radius = r_lo
                    elif e_span > 0.0:
                        radius = r_lo + (abs(energy) - e_min) / e_span * (r_hi - r_lo)
                    else:
                        radius = (r_lo + r_hi) * 0.5
                    radius *= self._conn_radius_scale
                    tube_blocks.append(np.concatenate([p1, p2, c, [radius]]))

            self._conn_tube_renderer.setBonds(
                np.array(tube_blocks, dtype=np.float32)
                if tube_blocks
                else np.empty((0, 10), dtype=np.float32)
            )

        self._set_atom_geometry(atom_blocks, bond_blocks)
        if not atom_blocks and not self._view_fitted:
            corners = [
                self._crystallography.frac_to_cart(_FRAC_CORNERS).astype(np.float32)
                + self._cell_cart(cell)
                - scene_centre
                for cell in cells
            ]
            self._camera.fitToObject(np.vstack(corners))
            self._view_fitted = True

        if not conns_active:
            self._conn_tube_renderer.setBonds(np.empty((0, 10), dtype=np.float32))
        self._conn_renderer.set_lines(
            np.array(conn_vertices, dtype=np.float32).flatten()
            if conn_vertices
            else np.array([], dtype=np.float32)
        )

    def _set_atom_geometry(self, atom_blocks: list, bond_blocks: list):
        """Upload atoms and bonds, fitting the camera to the atoms when needed."""
        if atom_blocks:
            atom_arr = np.vstack(atom_blocks).astype(np.float32)
            # Atoms sharing a lattice site are merged into one sphere split
            # into coloured sections; the rest draw as ordinary spheres.
            singles, sectioned = self._partition_overlaps(atom_arr)
            self._atom_renderer.setPoints(singles)
            self._multi_atom_renderer.setPoints(sectioned)
            if not self._view_fitted:
                self._camera.fitToObject(atom_arr[:, :3])
                self._view_fitted = True
        else:
            self._atom_renderer.setPoints(np.empty((0, 8), dtype=np.float32))
            self._multi_atom_renderer.setPoints(
                np.empty((0, self._multi_atom_renderer._in_floats), dtype=np.float32)
            )

        self._bond_renderer.setBonds(
            np.array(bond_blocks, dtype=np.float32)
            if bond_blocks
            else np.empty((0, 10), dtype=np.float32)
        )

    def _upload_site_scene(self):
        """Rebuild the GPU geometry for the selected site environment."""
        self._cell_renderer.set_lines(np.array([], dtype=np.float32))
        self._atom_label_data = []
        self._mol_label_data = []

        atom_blocks: list = []
        bond_blocks: list = []
        contact_vertices: list = []
        tube_blocks: list = []

        env = self._site_env
        if env is not None and self._templates:
            # Rotate about the target molecule
            self._scene_centre = self._templates[env.tile]["centroid"]

            def centroid(site: tuple) -> np.ndarray:
                tmpl = self._templates[site[3]]
                return tmpl["centroid"] + self._cell_cart(site[:3]) - self._scene_centre

            molecules = [((0, 0, 0, env.tile), _SITE_TARGET_COLOR)]
            if self._site_show_first:
                molecules += [(site, _SITE_FIRST_COLOR) for site in env.first_shell]
            if self._site_show_second:
                molecules += [(site, _SITE_SECOND_COLOR) for site in env.second_shell]

            for site, color in molecules:
                self._emit_molecule(
                    site[3],
                    self._cell_cart(site[:3]),
                    False,
                    atom_blocks,
                    bond_blocks,
                    color=color if self._site_color_by_shell else None,
                )

            if self._site_show_empty:
                for site in env.empty_first_shell:
                    atom_blocks.append(
                        np.concatenate(
                            [centroid(site), _SITE_EMPTY_COLOR, [0.0, _SITE_EMPTY_RADIUS]]
                        )[None, :]
                    )

            second = set(env.second_shell)
            for start, end in self._site_contacts:
                reaches_second = end in second
                if not self._site_show_first or (
                    reaches_second and not self._site_show_second
                ):
                    continue
                color = _SITE_SECOND_COLOR if reaches_second else _SITE_FIRST_COLOR
                p1, p2 = centroid(start), centroid(end)
                contact_vertices.append(np.concatenate([p1, color]))
                contact_vertices.append(np.concatenate([p2, color]))
                radius = _SITE_CONTACT_RADIUS * self._conn_radius_scale
                tube_blocks.append(np.concatenate([p1, p2, color, [radius]]))

        self._set_atom_geometry(atom_blocks, bond_blocks)
        self._conn_renderer.set_lines(
            np.array(contact_vertices, dtype=np.float32).flatten()
            if contact_vertices
            else np.array([], dtype=np.float32)
        )
        self._conn_tube_renderer.setBonds(
            np.array(tube_blocks, dtype=np.float32)
            if tube_blocks
            else np.empty((0, 10), dtype=np.float32)
        )

    def _partition_overlaps(self, atom_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split atoms into non-overlapping and coincident-site groups.

        ``atom_arr`` has one row per atom: position[3], color[3], selected[1],
        radius[1].  Atoms whose positions coincide (to ``_OVERLAP_DECIMALS``)
        are merged into a single sectioned-sphere instance so the overlap is
        visible as coloured wedges; every other atom stays a plain sphere.

        Returns ``(singles, sectioned)`` — shapes ``(N, 8)`` and ``(M, 18)``.
        """
        n_multi_floats = self._multi_atom_renderer._in_floats
        if len(atom_arr) == 0:
            return atom_arr, np.empty((0, n_multi_floats), dtype=np.float32)

        keys = np.round(atom_arr[:, :3], _OVERLAP_DECIMALS)
        groups: dict = {}
        for i, key in enumerate(map(tuple, keys)):
            groups.setdefault(key, []).append(i)

        singles: list = []
        sectioned: list = []
        for idxs in groups.values():
            if len(idxs) == 1:
                singles.append(atom_arr[idxs[0]])
                continue
            rows = atom_arr[idxs]
            position = rows[:, :3].mean(axis=0)
            radius = rows[:, 7].max()
            selected = rows[:, 6].max()
            count = min(len(idxs), MAX_SECTIONS)
            colors = np.zeros((MAX_SECTIONS, 3), dtype=np.float32)
            for j in range(count):
                colors[j] = rows[j, 3:6]
            sectioned.append(
                np.concatenate(
                    [position, [radius, selected, float(count)], colors.ravel()]
                )
            )

        singles_arr = (
            np.array(singles, dtype=np.float32)
            if singles
            else np.empty((0, 8), dtype=np.float32)
        )
        sectioned_arr = (
            np.array(sectioned, dtype=np.float32)
            if sectioned
            else np.empty((0, n_multi_floats), dtype=np.float32)
        )
        return singles_arr, sectioned_arr

    def _build_cell_lines(self, cells: list) -> np.ndarray:
        """Build edge vertex array for every cell of the supercell."""
        base = self._crystallography.frac_to_cart(_FRAC_CORNERS).astype(np.float32)
        vertices = []
        for cell in cells:
            cart = base + self._cell_cart(cell) - self._scene_centre
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
            "u_pointSize": self.point_size,
            "u_screenSize": screen_size,
            "u_scale": self._camera.scale,
            "u_lineScale": 2.0,
            "u_axesMat": axes,
            **self._render_settings.shader_uniforms(self._camera.perspectiveProjection),
        }

    # ------------------------------------------------------------------
    # Camera keyframes (used by the embedded animation timeline)
    # ------------------------------------------------------------------

    def snapshot(self):
        """Return a CameraSnapshot of the current view (for keyframe capture)."""
        return dataclasses.replace(self._camera.snapshot(), point_size=self.point_size)

    def apply_camera_snapshot(self, snap) -> None:
        """Apply a CameraSnapshot and schedule a repaint (for keyframe preview/render)."""
        self._camera.restore_snapshot(snap)
        self.point_size = snap.point_size
        self.update()

    def render_animation_frame(self, scale: float = 1.0):
        """Render the current view to a QImage (used by the animation render worker)."""
        return self.renderToImage(scale)

    # ------------------------------------------------------------------
    # Image / ray-trace export
    # ------------------------------------------------------------------

    def renderToImage(self, scale: float = 1.0) -> QImage:
        """Render the current view at ``scale`` × the viewport size to an off-screen FBO."""
        self.makeCurrent()
        w = max(1, int(self.width() * scale))
        h = max(1, int(self.height() * scale))
        gl = self.context().functions()
        fbo = QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject.CombinedDepthStencil)

        fbo.bind()
        gl.glViewport(0, 0, w, h)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDisable(GL_BLEND)
        gl.glClearColor(
            self.backgroundColor.redF(), self.backgroundColor.greenF(),
            self.backgroundColor.blueF(), 1.0,
        )
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._dirty:
            self._upload_geometry()
            self._dirty = False
        uniforms = self._build_uniforms()
        self._draw_scene(gl, uniforms)

        fbo.release()
        result = fbo.toImage()
        self.doneCurrent()
        result.reinterpretAsFormat(QImage.Format_RGB32)
        return result

    def saveRender(self, file_name, resolution):
        image = self.renderToImage(float(resolution[0]))
        image.save(file_name)

    def export_image_dialog(self):
        """Save the current view as a PNG at 1x/2x/4x the viewport resolution."""
        options = ["1x", "2x", "4x"]
        resolution, ok = QInputDialog.getItem(
            self, "Select Resolution", "Resolution:", options, 0, False
        )
        if not ok:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png)")
        if not file_name:
            return
        self.saveRender(file_name, resolution)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Image Saved")
        msg_box.setText(f"Image saved to:\n{file_name}")
        msg_box.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Open)
        open_folder_button = msg_box.addButton("Open Folder", QMessageBox.ActionRole)
        result = msg_box.exec_()
        if result == QMessageBox.Open:
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_name))
        elif msg_box.clickedButton() == open_folder_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(file_name).parent)))

    def export_raytrace_dialog(self):
        """Open the POV-Ray / Tachyon ray-trace export dialog for this viewer."""
        from .raytrace_dialog import RaytraceDialog

        RaytraceDialog(self, parent=self).exec()

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
            # Same camera orbit as the Crystal Viewer's default interaction mode.
            self._camera.orbit(dx, dy)
            self.update()

    def keyPressEvent(self, event):
        step = _ARROW_KEY_ROTATION.get(event.key())
        if step is None:
            super().keyPressEvent(event)
            return
        self._camera.orbit(*step)
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self._camera.zoom(delta / 120.0)
        self.update()


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------


class _WrappedLabel(QLabel):
    """Word-wrapped label that reserves the height its text needs at the current width."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setWordWrap(True)

    def setText(self, text: str):
        super().setText(text)
        self.setMinimumHeight(self.heightForWidth(self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setMinimumHeight(self.heightForWidth(self.width()))


class _ShareBarDelegate(QStyledItemDelegate):
    """Draws the fraction stored in a cell's user role as a bar across the cell."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        bar = QRect(option.rect)
        bar.setWidth(max(1, round(option.rect.width() * index.data(Qt.UserRole))))
        painter.fillRect(bar, _SHARE_BAR_COLOR)


def _percent(fraction: float) -> str:
    """Whole-number percentage that does not round a small share down to zero."""
    return "<1%" if fraction < 0.01 else f"{fraction:.0%}"


class TopologyViewerDialog(QDialog):
    """Crystallography → Topology Viewer dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topology Viewer")
        self.setMinimumSize(860, 580)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMaximizeButtonHint
        )

        self._viewer = TopologyViewerWidget()
        self._structure: Structure | None = None
        self._conn_annotations: dict | None = None

        # Site Types mode state
        self._sim_checkpoint_path: Path | None = None  # checkpoint of the open simulation
        self._site_checkpoint_path: Path | None = None  # checkpoint currently in use
        self._site_analysis: SiteEnvironmentAnalysis | None = None
        self._site_environments: list[SiteEnvironment] = []
        self._site_loading_key: tuple | None = None
        self._site_pending: int | None = None

        # --- Supercell controls ---
        supercell_group = QGroupBox("Supercell")
        layout = QHBoxLayout(supercell_group)

        self._supercell_sbs = []

        for axis in ("a", "b", "c"):
            label = QLabel(axis)
            sb = QSpinBox()
            sb.setRange(1, _MAX_SUPERCELL)
            sb.setValue(1)
            sb.setToolTip(f"Number of unit cells along {axis}")
            sb.valueChanged.connect(self._on_supercell_changed)

            layout.addWidget(label)
            layout.addWidget(sb)

            self._supercell_sbs.append(sb)

        layout.addStretch()

        # --- Net file (energy overlay) controls ---
        self._net_label = QLabel("No net file loaded")
        self._net_label.setWordWrap(True)

        self._import_btn = QPushButton("Import Net File…")
        self._import_btn.setToolTip(
            "Overlay interaction distances and energies from a CrystoGen net file"
        )
        self._import_btn.clicked.connect(self._on_import_net)

        self._clear_net_btn = QPushButton("Clear Net")
        self._clear_net_btn.setEnabled(False)
        self._clear_net_btn.clicked.connect(self._on_clear_net)

        # Checkboxes selecting which tiles fan out connection lines;
        # populated from the structure file's net connectivity.
        self._source_checkboxes: dict = {}  # tile → QCheckBox
        self._sources_group = QGroupBox("Show Connections From")
        self._sources_layout = QHBoxLayout(self._sources_group)
        self._sources_group.setVisible(False)
        self._sources_layout.addStretch()

        # --- Display toggles ---
        self._show_cell_cb = QCheckBox("Show Unit Cell")
        self._show_cell_cb.setChecked(True)
        self._show_cell_cb.toggled.connect(self._viewer.set_show_cell)

        self._show_mol_cb = QCheckBox("Show Molecules")
        self._show_mol_cb.setChecked(True)
        self._show_mol_cb.setEnabled(False)
        self._show_mol_cb.toggled.connect(self._viewer.set_show_molecules)

        self._show_conn_cb = QCheckBox("Show Connections")
        self._show_conn_cb.setChecked(True)
        self._show_conn_cb.setEnabled(False)
        self._show_conn_cb.toggled.connect(self._viewer.set_show_connections)

        self._show_neigh_cb = QCheckBox("Show Neighbours Outside Supercell")
        self._show_neigh_cb.setChecked(False)
        self._show_neigh_cb.setEnabled(False)
        self._show_neigh_cb.setToolTip(
            "Draw the molecules that connections at the supercell boundary point to"
        )
        self._show_neigh_cb.toggled.connect(self._viewer.set_show_neighbours)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._viewer.reset_view)

        self._status_label = QLabel("Load a simulation folder with a structure file to view molecules.")
        self._status_label.setWordWrap(True)

        # --- Appearance (separate non-modal dialog) ---
        self._appearance_dialog = TopologyAppearanceDialog(self._viewer, parent=self)
        self._appearance_btn = QPushButton("Appearance…")
        self._appearance_btn.clicked.connect(self._show_appearance_dialog)

        self._export_btn = QPushButton("Export Image…")
        self._export_btn.setToolTip("Save the current view as a standard or ray-traced image")
        self._export_btn.clicked.connect(self._on_export_image)

        # --- Animation (camera keyframes + video export) ---
        self._render_worker = None
        self._animation_timeline = AnimationTimeline()
        self._timeline_widget = KeyframeTimelineWidget()
        self._timeline_widget.set_timeline(self._animation_timeline)
        self._timeline_widget.hide()
        self._timeline_widget.keyframeAddRequested.connect(self._add_keyframe)
        self._timeline_widget.previewRequested.connect(self._on_preview_tick)
        self._timeline_widget.renderRequested.connect(self._open_render_dialog)

        self._anim_toggle_btn = QPushButton("Show Animation Timeline")
        self._anim_toggle_btn.setCheckable(True)
        self._anim_toggle_btn.toggled.connect(self._on_toggle_timeline)

        # --- Layout ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.addWidget(self._show_cell_cb)
        display_layout.addWidget(self._show_mol_cb)
        display_layout.addWidget(self._show_conn_cb)
        display_layout.addWidget(self._show_neigh_cb)

        self._import_struct_btn = QPushButton("Import Structure File…")
        self._import_struct_btn.setToolTip(
            "Load molecules and net connectivity from a CrystoGen structure file"
        )
        self._import_struct_btn.clicked.connect(self._on_import_structure)

        self._site_source_label = QLabel("No checkpoint loaded")
        self._site_source_label.setWordWrap(True)
        open_checkpoint_btn = QPushButton("Open Checkpoint…")
        open_checkpoint_btn.setAutoDefault(False)
        open_checkpoint_btn.setToolTip("Read site types from a CrystoGen checkpoint file")
        open_checkpoint_btn.clicked.connect(self._on_open_checkpoint)

        # The net file only applies to Unit Cell mode, the checkpoint to Site Types.
        self._net_file_row = QWidget()
        net_row_layout = QHBoxLayout(self._net_file_row)
        net_row_layout.setContentsMargins(0, 0, 0, 0)
        net_row_layout.addWidget(self._net_label, 1)
        net_row_layout.addWidget(self._import_btn)
        net_row_layout.addWidget(self._clear_net_btn)

        self._checkpoint_file_row = QWidget()
        checkpoint_row_layout = QHBoxLayout(self._checkpoint_file_row)
        checkpoint_row_layout.setContentsMargins(0, 0, 0, 0)
        checkpoint_row_layout.addWidget(self._site_source_label, 1)
        checkpoint_row_layout.addWidget(open_checkpoint_btn)
        self._checkpoint_file_row.setVisible(False)

        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)
        files_layout.addWidget(self._import_struct_btn)
        files_layout.addWidget(self._net_file_row)
        files_layout.addWidget(self._checkpoint_file_row)

        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)
        view_layout.addWidget(self._reset_btn)
        view_layout.addWidget(self._appearance_btn)
        view_layout.addWidget(self._export_btn)
        view_layout.addWidget(self._anim_toggle_btn)

        # --- Selection panel ---
        sel_group = QGroupBox("Molecules / Connections")
        sel_layout = QVBoxLayout(sel_group)

        self._sel_tree = QTreeWidget()
        self._sel_tree.setHeaderHidden(True)
        self._sel_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._sel_tree.setMinimumHeight(160)
        # Let long labels (e.g. annotated connections) scroll horizontally
        # instead of being clipped to the panel width.
        self._sel_tree.setTextElideMode(Qt.ElideNone)
        self._sel_tree.header().setStretchLastSection(False)
        self._sel_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._sel_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._sel_tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        sel_layout.addWidget(self._sel_tree)

        sel_btn_row = QHBoxLayout()
        self._hide_sel_btn = QPushButton("Hide")
        self._hide_sel_btn.setToolTip("Hide the selected items")
        self._hide_sel_btn.clicked.connect(self._on_hide_selected)
        self._isolate_btn = QPushButton("Isolate")
        self._isolate_btn.setToolTip(
            "Hide everything except the selected molecules and their connections"
        )
        self._isolate_btn.clicked.connect(self._on_hide_unselected)
        self._show_all_btn = QPushButton("Show All")
        self._show_all_btn.clicked.connect(self._on_show_all)
        sel_btn_row.addWidget(self._hide_sel_btn)
        sel_btn_row.addWidget(self._isolate_btn)
        sel_btn_row.addWidget(self._show_all_btn)
        sel_layout.addLayout(sel_btn_row)

        self._viewer.itemTreeChanged.connect(self._rebuild_item_list)

        unit_cell_page = QWidget()
        unit_cell_layout = QVBoxLayout(unit_cell_page)
        unit_cell_layout.setContentsMargins(0, 0, 0, 0)
        unit_cell_layout.addWidget(supercell_group)
        unit_cell_layout.addWidget(display_group)
        unit_cell_layout.addWidget(self._sources_group)
        unit_cell_layout.addWidget(sel_group)
        unit_cell_layout.addStretch()

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._make_scrollable(unit_cell_page))
        self._mode_stack.addWidget(self._make_scrollable(self._build_site_page()))

        self._mode_tabs = QTabBar()
        self._mode_tabs.addTab("Unit Cell")
        self._mode_tabs.addTab("Site Types")
        self._mode_tabs.currentChanged.connect(self._on_mode_changed)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(files_group)
        ctrl_layout.addWidget(self._mode_tabs)
        ctrl_layout.addWidget(self._mode_stack, 1)
        ctrl_layout.addWidget(view_group)
        ctrl_layout.addWidget(self._status_label)

        content_row = QHBoxLayout()
        content_row.addLayout(ctrl_layout, 0)
        content_row.addWidget(self._viewer, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(content_row, 1)
        main_layout.addWidget(self._timeline_widget)

    def _make_scrollable(self, page: QWidget) -> QScrollArea:
        """Scroll a mode page vertically so short windows never squeeze its controls."""
        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollbar_width = self.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        scroll.setMinimumWidth(page.minimumSizeHint().width() + scrollbar_width)
        return scroll

    def _build_site_page(self) -> QWidget:
        """Build the control panel for the Site Types mode."""
        page = QWidget()
        page.setMinimumWidth(360)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Site type choice ---
        type_group = QGroupBox("Site Type")
        type_layout = QVBoxLayout(type_group)
        spin_row = QHBoxLayout()
        self._site_spin = QSpinBox()
        self._site_spin.setEnabled(False)
        self._site_spin.setToolTip("Site number to build")
        self._site_build_btn = QPushButton("Build")
        self._site_build_btn.setDefault(True)
        self._site_build_btn.setEnabled(False)
        self._site_build_btn.setToolTip("List every environment of this site type")
        self._site_build_btn.clicked.connect(self._on_build_site)
        spin_row.addWidget(QLabel("Site number"))
        spin_row.addWidget(self._site_spin, 1)
        spin_row.addWidget(self._site_build_btn)
        type_layout.addLayout(spin_row)

        self._site_type_table = self._make_site_table(["Site", "State", "Sites", "Tiles"])
        self._site_type_table.itemSelectionChanged.connect(self._on_site_type_selected)
        type_layout.addWidget(self._site_type_table)

        # --- Environments of the chosen type ---
        env_group = QGroupBox("Environments")
        env_layout = QVBoxLayout(env_group)
        self._env_summary_label = _WrappedLabel("Choose a site type to list its environments.")
        env_layout.addWidget(self._env_summary_label)
        self._env_table = self._make_site_table(
            ["#", "Pattern", "Tile", "Sites", "Share", "1st", "2nd"]
        )
        self._env_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._env_table.horizontalHeader().setSectionResizeMode(
            _ENV_SHARE_COLUMN, QHeaderView.Stretch
        )
        self._env_table.setItemDelegateForColumn(
            _ENV_SHARE_COLUMN, _ShareBarDelegate(self._env_table)
        )
        self._env_table.itemSelectionChanged.connect(self._on_environment_selected)
        env_layout.addWidget(self._env_table)

        self._site_export_btn = QPushButton("Export Docking Site…")
        self._site_export_btn.setAutoDefault(False)
        self._site_export_btn.setEnabled(False)
        self._site_export_btn.setToolTip("Save the selected environment as docking XYZ or PDB")
        self._site_export_btn.clicked.connect(self._on_export_docking_site)
        env_layout.addWidget(self._site_export_btn)

        # --- Display of the selected environment ---
        site_display_group = QGroupBox("Display")
        site_display_layout = QGridLayout(site_display_group)
        legend = QLabel(
            '<span style="color:#ff5900">●</span> Target&nbsp;&nbsp;'
            '<span style="color:#00d933">●</span> 1st shell&nbsp;&nbsp;'
            '<span style="color:#2673ff">●</span> 2nd shell&nbsp;&nbsp;'
            '<span style="color:#8c8c8c">●</span> Empty position'
        )
        site_display_layout.addWidget(legend, 0, 0, 1, 2)

        self._site_first_cb = QCheckBox("Show 1st Shell")
        self._site_second_cb = QCheckBox("Show 2nd Shell")
        self._site_empty_cb = QCheckBox("Show Empty Neighbour Positions")
        self._site_contacts_cb = QCheckBox("Show Contacts")
        self._site_shell_color_cb = QCheckBox("Colour by Shell")
        for index, checkbox in enumerate(
            (
                self._site_first_cb,
                self._site_second_cb,
                self._site_contacts_cb,
                self._site_shell_color_cb,
                self._site_empty_cb,
            )
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_site_display_changed)
            site_display_layout.addWidget(checkbox, 1 + index // 2, index % 2)

        layout.addWidget(type_group, 2)
        layout.addWidget(env_group, 3)
        layout.addWidget(site_display_group)
        return page

    @staticmethod
    def _make_site_table(headers: list[str]) -> QTableWidget:
        """Create a read-only, single-row-selection table."""
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setMinimumHeight(110)
        return table

    # ------------------------------------------------------------------
    # Public API (called from MainWindow)
    # ------------------------------------------------------------------

    def set_crystallography(self, crystallography: Crystallography | None):
        """Update the unit cell box.  Call whenever lattice parameters change."""
        self._viewer.set_crystallography(crystallography)

    def set_structure(self, structure: Structure | None, crystallography: Crystallography | None = None):
        """Update molecule templates and net connectivity from the structure file."""
        if crystallography is not None:
            self._viewer.set_crystallography(crystallography)

        structure_changed = structure is not self._structure
        self._structure = structure
        if structure_changed:
            self._reset_site_analysis()
        self._conn_annotations = None
        self._viewer.set_conn_annotations(None)
        self._net_label.setText("No net file loaded")
        self._clear_net_btn.setEnabled(False)

        has_connections = bool(structure and structure.connections)

        self._viewer.set_structure(structure)
        self._show_mol_cb.setEnabled(bool(structure and structure.templates))
        self._show_conn_cb.setEnabled(has_connections)
        self._show_neigh_cb.setEnabled(has_connections)
        self._appearance_dialog.set_available(has_connections)
        self._populate_source_checkboxes(
            sorted(structure.connections) if has_connections else []
        )

        if self._mode_tabs.currentIndex() == _MODE_SITE_TYPES:
            self._ensure_site_analysis()
        else:
            self._show_structure_status()

    def _show_structure_status(self):
        """Describe the loaded structure in the status line."""
        structure = self._structure
        has_templates = bool(structure and structure.templates)
        has_connections = bool(structure and structure.connections)
        if has_templates and has_connections:
            n_conn = sum(len(c) for c in structure.connections.values())
            self._status_label.setText(
                f"Loaded {len(structure.templates)} molecule(s) with "
                f"{n_conn} net connections."
            )
        elif has_templates:
            self._status_label.setText(
                f"Loaded {len(structure.templates)} molecule template(s); "
                "no net connectivity found in the structure file."
            )
        else:
            self._status_label.setText(
                "No molecule templates found — showing cell edges only."
            )

    def set_simulation_checkpoint(self, path: Path | None):
        """Use the open simulation's checkpoint for the Site Types mode.

        A checkpoint opened by hand stays in use until the simulation changes.
        """
        if path == self._sim_checkpoint_path:
            return
        self._sim_checkpoint_path = path
        self._use_site_checkpoint(path)

    # ------------------------------------------------------------------
    # Site Types mode
    # ------------------------------------------------------------------

    def _on_mode_changed(self, index: int):
        self._mode_stack.setCurrentIndex(index)
        self._net_file_row.setVisible(index != _MODE_SITE_TYPES)
        self._checkpoint_file_row.setVisible(index == _MODE_SITE_TYPES)
        self._viewer.set_site_mode(index == _MODE_SITE_TYPES)
        if index != _MODE_SITE_TYPES:
            self._show_structure_status()
            return
        self._ensure_site_analysis()
        if self._site_analysis is not None:
            self._show_site_status()

    def _show_site_status(self):
        """Describe the selected environment, or prompt for a site type."""
        analysis = self._site_analysis
        if self._env_table.selectedItems():
            env = self._selected_environment()
            message = (
                f"Site type {env.site}, environment #{env.rank}: {len(env.first_shell)} in the "
                f"1st shell, {len(env.second_shell)} in the 2nd."
            )
        else:
            message = (
                f"{self._site_type_table.rowCount()} site types found. "
                "Choose one to build its environments."
            )
        if not analysis.populations_match:
            message = (
                "Warning: site counts differ from the checkpoint header, so this checkpoint "
                f"may not match the structure file. {message}"
            )
        self._status_label.setText(message)

    def _on_open_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Checkpoint File", "", "Checkpoint Files (*.txt);;All Files (*)"
        )
        if path:
            self._use_site_checkpoint(Path(path))

    def _use_site_checkpoint(self, path: Path | None):
        self._site_checkpoint_path = path
        self._reset_site_analysis()
        if self._mode_tabs.currentIndex() == _MODE_SITE_TYPES:
            self._ensure_site_analysis()

    def _reset_site_analysis(self):
        """Forget the loaded checkpoint and clear the Site Types panel."""
        self._site_analysis = None
        self._site_loading_key = None
        self._site_pending = None
        path = self._site_checkpoint_path
        self._site_source_label.setText(path.name if path else "No checkpoint loaded")
        self._site_spin.setEnabled(False)
        self._site_build_btn.setEnabled(False)
        self._site_type_table.setRowCount(0)
        self._show_environments([], "Choose a site type to list its environments.")

    def _ensure_site_analysis(self):
        """Start reading the current checkpoint unless it is loaded or loading."""
        path = self._site_checkpoint_path
        if path is None:
            self._status_label.setText(
                "No checkpoint for this simulation. Use Open Checkpoint… to choose one."
            )
            return

        structure = self._structure
        if not (structure and structure.templates and structure.connections and structure.cryst):
            self._status_label.setText(
                "Site types need a structure file with molecules and net connectivity."
            )
            return

        if self._site_analysis is not None or self._is_site_loading(path, structure):
            return

        self._site_loading_key = (path, structure)
        self._site_source_label.setText(f"Reading {path.name}…")
        worker = WorkerSiteEnvironmentLoad(path, structure)
        worker.signals.result.connect(self._on_site_analysis_loaded)
        worker.signals.error.connect(self._on_site_worker_failed)
        QThreadPool.globalInstance().start(worker)

    def _is_site_loading(self, path: Path, structure: Structure) -> bool:
        key = self._site_loading_key
        return key is not None and key[0] == path and key[1] is structure

    def _on_site_worker_failed(self, error: tuple):
        logger.error("Site type analysis failed: %s", error[2])
        self._site_loading_key = None
        self._site_pending = None
        if self._site_checkpoint_path is not None:
            self._site_source_label.setText(self._site_checkpoint_path.name)
        self._status_label.setText(f"Could not read site types: {error[2]}")

    def _on_site_analysis_loaded(self, analysis: SiteEnvironmentAnalysis):
        # Ignore results for a checkpoint or structure that has since been replaced
        if not self._is_site_loading(analysis.checkpoint_path, analysis.structure):
            return
        self._site_loading_key = None
        self._site_analysis = analysis
        self._site_source_label.setText(analysis.checkpoint_path.name)

        summaries = analysis.site_types()
        self._site_spin.setRange(1, analysis.n_ungrown_types * 2)
        self._site_spin.setValue(summaries[0].site)
        self._site_spin.setEnabled(True)
        self._site_build_btn.setEnabled(True)

        table = self._site_type_table
        table.blockSignals(True)
        table.setRowCount(len(summaries))
        for row, summary in enumerate(summaries):
            values = (
                summary.site,
                "Grown" if summary.grown else "Growth site",
                f"{summary.count:,}",
                ", ".join(f"M{tile}" for tile in summary.tiles),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, summary.site)
                table.setItem(row, column, item)
        table.blockSignals(False)

        self._show_site_status()

    def _on_site_type_selected(self):
        items = self._site_type_table.selectedItems()
        if items:
            self._site_spin.setValue(items[0].data(Qt.UserRole))
            self._on_build_site()

    def _on_build_site(self):
        analysis = self._site_analysis
        site = self._site_spin.value()
        self._site_pending = site

        table = self._site_type_table
        table.blockSignals(True)
        table.clearSelection()
        for row in range(table.rowCount()):
            if table.item(row, 0).data(Qt.UserRole) == site:
                table.selectRow(row)
                table.scrollToItem(table.item(row, 0))
                break
        table.blockSignals(False)

        self._status_label.setText(f"Building site type {site}…")
        worker = WorkerSiteEnvironmentBuild(analysis, site)
        worker.signals.result.connect(self._on_site_built)
        worker.signals.error.connect(self._on_site_worker_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_site_built(self, result: tuple):
        analysis, site, environments = result
        if analysis is not self._site_analysis or site != self._site_pending:
            return
        self._site_pending = None

        if not environments:
            self._show_environments([], f"Site type {site} is not present in this checkpoint.")
            self._status_label.setText(f"Site type {site} has no sites to show.")
            return

        self._show_environments(environments, self._environment_summary(environments))

    @staticmethod
    def _environment_summary(environments: list[SiteEnvironment]) -> str:
        """Describe how a site type's instances spread over its environments."""
        total = sum(env.count for env in environments)
        if len(environments) == 1:
            return f"1 environment across {total:,} sites."

        patterns = {env.pattern: env.pattern_fraction for env in environments}
        shown = ", ".join(f"P{p} {_percent(patterns[p])}" for p in sorted(patterns)[:4])
        if len(patterns) > 4:
            shown += ", …"

        top = environments[0]
        tied = sum(env.count == top.count for env in environments)
        most_common = (
            f"#{top.rank} is the most common with {top.count:,} sites ({top.fraction:.1%})."
            if tied == 1
            else f"{tied} are tied as most common with {top.count:,} sites each "
            f"({top.fraction:.1%})."
        )
        return (
            f"{len(environments):,} environments across {total:,} sites, in "
            f"{len(patterns)} first-shell patterns ({shown}). {most_common}"
        )

    def _show_environments(self, environments: list[SiteEnvironment], summary: str):
        """Fill the environment table and select the most common entry."""
        self._site_environments = environments
        self._env_summary_label.setText(summary)

        table = self._env_table
        table.blockSignals(True)
        table.clearSelection()
        table.setRowCount(len(environments))
        top_count = environments[0].count if environments else 0
        bold = QFont()
        bold.setBold(True)
        for row, env in enumerate(environments):
            is_top = env.count == top_count
            values = (
                f"★ {env.rank}" if is_top else str(env.rank),
                f"P{env.pattern}",
                f"M{env.tile}",
                f"{env.count:,}",
                f"{env.fraction:.1%}",
                len(env.first_shell),
                len(env.second_shell),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if is_top:
                    item.setFont(bold)
                if column == _ENV_SHARE_COLUMN:
                    item.setData(Qt.UserRole, env.fraction)
                table.setItem(row, column, item)
        table.blockSignals(False)

        if environments:
            table.selectRow(0)
        else:
            self._site_export_btn.setEnabled(False)
            self._viewer.set_site_environment(None, [])

    def _selected_environment(self) -> SiteEnvironment:
        return self._site_environments[self._env_table.selectedItems()[0].row()]

    def _on_environment_selected(self):
        if not self._env_table.selectedItems():
            return
        env = self._selected_environment()
        self._viewer.set_site_environment(env, self._site_analysis.contacts(env))
        self._site_export_btn.setEnabled(True)
        self._show_site_status()

    def _on_site_display_changed(self):
        self._viewer.set_site_display(
            self._site_first_cb.isChecked(),
            self._site_second_cb.isChecked(),
            self._site_empty_cb.isChecked(),
            self._site_contacts_cb.isChecked(),
            self._site_shell_color_cb.isChecked(),
        )

    def _on_export_docking_site(self):
        env = self._selected_environment()
        analysis = self._site_analysis
        source = analysis.checkpoint_path
        root = source.stem.removesuffix("_checkpoint")
        default = source.parent / f"{root}_site{env.site}_env{env.rank}_docking.XYZ"

        xyz_filter = "Docking XYZ (*.XYZ)"
        pdb_filter = "PDB (*.pdb)"
        chosen, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Docking Site", str(default), f"{xyz_filter};;{pdb_filter}"
        )
        if not chosen:
            return

        as_pdb = selected_filter == pdb_filter
        path = Path(chosen).with_suffix(".pdb" if as_pdb else ".XYZ")
        try:
            if as_pdb:
                cryst = analysis.structure.cryst
                write_docking_pdb(
                    path,
                    analysis.docking_data(env, path),
                    self._viewer._templates,
                    a_axis=cryst.cell.a,
                    crystallography=cryst,
                )
            else:
                analysis.write_docking_xyz(path, env)
        except (OSError, ValueError) as exc:
            logger.exception("Failed to export docking site: %s", exc)
            QMessageBox.critical(self, "Export Docking Site", str(exc))
            return
        self._status_label.setText(f"Docking site exported: {path.name}")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_supercell_changed(self):
        na, nb, nc = (sb.value() for sb in self._supercell_sbs)
        self._viewer.set_supercell(na, nb, nc)

    def _show_appearance_dialog(self):
        self._appearance_dialog.show()
        self._appearance_dialog.raise_()
        self._appearance_dialog.activateWindow()

    def _on_export_image(self):
        """Ask for the image format, then hand over to that format's export."""
        standard = "Standard Image (PNG)"
        raytraced = "Ray-Traced Image (POV-Ray/Tachyon)"
        image_format, ok = QInputDialog.getItem(
            self, "Export Image", "Format:", [standard, raytraced], 0, False
        )
        if not ok:
            return
        if image_format == standard:
            self._viewer.export_image_dialog()
        else:
            self._viewer.export_raytrace_dialog()

    # ------------------------------------------------------------------
    # Animation: camera keyframes + video export
    # ------------------------------------------------------------------

    def _on_toggle_timeline(self, checked: bool):
        self._timeline_widget.setVisible(checked)

    def _add_keyframe(self):
        """Capture the current camera view as a keyframe."""
        snap = self._viewer.snapshot()
        tl = self._animation_timeline
        t = tl.keyframes[-1].time + 1.0 if tl.keyframes else 0.0
        kf = Keyframe(time=t, camera=snap, data_frame=None)
        tl.add_keyframe(kf)
        self._timeline_widget.refresh()
        if not self._timeline_widget.isVisible():
            self._timeline_widget.show()
            self._anim_toggle_btn.setChecked(True)

    def _on_preview_tick(self, t: float):
        """Apply the interpolated camera at time t (timeline preview/scrub)."""
        tl = self._animation_timeline
        if len(tl.keyframes) < 2:
            return
        try:
            snapshot, _ = tl.get_state_at_time(t)
        except ValueError:
            return
        self._viewer.apply_camera_snapshot(snapshot)

    def _open_render_dialog(self):
        """Open the render-to-video dialog for the current camera keyframes."""
        from ..animation.render_dialog import RenderAnimationDialog

        if len(self._animation_timeline.keyframes) < 2:
            QMessageBox.information(
                self,
                "No Animation",
                "Add at least 2 keyframes before rendering.\n"
                "Use the Animation Timeline panel's “+ Add Keyframe” button to "
                "capture the current view.",
            )
            return
        dlg = RenderAnimationDialog(
            timeline=self._animation_timeline,
            viewport_width=self._viewer.width(),
            viewport_height=self._viewer.height(),
            parent=self,
        )
        dlg.renderStarted.connect(self._on_render_started)
        dlg.exec()

    def _on_render_started(self, worker):
        """Bridge: connect the render worker's frameRequested to the main-thread slot."""
        self._timeline_widget.stop_preview()
        self._render_worker = worker
        worker.frameRequested.connect(self._on_render_frame_requested)

    def _on_render_frame_requested(self, frame_idx: int, snapshot, data_frame):
        """Main-thread slot: render one frame and return the QImage to the worker."""
        self._viewer.apply_camera_snapshot(snapshot)

        worker = self._render_worker
        backend = getattr(worker, "raytrace_backend", None) if worker else None
        if backend:
            img = self._raytrace_animation_frame(frame_idx, worker)
        else:
            img = self._viewer.render_animation_frame()
        if worker is not None:
            worker.frame_ready(img)

    def _raytrace_animation_frame(self, frame_idx: int, worker):
        """Render one animation frame via POV-Ray/Tachyon, reusing a scratch dir."""
        from PySide6.QtGui import QImage

        from ..visualisation import raytrace_export as rt

        w, h = worker.resolution
        scene = rt.build_scene_from_widget(self._viewer, w, h, photoreal=worker.photoreal)
        if not len(scene.spheres) and not len(scene.cylinders):
            return QImage()

        scratch = getattr(worker, "_rt_scratch", None)
        if scratch is None:
            import tempfile

            scratch = Path(tempfile.mkdtemp(prefix="cga_ucv_raytrace_"))
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

    def _on_import_structure(self):
        """Load a structure file directly when one wasn't auto-loaded from a folder."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure Files (*.txt *.dat);;All Files (*)",
        )
        if not path:
            return
        try:
            structure = Structure.from_file(path)
        except Exception as exc:
            logger.exception("Failed to load structure file: %s", exc)
            QMessageBox.warning(self, "Error", f"Failed to load structure file:\n{exc}")
            return

        if structure.cryst is None and not structure.templates:
            QMessageBox.warning(
                self,
                "Invalid Structure",
                "No lattice parameters or molecule templates could be parsed "
                "from that file.",
            )
            return

        self.set_structure(structure, structure.cryst)

    def _on_import_net(self):
        if not (self._structure and self._structure.connections):
            QMessageBox.information(
                self,
                "No Connectivity",
                "Load a structure file with net connectivity before importing "
                "a net file — the net file only supplies energies.",
            )
            return

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

        annotations = self._match_net_to_connections(net)
        if annotations is None:
            QMessageBox.warning(
                self,
                "Net Mismatch",
                "Could not match the net file's interactions to the structure "
                "file's connectivity (molecule or interaction counts differ).",
            )
            return

        self._conn_annotations = annotations
        self._viewer.set_conn_annotations(annotations)
        self._net_label.setText(Path(path).name)
        self._clear_net_btn.setEnabled(True)
        n_energies = sum(
            1 for a in annotations.values() if isinstance(a.get("energy"), (int, float))
        )
        self._status_label.setText(
            f"Net energies matched to {len(annotations)} connections "
            f"({n_energies} with numeric energies)."
        )
        self._rebuild_item_list()

    def _match_net_to_connections(self, net: CGNet) -> dict | None:
        """Map net-file interactions onto structure connections.

        Tries a one-to-one mapping of net molecules to tiles in order first,
        then falls back to matching net labels against tile formulas (all
        symmetry-equivalent tiles of a type share the same interaction list).
        Returns {(tile, conn_idx): {"r": ..., "energy": ...}} or None.
        """
        connections = self._structure.connections
        tiles = sorted(connections)
        annotations: dict = {}

        def add(tile: int, molecule) -> bool:
            if molecule.n_interactions != len(connections[tile]):
                return False
            for k, intr in enumerate(molecule.interactions):
                annotations[(tile, k)] = {"r": intr.r, "energy": intr.energy}
            return True

        if len(net.molecules) == len(tiles):
            if all(add(t, m) for t, m in zip(tiles, net.molecules)):
                return annotations
            annotations = {}

        by_label = {m.label: m for m in net.molecules}
        templates = self._structure.templates
        for tile in tiles:
            tmpl = templates.get(tile)
            molecule = by_label.get(tmpl.formula) if tmpl is not None else None
            if molecule is None or not add(tile, molecule):
                return None
        return annotations

    def _on_clear_net(self):
        self._conn_annotations = None
        self._viewer.set_conn_annotations(None)
        self._net_label.setText("No net file loaded")
        self._clear_net_btn.setEnabled(False)
        self._status_label.setText("Net energies cleared.")
        self._rebuild_item_list()

    def _populate_source_checkboxes(self, tiles):
        """Build one checkbox per tile that has net connectivity."""
        self._clear_source_checkboxes()
        for tile in tiles:
            cb = QCheckBox(f"M{tile}")
            cb.setChecked(True)
            cb.toggled.connect(self._on_source_changed)
            self._source_checkboxes[tile] = cb
            self._sources_layout.addWidget(cb)
        self._sources_group.setVisible(bool(tiles))
        self._viewer.set_conn_source_tiles(set(tiles))

    def _clear_source_checkboxes(self):
        for cb in self._source_checkboxes.values():
            self._sources_layout.removeWidget(cb)
            cb.deleteLater()
        self._source_checkboxes.clear()
        self._sources_group.setVisible(False)

    def _on_source_changed(self):
        checked = {t for t, cb in self._source_checkboxes.items() if cb.isChecked()}
        self._viewer.set_conn_source_tiles(checked)

    # ------------------------------------------------------------------
    # Selection panel
    # ------------------------------------------------------------------

    def _conn_item_label(self, tile: int, k: int, conn: TileConnection) -> str:
        dx, dy, dz = conn.offset
        label = f"M{tile} → M{conn.target}  ({dx},{dy},{dz})"
        ann = (self._conn_annotations or {}).get((tile, k))
        if ann:
            if ann.get("r") is not None:
                label += f"  r={ann['r']:.2f} Å"
            energy = ann.get("energy")
            if isinstance(energy, (int, float)):
                label += f"  E={energy:.3f}"
            elif energy:
                label += f"  E={energy}"
        return label

    def _rebuild_item_list(self):
        """Repopulate the tree from whatever is currently loaded in the viewer."""
        self._sel_tree.blockSignals(True)
        self._sel_tree.clear()

        templates = self._viewer._templates
        cells = self._viewer.cells()
        if templates:
            mol_root = QTreeWidgetItem(self._sel_tree, ["Molecules"])
            mol_root.setFlags(mol_root.flags() & ~Qt.ItemIsSelectable)
            for tile, tmpl in sorted(templates.items()):
                tile_item = QTreeWidgetItem(mol_root, [f"M{tile}"])
                tile_item.setData(0, Qt.UserRole, ("moltype", tile))

                if len(cells) > 1:
                    for cell in cells:
                        cell_item = QTreeWidgetItem(
                            tile_item, [f"cell ({cell[0]},{cell[1]},{cell[2]})"]
                        )
                        cell_item.setData(0, Qt.UserRole, ("mol", tile, cell))

                atoms_item = QTreeWidgetItem(tile_item, ["Atoms"])
                atoms_item.setFlags(atoms_item.flags() & ~Qt.ItemIsSelectable)
                for i, sym in enumerate(tmpl["symbols"]):
                    atom_item = QTreeWidgetItem(atoms_item, [f"{sym}  (atom {i})"])
                    atom_item.setData(0, Qt.UserRole, ("atom", tile, i))
            mol_root.setExpanded(True)

        connections = self._viewer._connections
        if connections and templates:
            conn_root = QTreeWidgetItem(self._sel_tree, ["Connections"])
            conn_root.setFlags(conn_root.flags() & ~Qt.ItemIsSelectable)
            for tile in sorted(connections):
                for k, conn in enumerate(connections[tile]):
                    item = QTreeWidgetItem(
                        conn_root, [self._conn_item_label(tile, k, conn)]
                    )
                    item.setData(0, Qt.UserRole, ("conn", tile, k))
            conn_root.setExpanded(False)

        self._sel_tree.blockSignals(False)
        self._update_tree_hidden_state()

    def _expand_selection(self, ids: set) -> set:
        """Expand tile-level ("moltype") IDs to all their cell instances."""
        expanded: set = set()
        cells = self._viewer.cells()
        for item_id in ids:
            if item_id[0] == "moltype":
                expanded |= {("mol", item_id[1], cell) for cell in cells}
            else:
                expanded.add(item_id)
        return expanded

    def _on_tree_selection_changed(self):
        ids: set = set()
        for item in self._sel_tree.selectedItems():
            item_id = item.data(0, Qt.UserRole)
            if item_id is not None:
                ids.add(item_id)
        self._viewer.set_selected_ids(self._expand_selection(ids))

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
        cells = self._viewer.cells()
        dim = QBrush(QColor(100, 100, 100))
        normal = QBrush(QColor(220, 220, 220))

        def _is_hidden(item_id) -> bool:
            if item_id is None:
                return False
            if item_id[0] == "moltype":
                return all(("mol", item_id[1], cell) in hidden for cell in cells)
            return item_id in hidden

        def _apply(item: QTreeWidgetItem):
            item.setForeground(0, dim if _is_hidden(item.data(0, Qt.UserRole)) else normal)
            for i in range(item.childCount()):
                _apply(item.child(i))

        root = self._sel_tree.invisibleRootItem()
        for i in range(root.childCount()):
            _apply(root.child(i))


# ---------------------------------------------------------------------------
# Standalone entry point (``cgaspects-top``)
# ---------------------------------------------------------------------------


def main():
    """Launch the Topology Viewer as a standalone window (``cgaspects-top``)."""
    import argparse
    import sys

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    parser = argparse.ArgumentParser(
        description="Open the CGAspects Topology Viewer."
    )
    parser.add_argument(
        "structure",
        nargs="?",
        help="Optional CrystoGen structure file to load on startup.",
    )
    args = parser.parse_args()

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CGAspects Topology Viewer")

    dialog = TopologyViewerDialog()
    if args.structure:
        structure = Structure.from_file(args.structure)
        dialog.set_structure(structure, structure.cryst)
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

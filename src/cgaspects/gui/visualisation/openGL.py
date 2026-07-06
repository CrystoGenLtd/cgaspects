import logging
from pathlib import Path
import dataclasses

import numpy as np
import trimesh
from matplotlib import cm
from OpenGL.GL import GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST
from PySide6 import QtCore
from PySide6.QtCore import QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QQuaternion, QVector3D
from PySide6.QtOpenGL import QOpenGLDebugLogger, QOpenGLFramebufferObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from scipy.spatial import ConvexHull

from ...fileio.cg_checkpoint import Checkpoint
from ...fileio.xyz_file import CrystalCloud
from ..widgets.overlay_widget import TransparentOverlay
from .atom_renderer import AtomRenderer
from .axes_renderer import AxesRenderer
from .bond_renderer import BondRenderer
from .camera import Camera
from .direction_renderer import DirectionRenderer
from .line_renderer import LineRenderer
from .mesh_renderer import MeshRenderer
from .plane_renderer import PlaneRenderer
from .point_cloud_renderer import SimplePointRenderer
from .shading import RenderSettings
from .sphere_renderer import SphereRenderer
from .sphere_selection_renderer import SphereSelectionRenderer
from .visual_data import VisualData

logger = logging.getLogger("CA:OpenGL")


class VisualisationWidget(QOpenGLWidget):
    # Data source shown in the viewport. Orthogonal to the atom/centroid toggle
    # and the render option, so any combination of the three is valid.
    VIS_MODES = ("Crystal", "Docking", "Checkpoint")
    # How centroids are drawn when the atom view is off.
    RENDER_OPTIONS = ("Spheres", "Points", "Convex Hull")

    vis_mode = "Crystal"
    render_option = "Spheres"
    show_mesh_edges = False

    # Non-configurable viewport shortcuts shown read-only in the Keyboard Shortcuts dialog.
    # Arrow-key rotation and mouse-based actions can't be QActions so they live here.
    # Axis alignment, rotation lock, view control, and point size are configurable QActions
    # in the View menu and are tracked by ShortcutsManager automatically.
    VIEWPORT_SHORTCUTS: dict[str, list[tuple[str, str]]] = {
        "Rotation": [
            ("Arrow Keys", "Rotate view freely"),
            ("Shift + Arrow Keys", "Rotate on a single axis"),
        ],
        "Selection": [
            ("Ctrl/Cmd + Click", "Select a point"),
            ("Shift + Click", "Anchor sphere selection"),
            ("Shift + Drag", "Adjust sphere selection radius"),
        ],
        "Playback": [("Space", "Play / Pause animation")],
        "Window": [("Escape", "Exit fullscreen mode")],
    }

    # Signals for point interaction
    pointHovered = Signal(object, object)  # (point_index, point_data) or (None, None)
    selectionChanged = Signal(set, object)  # (selected_indices, last_selected_index)
    pointsDeleted = Signal(int)  # Number of points deleted
    pointSizeChanged = Signal(int)  # Emitted when point size changes (integer value)
    legendChanged = Signal(dict)  # Emitted when the colour legend data changes
    # Emitted when the visualisation mode, atom/centroid toggle, or render option changes.
    viewStateChanged = Signal()
    renderedCountChanged = Signal(int, str)  # (count, label) after legend filtering is applied

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lastMousePosition = QtCore.QPoint()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.camera = Camera()
        self.restrict_axis = None
        self.rotation_lock_axis = None  # "x", "y", "z", or None; toggled by menu actions
        self.interaction_mode = "camera"  # "camera" | "object"
        self.geom = self.geometry()
        self.centre = self.geom.center()
        # print(self.geom)
        # print(self.geom.getRect())

        # Point picking and selection
        self.setMouseTracking(True)  # Enable hover detection
        self._hovered_point_index = None
        self._selected_points = set()  # Set of selected point indices
        self._last_selected_index = None  # For shift-click range selection
        self._pick_radius = 0.1  # Picking radius in normalized coordinates
        self._deleted_points = set()  # Set of deleted point indices

        self._raw_planes = []
        self._planes_crystallography = None
        self._raw_directions = []
        self._directions_crystallography = None
        self._directions_max_extent = 1.0

        # Atom / bond view state
        self._mol_templates = None  # dict[int, MolTemplate] from structure file
        self._mol_crystallography = None  # Crystallography used to convert frac → cart
        self.atom_renderer = None
        self.bond_renderer = None

        # Per-element overrides (set via Atom Mode Settings dialog)
        self._atom_color_overrides: dict[str, tuple[float, float, float]] = {}
        self._atom_radius_overrides: dict[str, float] = {}  # absolute VdW radius in Å
        self._bond_radius: float = 0.40  # bond cylinder radius in Å

        # Sphere selection state (Shift + Left click + drag)
        self._sphere_sel_center_world = None  # np.array [x, y, z] — set on press
        self._sphere_sel_anchor_idx = None  # index of the point the sphere starts on
        self._sphere_sel_radius = 0.0
        self._sphere_sel_start_screen = None  # QPoint of the initial click
        self._sphere_sel_active = False  # True once the user starts dragging

        self.xyz_path_list = []
        self.sim_num = 0
        self.point_cloud_renderer = None
        self.sphere_renderer = None
        self.sphere_selection_renderer = None
        self.mesh_renderer = None
        self.axes_renderer = None

        self.crystal = None
        # self.object = 0

        self.colormap = "Viridis"
        self.color_by = "Layer"
        self.single_color = QColor(128, 128, 128)  # Default grey color
        self._legend_info = None

        # Legend-driven point filter: set of *visible* legend keys, or None (show all).
        # Only valid for the (vis_mode, atom_view, color_by) recorded in _legend_filter_sig.
        self._legend_filter: set | None = None
        self._legend_filter_sig: tuple | None = None

        self.viewInitialized = False
        self.point_size = 20.0
        self.point_type = "Point"
        self.backgroundColor = QColor(Qt.white)
        self.render_settings = RenderSettings()

        self.overlay = TransparentOverlay(self)
        self.overlay.setGeometry(self.geometry())
        self.overlay.showIcon()

        self.lattice_parameters = None

        self.availableColormaps = {
            "Viridis": cm.viridis,
            "Cividis": cm.cividis,
            "Plasma": cm.plasma,
            "Inferno": cm.inferno,
            "Magma": cm.magma,
            "Twilight": cm.twilight,
            "HSV": cm.hsv,
        }

        self.columnLabelToIndex = {
            "Atom/Molecule Type": 0,
            "Atom/Molecule Number": 1,
            "Layer": 2,
            "Single Colour": -1,
            # Atom/docking-specific modes (not used in point-cloud path)
            "Atom": -2,
            "Coordination Shell": -3,
            "Atom Type": -4,
            "Site Number": 6,
            "Particle Energy": 7,
        }

        self.availableColumns = {}

        # Site highlighting - support multiple groups
        self.highlight_groups = []  # List of (site_set, color) tuples
        self.background_color_override = None  # Background color for non-highlighted sites

        # Direct per-particle colour override — bypasses both normal coloring and highlight_groups
        self._colour_override: np.ndarray | None = None

        # Docking site data
        self._docking_data = None  # DockingData instance or None
        self._docking_visual_data: VisualData | None = None  # separate from _visual_data
        self._docking_shell_color_overrides: dict[int, tuple[float, float, float]] = {}

        # Checkpoint grid data
        self._checkpoint: Checkpoint | None = None
        self._checkpoint_coords = None  # np.ndarray (N, 3) centred, for sphere mode
        self._checkpoint_center: np.ndarray | None = None
        self._checkpoint_visual_data: VisualData | None = None  # active view (full or edges)
        self._checkpoint_vd_full: VisualData | None = None  # all cells, expanded once
        self._checkpoint_vd_edges: VisualData | None = None  # cached edges-only subset
        # {field_label: {site_number: value}} from the site-analysis workflow
        self._site_metadata: dict[str, dict[int, float]] = {}

        # Unified display data – rebuilt whenever the active data source changes
        self._visual_data: VisualData | None = None

        # Atom (True) vs centroid (False) representation, remembered per mode.
        self._atom_view: dict[str, bool] = {mode: False for mode in self.VIS_MODES}

    @property
    def xyz(self):
        """Centroid positions (N, 3) or None – read-only shim for external callers."""
        return self._visual_data.centroids if self._visual_data else None

    def pass_XYZ(self, xyz):
        self._visual_data = (
            VisualData.from_xyz(xyz, self._mol_templates, self._mol_crystallography)
            if xyz is not None
            else None
        )
        logger.debug("XYZ coordinates passed on OpenGL widget")

    def pass_XYZ_list(self, xyz_path_list):
        self.xyz_path_list = xyz_path_list
        logger.info("XYZ file paths (list) passed to OpenGL widget")

    def get_XYZ_from_list(self, value):
        if self.sim_num != value:
            self.sim_num = value
            path = self.xyz_path_list[value]
            if Path(path).stem.endswith("_checkpoint"):
                # Checkpoint files are loaded via set_checkpoint(); skip here.
                self.showNoDataOverlay()
                return
            self.crystal = CrystalCloud.from_file(path)
            if self.crystal.empty:
                self.showNoDataOverlay()
                return
            self.pass_XYZ(self.crystal.get_raw_frame_coords(0))
            self.initGeometry()
            self.update()

    def showNoDataOverlay(self):
        """Show an overlay message when there are no points to display."""
        self.overlay.setText("No point data available for this simulation")
        self.overlay.setVisible(True)
        self.update()

    def apply_coord_scale(self, crystallography) -> None:
        """Scale xyz centroids to Cartesian Å using the a-axis from crystallography.

        Used by the lattice-parameters dialog when no structure file is present.
        """
        if crystallography is None or crystallography.cell is None:
            return
        self._mol_crystallography = crystallography
        if (
            self._visual_data is not None
            and self._visual_data.source == "xyz"
            and self._visual_data._raw is not None
        ):
            self._visual_data = VisualData.from_xyz(
                self._visual_data._raw, self._mol_templates, crystallography
            )
            self.viewInitialized = False
            self.initGeometry()

    def set_fractional_axes(self, crystallography):
        """Set the axes to fractional coordinates using the provided crystallography object."""
        if self.axes_renderer is not None:
            self.axes_renderer.set_crystallography(crystallography)
            self.update()
            logger.info("Axes set to fractional coordinates")

    def set_cartesian_axes(self):
        """Reset the axes to Cartesian coordinates."""
        if self.axes_renderer is not None:
            self.axes_renderer.set_cartesian()
            self.update()
            logger.info("Axes reset to Cartesian coordinates")

    def set_directions(self, directions, crystallography=None, max_extent=1.0):
        """Set crystallographic directions to render (cached for translation re-apply)."""
        self._raw_directions = list(directions)
        self._directions_crystallography = crystallography
        self._directions_max_extent = max_extent
        self._apply_directions()

    def _apply_directions(self):
        if self.direction_renderer is None:
            return
        extent = (
            self._cart_max_extent()
            if self._visual_data is not None
            else self._directions_max_extent
        )
        self.direction_renderer.set_directions(
            self._raw_directions, self._directions_crystallography, extent
        )
        self.update()

    def set_planes(self, planes, crystallography=None):
        """Set crystallographic planes to render (cached for translation re-apply)."""
        self._raw_planes = list(planes)
        self._planes_crystallography = crystallography
        self._apply_planes()

    def _apply_planes(self):
        if self.plane_renderer is None:
            return
        from dataclasses import replace

        visible = [p for p in self._raw_planes if p.visible]
        if self._visual_data is not None:
            extent = self._cart_max_extent()
            converted = [replace(p, size=p.size_relative * extent) for p in visible]
        else:
            converted = visible
        self.plane_renderer.set_planes(converted, self._planes_crystallography)
        if self._visual_data is not None:
            if self.vis_mode == "Crystal" and self.is_atom_view:
                self._update_atom_view()
            else:
                self.initGeometry()
        self.update()

    def _cart_max_extent(self):
        """Half-range of the crystal in world units (Å after pre-scaling)."""
        if self._visual_data is None or self._visual_data.n_centroids == 0:
            return self._directions_max_extent
        coords = self._visual_data.centroids.astype(np.float64)
        extents = coords.max(axis=0) - coords.min(axis=0)
        return float(extents.max()) / 2.0

    def _frac_max_extent(self):
        """Half-range of the crystal in world units (same as _cart_max_extent after pre-scaling)."""
        return self._cart_max_extent()

    # (shell_id, display_name) pairs — use _shell_name() / _shell_id() helpers
    _SHELL_NAMES: tuple = ((30, "Central"), (31, "1st Shell"), (32, "2nd Shell"))

    _NORMAL_COLOR_BY = (
        "Layer",
        "Atom/Molecule Type",
        "Atom/Molecule Number",
        "Single Colour",
        "Site Number",
        "Particle Energy",
    )
    _ATOM_COLOR_BY = ("Atom",) + _NORMAL_COLOR_BY
    _DOCKING_COLOR_BY = ("Coordination Shell", "Atom Type")
    _DOCKING_ATOM_COLOR_BY = ("Atom", "Coordination Shell", "Atom Type")
    # Site-analysis fields appended dynamically when metadata is loaded.
    _SITE_METADATA_COLOR_BY = ("Coordination", "Energy", "Events/Population")
    _CHECKPOINT_COLOR_BY = ("Single Colour", "Z Layer")
    _CHECKPOINT_ATOM_COLOR_BY = ("Atom", "Tile")

    @property
    def is_atom_view(self) -> bool:
        """True when the current mode is showing atoms rather than centroids."""
        return self._atom_view.get(self.vis_mode, False)

    def _active_visual_data(self) -> VisualData | None:
        """The VisualData backing the current visualisation mode."""
        if self.vis_mode == "Docking":
            return self._docking_visual_data
        if self.vis_mode == "Checkpoint":
            return self._checkpoint_visual_data
        return self._visual_data

    def _apply_view_state(self):
        """Reconcile colour options with the new view state and rebuild the scene."""
        opts, default = self._color_by_options_for_state()
        if self.color_by not in opts:
            self.color_by = default
        self.viewStateChanged.emit()
        self.initGeometry()
        self.update()

    def set_visualisation_mode(self, mode: str) -> bool:
        """Switch the data source (Crystal / Docking / Checkpoint)."""
        if mode not in self.VIS_MODES:
            logger.warning("Unknown visualisation mode: %s", mode)
            return False
        if mode == self.vis_mode:
            return False
        self.vis_mode = mode
        self._apply_view_state()
        return True

    def set_atom_view(self, enabled: bool) -> bool:
        """Switch the current mode between atom and centroid representation."""
        enabled = bool(enabled)
        if enabled == self.is_atom_view:
            return True
        if enabled and self._mol_templates is None:
            QMessageBox.information(
                self,
                "No Molecular Data",
                "No structure file was found.\n"
                "Load a CrystalGrower simulation folder that includes a structure file.",
            )
            return False
        self._atom_view[self.vis_mode] = enabled
        self._apply_view_state()
        return True

    def toggle_atom_view(self):
        """Toggle atom/centroid representation for the current mode. Shift+V shortcut."""
        self.set_atom_view(not self.is_atom_view)

    def set_render_option(self, option: str) -> bool:
        """Set how centroids are drawn (Spheres / Points / Convex Hull)."""
        if option not in self.RENDER_OPTIONS:
            logger.warning("Unknown render option: %s", option)
            return False
        if option == self.render_option:
            return True
        self.render_option = option
        self._apply_view_state()
        return True

    def recentre_view(self):
        """Fit the camera to the current geometry.  Used as a menu action (F)
        and called automatically when entering Docking style."""
        coords = (
            self._visual_data.centroids
            if self._visual_data is not None and self._visual_data.n_centroids > 0
            else None
        )

        if coords is not None and len(coords) > 0:
            self.camera.fitToObject(coords)
            self.update()

    def highlight_sites(self, highlight_groups, background_color=None):
        """Highlight multiple groups of sites with different colors.

        Args:
            highlight_groups: List of (site_numbers, color) tuples
                             Each color is RGB array or list [r, g, b] in range [0, 1]
                             site_numbers can be a single number, list, or set
            background_color: RGB color for non-highlighted sites [r, g, b] in range [0, 1]
                            If None, uses original coloring
        """
        self.highlight_groups = []
        for site_numbers, color in highlight_groups:
            if color is None:
                color = [1.0, 0.0, 0.0]  # Red by default

            # Handle single number, list, or set
            if isinstance(site_numbers, (int, np.integer)):
                site_set = {site_numbers}
            elif isinstance(site_numbers, set):
                site_set = site_numbers
            else:
                site_set = set(site_numbers)

            color_array = np.array(color, dtype=np.float32)
            self.highlight_groups.append((site_set, color_array))

        if background_color is not None:
            self.background_color_override = np.array(background_color, dtype=np.float32)
        else:
            self.background_color_override = None

        total_sites = sum(len(sites) for sites, _ in self.highlight_groups)
        logger.info(
            f"Highlighting {len(self.highlight_groups)} groups with {total_sites} total sites"
        )

        # Re-initialize geometry to apply the highlighting
        self.initGeometry()
        self.update()

    def clear_highlighted_sites(self):
        """Clear all highlighted sites."""
        self.highlight_groups.clear()
        self.background_color_override = None
        logger.info("Cleared all highlighted sites")
        self.initGeometry()
        self.update()

    def set_colour_override(self, colours: np.ndarray | None):
        """Apply a pre-computed per-particle colour array (N×3 float32, range 0–1).

        Clears highlight_groups so the two override paths stay mutually exclusive.
        Pass None to remove the override.
        """
        self._colour_override = colours
        if colours is not None:
            self.highlight_groups.clear()
            self.background_color_override = None
        self.initGeometry()
        self.update()

    def clear_colour_override(self):
        self._colour_override = None
        self.initGeometry()
        self.update()

    def saveRenderDialog(self):
        # First ask user what type of export they want
        export_options = ["2D Image (PNG)", "3D Mesh"]

        # Only allow 3D mesh export if not in Points mode
        if not self.is_atom_view and self.render_option == "Points":
            export_options = ["2D Image (PNG)"]

        export_type, ok = QInputDialog.getItem(
            self, "Select Export Type", "Export as:", export_options, 0, False
        )

        if not ok:
            return

        if export_type == "2D Image (PNG)":
            # Original image export workflow
            options = ["1x", "2x", "4x"]
            resolution, ok = QInputDialog.getItem(
                self, "Select Resolution", "Resolution:", options, 0, False
            )

            if ok:
                file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Images (*.png)")
                if file_name:
                    self.saveRender(file_name, resolution)

                    # Confirmation dialog
                    msgBox = QMessageBox(self)
                    msgBox.setWindowTitle("Render Saved")
                    msgBox.setText(f"Image saved to:\n{file_name}")
                    msgBox.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel)
                    msgBox.setDefaultButton(QMessageBox.Open)

                    open_folder_button = msgBox.addButton("Open Folder", QMessageBox.ActionRole)

                    result = msgBox.exec_()

                    if result == QMessageBox.Open:
                        QDesktopServices.openUrl(QUrl.fromLocalFile(file_name))
                    elif msgBox.clickedButton() == open_folder_button:
                        QDesktopServices.openUrl(QUrl.fromLocalFile(Path(file_name).parent))

        elif export_type == "3D Mesh":
            # 3D mesh export workflow
            self.saveMeshDialog()

    def renderToImage(self, scale):
        self.makeCurrent()
        w = self.width() * scale
        h = self.height() * scale
        gl = self.context().functions()
        gl.glViewport(0, 0, w, h)
        fbo = QOpenGLFramebufferObject(w, h, QOpenGLFramebufferObject.CombinedDepthStencil)

        fbo.bind()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw(gl)
        fbo.release()
        result = fbo.toImage()
        self.doneCurrent()
        return result

    def saveRender(self, file_name, resolution):
        image = self.renderToImage(float(resolution[0]))
        image.save(file_name)

    def snapshot(self):
        """Return a CameraSnapshot including all current viewport render settings."""

        return dataclasses.replace(self.camera.snapshot(), point_size=self.point_size)

    def apply_camera_snapshot(self, snap) -> None:
        """Apply a CameraSnapshot (camera + render settings) and schedule a repaint."""
        self.camera.restore_snapshot(snap)
        self.point_size = snap.point_size
        self.pointSizeChanged.emit(round(snap.point_size))
        self.update()

    def render_animation_frame(self, scale: float = 1.0):
        """Render the current view to a QImage (used by the animation render worker)."""
        return self.renderToImage(scale)

    def saveMeshDialog(self):
        # Ask user for mesh file format
        mesh_formats = [".obj", ".stl", ".ply", ".glb", ".off"]
        file_format, ok = QInputDialog.getItem(
            self, "Select Mesh Format", "Format:", mesh_formats, 0, False
        )

        if not ok:
            return

        # Ask user for mesh resolution (sphere subdivision level)
        resolution_options = {
            "Low (Fast)": 1,
            "Medium (Balanced)": 2,
            "High (Detailed)": 3,
            "Ultra (Slow)": 4,
        }
        resolution_choice, ok = QInputDialog.getItem(
            self,
            "Select Mesh Resolution",
            "Resolution (sphere detail):",
            list(resolution_options.keys()),
            1,  # Default to "Medium"
            False,
        )

        if not ok:
            return

        subdivision_level = resolution_options[resolution_choice]

        # Get file name from user
        filter_str = f"3D Mesh (*{file_format})"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Mesh", "", filter_str)

        if file_name:
            # Ensure file has correct extension
            if not file_name.endswith(file_format):
                file_name += file_format

            try:
                self.saveMesh(file_name, subdivision_level=subdivision_level)

                # Confirmation dialog
                msgBox = QMessageBox(self)
                msgBox.setWindowTitle("Mesh Saved")
                msgBox.setText(f"3D mesh saved to:\n{file_name}")
                msgBox.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel)
                msgBox.setDefaultButton(QMessageBox.Open)

                open_folder_button = msgBox.addButton("Open Folder", QMessageBox.ActionRole)

                result = msgBox.exec_()

                if result == QMessageBox.Open:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_name))
                elif msgBox.clickedButton() == open_folder_button:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(Path(file_name).parent))

            except Exception as e:
                logger.error("Failed to save mesh: %s", e)
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setWindowTitle("Export Failed")
                msgBox.setText(f"Failed to export mesh:\n{str(e)}")
                msgBox.exec_()

    def exportXYZDialog(self):
        """Open dialog to export the current point cloud as an XYZ file."""
        if self._visual_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "No point cloud data loaded to export.",
            )
            return

        # Get active points (excluding deleted ones)
        active_xyz = self.get_active_xyz()
        if active_xyz is None or len(active_xyz) == 0:
            QMessageBox.warning(
                self,
                "No Data",
                "No points available to export (all points may have been deleted).",
            )
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export XYZ File", "", "XYZ Files (*.XYZ);;All Files (*)"
        )

        if file_name:
            if not file_name.upper().endswith(".XYZ"):
                file_name += ".XYZ"

            try:
                self.exportXYZ(file_name, active_xyz)

                # Confirmation dialog
                deleted_count = len(self._deleted_points)
                msg = f"XYZ file saved to:\n{file_name}\n\n"
                msg += f"Points exported: {len(active_xyz)}"
                if deleted_count > 0:
                    msg += f"\nPoints omitted (deleted): {deleted_count}"

                msgBox = QMessageBox(self)
                msgBox.setWindowTitle("XYZ Exported")
                msgBox.setText(msg)
                msgBox.setStandardButtons(QMessageBox.Open | QMessageBox.Ok)
                msgBox.setDefaultButton(QMessageBox.Ok)

                open_folder_button = msgBox.addButton("Open Folder", QMessageBox.ActionRole)

                result = msgBox.exec_()

                if result == QMessageBox.Open:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(file_name))
                elif msgBox.clickedButton() == open_folder_button:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(Path(file_name).parent))

            except Exception as e:
                logger.error("Failed to export XYZ: %s", e)
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export XYZ file:\n{str(e)}",
                )

    def exportXYZ(self, file_name, xyz_data=None):
        """Export point cloud data to an XYZ file.

        Args:
            file_name: Path to save the XYZ file
            xyz_data: Optional XYZ data array. If None, uses active (non-deleted) points.
        """
        if xyz_data is None:
            xyz_data = self.get_active_xyz()

        if xyz_data is None or len(xyz_data) == 0:
            raise ValueError("No point cloud data to export")

        num_points = len(xyz_data)

        with open(file_name, "w") as f:
            # Write header line (number of points)
            f.write(f"{num_points}\n")

            # Write comment line
            comment = f"Exported from CrystalAspects // {num_points}"
            f.write(f"{comment}\n")

            # Write point data
            # Format: type number layer x y z [site] [energy]
            for row in xyz_data:
                if len(row) >= 6:
                    # Basic format: type number layer x y z
                    line = f"{int(row[0])} {int(row[1])} {int(row[2])} {row[3]:.6f} {row[4]:.6f} {row[5]:.6f}"

                    # Add optional columns if present
                    if len(row) > 6:
                        line += f" {int(row[6])}"  # Site number
                    if len(row) > 7:
                        line += f" {row[7]:.6f}"  # Energy

                    f.write(line + "\n")

        logger.info(f"Exported {num_points} points to {file_name}")

    def saveMesh(self, file_name, subdivision_level=2):
        """Export the current visualization as a 3D mesh file

        Args:
            file_name: Path to save the mesh file
            subdivision_level: Level of sphere subdivision detail (1-4, default 2)
        """
        mesh = None

        if self.is_atom_view:
            raise ValueError("Cannot export mesh in atom view")

        if self.render_option == "Convex Hull":
            # Use the existing convex hull mesh (without colors)
            mesh = self.mesh_renderer.mesh
        elif self.render_option == "Spheres" and self.vis_mode == "Crystal":
            # Generate mesh from sphere instances with colors
            mesh = self._generateSphereMesh(subdivision_level=subdivision_level)

        if mesh is None:
            raise ValueError(
                f"Cannot export mesh in '{self.vis_mode} / {self.render_option}' mode"
            )

        # Export using trimesh
        mesh.export(file_name)
        logger.info("Mesh exported to %s with subdivision level %d", file_name, subdivision_level)

    def _generateSphereMesh(self, subdivision_level=2):
        """Generate a combined mesh from all sphere instances with colors

        Args:
            subdivision_level: Level of icosphere subdivision (1-4)
                1 = 42 vertices (low detail, fast)
                2 = 162 vertices (medium detail)
                3 = 642 vertices (high detail)
                4 = 2562 vertices (ultra detail, slow)
        """
        if self.sphere_renderer.numberOfInstances() <= 0:
            raise ValueError("No spheres to export")

        # Get the point cloud data (positions and colors)
        varray = self.updatePointCloudVertices()
        if varray is None or len(varray) == 0:
            raise ValueError("No point cloud data available")

        # Create base sphere mesh
        from trimesh.creation import icosphere

        base_sphere = icosphere(subdivisions=subdivision_level, radius=1.0)

        # Scale the sphere by point size (matching the shader: u_pointSize * 0.2)
        scale_factor = self.point_size * 0.2

        all_vertices = []
        all_faces = []
        all_vertex_colors = []
        vertex_offset = 0

        # For each point, create a transformed sphere with its color
        for point_data in varray:
            position = point_data[:3]
            color = point_data[3:6]

            # Transform sphere vertices to this position
            transformed_vertices = base_sphere.vertices * scale_factor + position

            # Create color array for all vertices of this sphere (same color for all vertices)
            num_vertices = len(base_sphere.vertices)
            vertex_colors = np.tile(color, (num_vertices, 1))

            all_vertices.append(transformed_vertices)
            all_faces.append(base_sphere.faces + vertex_offset)
            all_vertex_colors.append(vertex_colors)
            vertex_offset += num_vertices

        # Combine all meshes
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        combined_colors = np.vstack(all_vertex_colors)

        # Convert colors from [0,1] float to [0,255] uint8 for better compatibility
        combined_colors_uint8 = (combined_colors * 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=combined_vertices, faces=combined_faces, vertex_colors=combined_colors_uint8
        )
        return mesh

    def set_render_settings(self, settings: RenderSettings):
        """Apply material / lighting / ambient-occlusion settings from the dialog.

        Most fields are plain uniforms and take effect on the next repaint; the
        AO toggle re-uploads instance buffers (occlusion is precomputed per point).
        """
        self.render_settings = settings
        for renderer in (self.sphere_renderer, self.atom_renderer):
            if renderer is not None:
                renderer.set_ao_enabled(settings.ao_enabled)
        self.update()

    def setBackgroundColor(self, color):
        self.backgroundColor = QColor(color)
        self.makeCurrent()
        gl = self.context().functions()
        gl.glClearColor(color.redF(), color.greenF(), color.blueF(), 1)
        self.doneCurrent()

    def updateSettings(self, **kwargs):
        if not kwargs:
            return

        def present_and_changed(key, prev_val):
            return (key in kwargs) and (prev_val != kwargs[key])

        needs_reinit = False
        if present_and_changed("Color Map", self.colormap):
            self.colormap = kwargs["Color Map"]
            needs_reinit = True

        if "Style" in kwargs:
            # The settings panel sends the render option (Spheres / Points / Convex
            # Hull); set_render_option rebuilds the scene itself when it changes.
            self.set_render_option(kwargs["Style"])

        if present_and_changed("Show Mesh Edges", self.show_mesh_edges):
            self.show_mesh_edges = kwargs["Show Mesh Edges"]
            needs_reinit = True

        if present_and_changed("Background Color", self.backgroundColor):
            color = kwargs["Background Color"]
            self.setBackgroundColor(color)

        if present_and_changed("Color By", self.color_by):
            self.color_by = kwargs.get("Color By", self.color_by)
            needs_reinit = True

        if present_and_changed("Single Color", self.single_color):
            self.single_color = kwargs.get("Single Color", self.single_color)
            needs_reinit = True

        if present_and_changed("Point Size", self.point_size):
            self.point_size = float(kwargs["Point Size"])

        if "Axes Thickness" in kwargs and self.axes_renderer is not None:
            self.axes_renderer.set_axes_thickness(float(kwargs["Axes Thickness"]))

        if needs_reinit:
            self.initGeometry()

        self.update()

    def resizeGL(self, width, height):
        super().resizeGL(width, height)
        self.aspect_ratio = width / float(height)
        self.screen_size = width, height

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.geometry())

    def wheelEvent(self, event):
        degrees = event.angleDelta() / 8

        steps = degrees.y() / 15

        self.camera.zoom(steps)
        self.update()

    def mousePressEvent(self, event):
        self.lastMousePosition = event.pos()
        if event.button() == QtCore.Qt.LeftButton:
            modifiers = event.modifiers()
            if modifiers & QtCore.Qt.ControlModifier:
                # Cmd+Click (macOS): Select single point, or clear selection on whitespace
                point_idx, _ = self._find_point_at_screen_pos(event.pos().x(), event.pos().y())
                if point_idx is not None:
                    self.select_point(point_idx)
                else:
                    self.clear_selection()
            elif modifiers & QtCore.Qt.ShiftModifier:
                # Shift+Click (+ optional drag): sphere selection mode.
                # The sphere centre is always anchored to the nearest data point
                # under the cursor.  If no point is found, nothing happens.
                point_idx, _ = self._find_point_at_screen_pos(event.pos().x(), event.pos().y())
                if point_idx is not None:
                    self._sphere_sel_start_screen = event.pos()
                    self._sphere_sel_center_world = self._visual_data.centroids[point_idx].copy()
                    self._sphere_sel_anchor_idx = point_idx
                    self._sphere_sel_radius = 0.0
                    self._sphere_sel_active = False
            # Plain left click: camera orbit only (handled in mouseMoveEvent)

    def toggle_interaction_mode(self):
        """Toggle between camera orbit mode and object rotation mode."""
        self.interaction_mode = "object" if self.interaction_mode == "camera" else "camera"
        self.update()

    def toggle_mesh_edges(self):
        self.show_mesh_edges = not self.show_mesh_edges
        self.update()

    def increase_bond_radius(self):
        self._bond_radius = min(round(self._bond_radius + 0.05, 3), 2.0)
        self.initGeometry()

    def decrease_bond_radius(self):
        self._bond_radius = max(round(self._bond_radius - 0.05, 3), 0.02)
        self.initGeometry()

    def keyPressEvent(self, event):
        dx, dy = 0, 0
        self.restrict_axis = None
        modifiers = event.modifiers()

        # Arrow keys for rotation
        if event.key() == Qt.Key_Up:
            dy -= 10
        if event.key() == Qt.Key_Down:
            dy += 10
        if event.key() == Qt.Key_Left:
            dx -= 10
        if event.key() == Qt.Key_Right:
            dx += 10

        # Check for Shift + arrow keys for restricted rotation
        if modifiers & Qt.ShiftModifier:
            if event.key() == Qt.Key_Left or event.key() == Qt.Key_Right:
                self.restrict_axis = "shift_x"
            elif event.key() == Qt.Key_Up or event.key() == Qt.Key_Down:
                self.restrict_axis = "shift_y"

        super().keyPressEvent(event)

        if self.interaction_mode == "object":
            self.camera.rotate_model(dx, dy)
        else:
            self.camera.orbit(dx, dy, restrict_axis=self.restrict_axis)
        self.update()

    def keyReleaseEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            self.restrict_axis = None
        super().keyReleaseEvent(event)

    def _align_view_to_axis(self, axis):
        """Align the camera view to look along a specific axis.

        Args:
            axis: 'x', 'y', 'z' for Cartesian axes or 'a', 'b', 'c' for fractional axes
        """
        from PySide6.QtGui import QVector3D

        # Define view directions for each axis
        # Looking along +axis means camera is on -axis side looking toward origin
        if axis == "x":
            # Look along X axis (camera on -X, looking toward +X)
            direction = QVector3D(1, 0, 0)
            up = QVector3D(0, 1, 0)
        elif axis == "y":
            # Look along Y axis (camera on -Y, looking toward +Y)
            direction = QVector3D(0, 1, 0)
            up = QVector3D(0, 0, 1)
        elif axis == "z":
            # Look along Z axis (camera on -Z, looking toward +Z)
            direction = QVector3D(0, 0, 1)
            up = QVector3D(0, 1, 0)
        elif axis == "a":
            # Fractional a-axis - use crystallography if available
            if self.axes_renderer and self.axes_renderer.crystallography:
                frac_a = np.array([1, 0, 0])
                cart_a = self.axes_renderer.crystallography.frac_to_cart(frac_a.reshape(1, -1))[0]
                cart_a = cart_a / np.linalg.norm(cart_a)
                direction = QVector3D(cart_a[0], cart_a[1], cart_a[2])
                up = QVector3D(0, 1, 0)
            else:
                # Fall back to Cartesian X
                direction = QVector3D(1, 0, 0)
                up = QVector3D(0, 1, 0)
        elif axis == "b":
            # Fractional b-axis - use crystallography if available
            if self.axes_renderer and self.axes_renderer.crystallography:
                frac_b = np.array([0, 1, 0])
                cart_b = self.axes_renderer.crystallography.frac_to_cart(frac_b.reshape(1, -1))[0]
                cart_b = cart_b / np.linalg.norm(cart_b)
                direction = QVector3D(cart_b[0], cart_b[1], cart_b[2])
                up = QVector3D(0, 0, 1)
            else:
                # Fall back to Cartesian Y
                direction = QVector3D(0, 1, 0)
                up = QVector3D(0, 0, 1)
        elif axis == "c":
            # Fractional c-axis - use crystallography if available
            if self.axes_renderer and self.axes_renderer.crystallography:
                frac_c = np.array([0, 0, 1])
                cart_c = self.axes_renderer.crystallography.frac_to_cart(frac_c.reshape(1, -1))[0]
                cart_c = cart_c / np.linalg.norm(cart_c)
                direction = QVector3D(cart_c[0], cart_c[1], cart_c[2])
                up = QVector3D(0, 1, 0)
            else:
                # Fall back to Cartesian Z
                direction = QVector3D(0, 0, 1)
                up = QVector3D(0, 1, 0)
        else:
            return

        # The object axes may be rotated by model_rotation (OBJ mode), so always
        # align to the rendered axis direction rather than the raw world axis.
        q = self.camera.model_rotation
        direction = q.rotatedVector(direction)
        up = q.rotatedVector(up)

        # Set camera position along the negative direction, looking toward target
        distance = (self.camera.position - self.camera.target).length()
        self.camera.position = self.camera.target - direction * distance
        self.camera.up = up
        self.camera.right = QVector3D.crossProduct(up, direction).normalized()
        self.update()

    # ------------------------------------------------------------------
    # Public viewport action methods (wired to menu QActions)
    # ------------------------------------------------------------------

    def align_view_x(self):
        self._align_view_to_axis("x")

    def align_view_y(self):
        self._align_view_to_axis("y")

    def align_view_z(self):
        self._align_view_to_axis("z")

    def align_view_a(self):
        self._align_view_to_axis("a")

    def align_view_b(self):
        self._align_view_to_axis("b")

    def align_view_c(self):
        self._align_view_to_axis("c")

    def reset_view(self):
        self.camera.resetOrientation()
        self.update()

    def store_view(self):
        self.camera.storeOrientation()

    def set_point_size(self, value: int):
        self.point_size = float(value)
        self.update()

    def toggle_rotation_lock(self, axis: str, checked: bool):
        """Set/clear crystal rotation lock. Called by checkable menu QActions."""
        self.rotation_lock_axis = axis if checked else None

    def _screen_to_ray(self, screen_x, screen_y):
        """Convert screen coordinates to a ray in world space.

        Returns:
            tuple: (ray_origin, ray_direction) as numpy arrays
        """
        # Get normalized device coordinates
        ndc_x = (2.0 * screen_x / self.width()) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / self.height())

        # Get the inverse of the model-view-projection matrix
        mvp = self.camera.modelViewProjectionMatrix(self.aspect_ratio)
        mvp_inv = mvp.inverted()[0]

        # Near and far points in NDC
        near_point = mvp_inv.map(QVector3D(ndc_x, ndc_y, -1.0))
        far_point = mvp_inv.map(QVector3D(ndc_x, ndc_y, 1.0))

        # Ray direction
        ray_dir = far_point - near_point
        ray_dir.normalize()

        ray_origin = np.array([near_point.x(), near_point.y(), near_point.z()])
        ray_direction = np.array([ray_dir.x(), ray_dir.y(), ray_dir.z()])

        return ray_origin, ray_direction

    def _find_point_at_screen_pos(self, screen_x, screen_y):
        """Find the closest point to a screen position.

        Args:
            screen_x: X coordinate in screen space
            screen_y: Y coordinate in screen space

        Returns:
            tuple: (point_index, distance) or (None, None) if no point found
        """
        if self._visual_data is None or self._visual_data.n_centroids == 0:
            return None, None

        ray_origin, ray_direction = self._screen_to_ray(screen_x, screen_y)

        points = self._visual_data.centroids

        # Calculate distance from each point to the ray
        # Using point-to-line distance formula
        # d = ||(p - o) - ((p - o) · d) * d|| where p=point, o=origin, d=direction

        # Vector from ray origin to each point
        to_points = points - ray_origin

        # Project onto ray direction
        projections = np.dot(to_points, ray_direction)

        # Only consider points in front of the camera
        valid_mask = projections > 0

        # Exclude deleted points
        for idx in self._deleted_points:
            if idx < len(valid_mask):
                valid_mask[idx] = False

        if not np.any(valid_mask):
            return None, None

        # Calculate perpendicular distance to ray
        closest_on_ray = ray_origin + np.outer(projections, ray_direction)
        distances = np.linalg.norm(points - closest_on_ray, axis=1)

        # Apply mask for valid points
        distances[~valid_mask] = np.inf

        # Find minimum distance
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        # Check if within picking radius (scale by point size)
        pick_threshold = self._pick_radius * self.point_size
        if min_dist < pick_threshold:
            return min_idx, min_dist

        return None, None

    # ------------------------------------------------------------------
    # Sphere selection helpers
    # ------------------------------------------------------------------

    def _compute_sphere_radius(self, current_screen_pos):
        """Compute world-space sphere radius from the current mouse position.

        Projects the current cursor onto the plane that passes through the sphere
        centre and is perpendicular to the view direction, then returns the 3D
        distance to the centre.
        """
        if self._sphere_sel_center_world is None:
            return 0.0

        ray_origin, ray_dir = self._screen_to_ray(current_screen_pos.x(), current_screen_pos.y())
        center = self._sphere_sel_center_world

        # View direction (camera.screen points from camera toward target)
        view_dir = np.array(
            [self.camera.screen.x(), self.camera.screen.y(), self.camera.screen.z()]
        )

        denom = np.dot(ray_dir, view_dir)
        if abs(denom) < 1e-6:
            # Ray nearly parallel to plane — keep last radius
            return self._sphere_sel_radius

        t = np.dot(center - ray_origin, view_dir) / denom
        if t < 0:
            return 0.0

        current_world = ray_origin + t * ray_dir
        return float(np.linalg.norm(current_world - center))

    def _update_sphere_selection(self):
        """Recompute which points lie within the current selection sphere."""
        if self._visual_data is None or self._sphere_sel_center_world is None:
            return

        points = self._visual_data.centroids
        diffs = points - self._sphere_sel_center_world
        distances = np.linalg.norm(diffs, axis=1)

        within = set(np.where(distances <= self._sphere_sel_radius)[0].tolist())
        # Remove deleted points from the candidate set
        within -= self._deleted_points

        self._selected_points = within
        self.initGeometry()

    def _draw_sphere_selection(self, gl, uniforms):
        """Draw the transparent selection sphere with alpha blending."""
        from OpenGL.GL import GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA

        if self.sphere_selection_renderer is None:
            return

        self.sphere_selection_renderer.set_sphere(
            self._sphere_sel_center_world, self._sphere_sel_radius
        )

        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.sphere_selection_renderer.bind()
        self.sphere_selection_renderer.setUniforms(**uniforms)
        self.sphere_selection_renderer.draw(gl)
        self.sphere_selection_renderer.release()

        gl.glDisable(GL_BLEND)

    def _get_point_data(self, point_index):
        """Get data for a specific point.

        Args:
            point_index: Index of the point

        Returns:
            dict: Point data including position, type, number, layer, etc.
        """
        if (
            self._visual_data is None
            or point_index is None
            or point_index >= self._visual_data.n_centroids
        ):
            return None

        vd = self._visual_data
        pos = vd.centroids[point_index]
        data = {
            "position": (float(pos[0]), float(pos[1]), float(pos[2])),
            "type": int(vd.mol_types[point_index]),
            "number": int(vd.mol_numbers[point_index]) if vd.mol_numbers is not None else 0,
            "layer": int(vd.layers[point_index]) if vd.layers is not None else 0,
        }
        if vd.site_numbers is not None:
            data["site"] = int(vd.site_numbers[point_index])
        if vd.energies is not None:
            data["energy"] = float(vd.energies[point_index])
        return data

    def get_selected_points(self):
        """Get the set of currently selected point indices."""
        return self._selected_points.copy()

    def clear_selection(self):
        """Clear the current point selection."""
        self._selected_points.clear()
        self._last_selected_index = None
        self.selectionChanged.emit(self._selected_points.copy(), None)
        self.initGeometry()
        self.update()

    def select_point(self, index, add_to_selection=False, toggle=False):
        """Select a point by index.

        Args:
            index: Point index to select
            add_to_selection: If True, add to existing selection
            toggle: If True, toggle selection state
        """
        if index is None:
            return

        if toggle:
            if index in self._selected_points:
                self._selected_points.discard(index)
            else:
                self._selected_points.add(index)
        elif add_to_selection:
            self._selected_points.add(index)
        else:
            self._selected_points = {index}

        self._last_selected_index = index
        self.selectionChanged.emit(self._selected_points.copy(), index)
        self.initGeometry()
        self.update()

    def select_range(self, end_index):
        """Select a range of points from last selected to end_index."""
        if self._last_selected_index is None or end_index is None:
            return

        start = min(self._last_selected_index, end_index)
        end = max(self._last_selected_index, end_index)

        for i in range(start, end + 1):
            if i not in self._deleted_points:
                self._selected_points.add(i)

        self.selectionChanged.emit(self._selected_points.copy(), end_index)
        self.initGeometry()
        self.update()

    def delete_selected_points(self):
        """Delete the currently selected points.

        Returns:
            int: Number of points deleted
        """
        if not self._selected_points:
            return 0

        count = len(self._selected_points)
        self._deleted_points.update(self._selected_points)
        self._selected_points.clear()
        self._last_selected_index = None

        self.selectionChanged.emit(self._selected_points.copy(), None)
        self.pointsDeleted.emit(count)
        self.initGeometry()
        self.update()

        logger.info(f"Deleted {count} points, total deleted: {len(self._deleted_points)}")
        return count

    def restore_deleted_points(self):
        """Restore all deleted points."""
        count = len(self._deleted_points)
        self._deleted_points.clear()
        self.initGeometry()
        self.update()
        logger.info(f"Restored {count} deleted points")
        return count

    def get_legend_info(self):
        """Return the most recently computed legend info dict, or None if not yet available."""
        return self._legend_info

    # ------------------------------------------------------------------
    # Legend colour filter (show only selected legend colours)
    # ------------------------------------------------------------------

    def set_legend_filter(self, visible_keys):
        """Restrict the view to points whose colour matches the given legend keys.

        ``visible_keys`` is a set of legend row keys (element symbols, shell
        names or numeric values, matching the current legend's ``rows``), or
        ``None`` to clear the filter and show every colour again.
        """
        self._legend_filter = set(visible_keys) if visible_keys is not None else None
        self._legend_filter_sig = (self.vis_mode, self.is_atom_view, self.color_by)
        self.initGeometry()
        self.update()

    def _sync_filter_signature(self):
        """Drop the legend filter when the (style, color_by) it was set for changes.

        Called at the start of each geometry builder so a filter picked for one
        colouring never leaks into a different one.
        """
        sig = (self.vis_mode, self.is_atom_view, self.color_by)
        if sig != self._legend_filter_sig:
            self._legend_filter = None
            self._legend_filter_sig = sig

    @staticmethod
    def _pack_rgb(colors: np.ndarray) -> np.ndarray:
        """Pack an (N, 3) float RGB array into (N,) int64 codes for exact matching."""
        q = np.clip(np.rint(np.asarray(colors, dtype=np.float64) * 65535.0), 0, 65535).astype(
            np.int64
        )
        return (q[:, 0] << 32) | (q[:, 1] << 16) | q[:, 2]

    def _visible_color_codes(self):
        """Packed RGB codes for the currently-visible legend rows, or None (no filter)."""
        if self._legend_filter is None or not self._legend_info:
            return None
        rows = self._legend_info.get("rows", [])
        visible_rgbs = [rgb for key, rgb in rows if key in self._legend_filter]
        if not visible_rgbs:
            return np.zeros(0, dtype=np.int64)
        return np.unique(self._pack_rgb(np.array(visible_rgbs, dtype=np.float64)))

    def _color_visibility_mask(self, colors: np.ndarray):
        """Boolean keep-mask over ``colors`` for the active filter, or None if inactive."""
        codes_visible = self._visible_color_codes()
        if codes_visible is None:
            return None
        if len(colors) == 0:
            return np.zeros(0, dtype=bool)
        return np.isin(self._pack_rgb(colors), codes_visible)

    def _apply_atom_color_filter(self, atom_arr, bond_arr):
        """Drop atoms (and their bonds) whose colour is hidden by the legend filter."""
        if self._legend_filter is None or atom_arr is None or len(atom_arr) == 0:
            return atom_arr, bond_arr
        amask = self._color_visibility_mask(atom_arr[:, 3:6])
        if amask is None:
            return atom_arr, bond_arr
        atom_arr = atom_arr[amask]
        if bond_arr is not None and len(bond_arr):
            bmask = self._color_visibility_mask(bond_arr[:, 6:9])
            if bmask is not None:
                bond_arr = bond_arr[bmask]
        return atom_arr, bond_arr

    def _emit_rendered_count(self, count: int, label: str):
        """Publish the number of primitives actually uploaded (post-filter)."""
        self.renderedCountChanged.emit(int(count), label)

    def get_active_xyz(self):
        """Get XYZ data excluding deleted points.

        Returns:
            np.ndarray: XYZ data with deleted points removed
        """
        if self._visual_data is None:
            return None
        vd = self._visual_data
        n = vd.n_centroids
        # Reconstruct the (N, 7+) column layout expected by exportXYZ
        mol_nums = vd.mol_numbers if vd.mol_numbers is not None else np.zeros(n, int)
        layers = vd.layers if vd.layers is not None else np.zeros(n, int)
        cols = [
            vd.mol_types.reshape(-1, 1).astype(float),
            mol_nums.reshape(-1, 1).astype(float),
            layers.reshape(-1, 1).astype(float),
            vd.centroids.astype(float),
        ]
        if vd.site_numbers is not None:
            cols.append(vd.site_numbers.reshape(-1, 1).astype(float))
            if vd.energies is not None:
                cols.append(vd.energies.reshape(-1, 1).astype(float))
        raw = np.hstack(cols)

        if not self._deleted_points:
            return raw
        mask = np.ones(n, dtype=bool)
        for idx in self._deleted_points:
            if idx < n:
                mask[idx] = False
        return raw[mask]

    def get_display_count(self) -> tuple[int | None, str]:
        """(count, label) for the crystal-info panel based on the active render mode."""
        if self._visual_data is None:
            return None, "Points"
        return self._visual_data.display_count()

    def _active_slice_planes(self) -> list:
        """Return resolved (normal, origin, two_sided, thickness) for active slice planes."""
        planes = []
        for plane in self._raw_planes:
            if not plane.slice_enabled:
                continue
            normal = np.array(plane.normal, dtype=np.float64)
            if plane.fractional and self._planes_crystallography is not None:
                # Plane normals are Miller indices (hkl) — use reciprocal lattice transform.
                normal = self._planes_crystallography.miller_to_cart_normal(normal)
            n_len = np.linalg.norm(normal)
            if n_len < 1e-9:
                continue
            normal /= n_len
            origin = np.array(plane.origin, dtype=np.float64)
            planes.append((normal, origin, plane.slice_two_sided, plane.slice_thickness))
        return planes

    def _slice_centroid_mask(self, centroids: np.ndarray) -> np.ndarray | None:
        """Boolean keep-mask (N,) for *centroids* against all active slice planes.

        Returns None when no slice planes are active so callers can skip the filter
        entirely without allocating an all-True array.
        """
        slice_planes = self._active_slice_planes()
        if not slice_planes:
            return None
        points = centroids.astype(np.float64)
        mask = np.ones(len(points), dtype=bool)
        for normal, origin, two_sided, thickness in slice_planes:
            d = (points - origin) @ normal
            if two_sided:
                mask &= np.abs(d) <= thickness / 2.0
            else:
                mask &= d >= -thickness
        return mask

    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.lastMousePosition.x()
        dy = event.pos().y() - self.lastMousePosition.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            if self._sphere_sel_center_world is not None:
                # Shift+drag sphere selection: update radius and selection
                self._sphere_sel_active = True
                self._sphere_sel_radius = self._compute_sphere_radius(event.pos())
                self._update_sphere_selection()
            elif self.interaction_mode == "object":
                axis = self.rotation_lock_axis
                if axis:
                    q = QQuaternion.fromAxisAndAngle(
                        QVector3D(
                            1 if axis == "x" else 0,
                            1 if axis == "y" else 0,
                            1 if axis == "z" else 0,
                        ),
                        dx * self.camera.rotationSpeed,
                    )
                    self.camera.model_rotation = (q * self.camera.model_rotation).normalized()
                else:
                    self.camera.rotate_model(dx, dy)
            elif self.rotation_lock_axis:
                # Camera mode with rotation lock: orbit restricted to axis
                self.camera.orbit(
                    dx,
                    dy,
                    restrict_axis=self.rotation_lock_axis,
                    event_pos=event.pos() - self.geometry().center(),
                )
            else:
                self.camera.orbit(
                    dx,
                    dy,
                    restrict_axis=self.restrict_axis,
                    event_pos=event.pos() - self.geometry().center(),
                )

        # Handle hover detection when no button is pressed
        elif event.buttons() == QtCore.Qt.NoButton:
            point_idx, _ = self._find_point_at_screen_pos(event.pos().x(), event.pos().y())
            if point_idx != self._hovered_point_index:
                self._hovered_point_index = point_idx
                point_data = self._get_point_data(point_idx) if point_idx is not None else None
                self.pointHovered.emit(point_idx, point_data)

        self.lastMousePosition = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._sphere_sel_center_world is not None:
            if not self._sphere_sel_active:
                # Plain Shift+Click (no drag): toggle the anchor point
                self.select_point(self._sphere_sel_anchor_idx, toggle=True)
            else:
                # Emit final selection signal after sphere drag
                self.selectionChanged.emit(self._selected_points.copy(), None)
            # Clear sphere selection state
            self._sphere_sel_active = False
            self._sphere_sel_center_world = None
            self._sphere_sel_anchor_idx = None
            self._sphere_sel_radius = 0.0
            self.update()

    def rotatePointCloud(self, dx, axis):
        """Deprecated: rotates object via model matrix instead of mutating vertex data."""
        axis_map = {"x": QVector3D(1, 0, 0), "y": QVector3D(0, 1, 0), "z": QVector3D(0, 0, 1)}
        if axis not in axis_map:
            return
        q = QQuaternion.fromAxisAndAngle(axis_map[axis], dx * self.camera.rotationSpeed)
        self.camera.model_rotation = (q * self.camera.model_rotation).normalized()
        self.update()

    def initGeometry(self):
        # self.update()
        if self.point_cloud_renderer is None:
            return

        if self.vis_mode == "Docking":
            if self.is_atom_view:
                self._update_docking_atom_view()
            else:
                self._update_docking_sphere_view()
            self.update()
            return

        if self.vis_mode == "Checkpoint":
            if self.is_atom_view:
                self._update_checkpoint_atom_view()
            else:
                self._update_checkpoint_view()
            self.update()
            return

        if self.is_atom_view:
            self._update_atom_view()
            self._apply_planes()
            self._apply_directions()
            self.update()
            return

        varray = self.updatePointCloudVertices()
        self._upload_centroid_varray(varray)
        self.update()

    def _upload_centroid_varray(self, varray, hull_points=None):
        """Upload a (N, 7) centroid vertex array to the renderer for the active
        render style, clearing the inactive one.

        Only the active renderer is fed: at checkpoint scale the array can run to
        gigabytes, so mirroring it into a renderer that is never drawn doubles the
        upload time and GPU memory for nothing. Switching render style goes through
        set_render_option → initGeometry, which re-uploads to the new target.

        When Convex Hull is active the hull is built from ``hull_points`` (an
        (M, 3) position array) if given, else from the varray positions. Callers
        with a surface-only subset should pass it — interior points can never be
        hull vertices, so the result is identical and Qhull runs on far fewer
        points.
        """
        if varray is None:
            varray = np.zeros((0, 7), dtype=np.float32)
        empty = np.zeros((0, 7), dtype=np.float32)

        if self.render_option == "Points":
            self.point_cloud_renderer.setPoints(varray)
            self.sphere_renderer.setPoints(empty)
            return

        if self.render_option == "Convex Hull":
            self.point_cloud_renderer.setPoints(empty)
            self.sphere_renderer.setPoints(empty)
            points = varray[:, :3] if hull_points is None else hull_points
            if len(points) < 4:
                self.mesh_renderer.setMesh(None)
                return
            try:
                hull = ConvexHull(points)
            except Exception as exc:  # QhullError on degenerate point sets
                logger.warning("Convex hull failed: %s", exc)
                self.mesh_renderer.setMesh(None)
                return
            mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
            # can pass vertex colors here, but I wouldn't
            self.mesh_renderer.setMesh(mesh)

            if self.show_mesh_edges:
                self.line_renderer.setLines(self.mesh_renderer.getLines())
            return

        self.sphere_renderer.setPoints(varray)
        self.point_cloud_renderer.setPoints(empty)

    def updatePointCloudVertices(self):
        self.overlay.setVisible(False)
        self._sync_filter_signature()
        vd = self._visual_data
        if vd is None:
            # No point-cloud data (e.g. a checkpoint-only sim shown in grid mode).
            return np.zeros((0, 7), dtype=np.float32)
        logger.debug("Loading Vertices: %s centroids", vd.n_centroids)

        col_idx = self.columnLabelToIndex.get(self.color_by, 2)
        if col_idx < -1:
            col_idx = 2

        # Resolve the 1-D values array for colormap, respecting available attributes
        if col_idx == -1:
            # Single colour
            single_rgb = self._single_color_rgb()
            colors = vd.colors_uniform(single_rgb)
            min_val, max_val, legend_rows = 0.0, 0.0, [(None, tuple(single_rgb.tolist()))]
        else:
            # Map col_idx to VisualData attribute
            if col_idx == 0:
                values = vd.mol_types.astype(np.float32)
            elif col_idx == 1:
                values = (
                    vd.mol_numbers.astype(np.float32)
                    if vd.mol_numbers is not None
                    else np.arange(vd.n_centroids, dtype=np.float32)
                )
            elif col_idx == 2:
                if vd.layers is not None:
                    raw_layers = vd.layers.astype(np.float32)
                    valid = raw_layers[raw_layers < 99]
                    max_layers = int(np.nanmax(valid)) if valid.size else 1
                    values = raw_layers
                    min_val, max_val = 1.0, float(max_layers)
                else:
                    values = np.zeros(vd.n_centroids, dtype=np.float32)
                    min_val = max_val = 0.0
            elif col_idx == 3:
                values = np.arange(vd.n_centroids, dtype=np.float32)
            elif col_idx == 6:
                values = (
                    vd.site_numbers.astype(np.float32)
                    if vd.site_numbers is not None
                    else np.zeros(vd.n_centroids, dtype=np.float32)
                )
                if vd.site_numbers is None:
                    logger.warning("Old CrystalGrower version! %s not available.", self.color_by)
            elif col_idx == 7:
                values = (
                    vd.energies.astype(np.float32)
                    if vd.energies is not None
                    else np.zeros(vd.n_centroids, dtype=np.float32)
                )
                if vd.energies is None:
                    logger.warning("Old CrystalGrower version! %s not available.", self.color_by)
            else:
                values = np.arange(vd.n_centroids, dtype=np.float32)

            if col_idx != 2:
                min_val = float(np.nanmin(values))
                max_val = float(np.nanmax(values))
            colors = vd.colors_by_array(
                values, self.availableColormaps[self.colormap], min_val, max_val
            )
            unique_vals = np.unique(values)
            rng = max_val - min_val if max_val != min_val else 1.0
            norm_u = (unique_vals - min_val) / rng
            unique_rgb = self.availableColormaps[self.colormap](norm_u)[:, :3]
            legend_rows = [(float(v), tuple(c.tolist())) for v, c in zip(unique_vals, unique_rgb)]

        self._legend_info = {
            "color_by": self.color_by,
            "colormap": self.colormap,
            "min_val": float(min_val),
            "max_val": float(max_val),
            "rows": legend_rows,
        }
        self.legendChanged.emit(self._legend_info)

        points = vd.centroids.astype(np.float32)
        colors = colors.astype(np.float32)

        if not self.viewInitialized:
            self.camera.fitToObject(points)
            self.viewInitialized = True

        # Site highlighting
        if self.highlight_groups and vd.site_numbers is not None:
            if self.background_color_override is not None:
                colors[:] = self.background_color_override
            for site_set, highlight_color in self.highlight_groups:
                mask = np.isin(vd.site_numbers, list(site_set))
                colors[mask] = highlight_color

        # Direct colour override (coordination number, cluster colours, etc.)
        if self._colour_override is not None:
            if self._colour_override.shape == colors.shape:
                colors = self._colour_override
            else:
                # Frame mismatch — positional indexing is unreliable, show all grey
                logger.warning(
                    "Colour override length %d != point cloud length %d — "
                    "showing grey (frame mismatch). Use 'Show Analysis Data' to diagnose.",
                    len(self._colour_override), len(colors),
                )
                colors = np.full(colors.shape, 0.5, dtype=np.float32)

        # Selection flags
        selection_flags = np.zeros((len(points), 1), dtype=np.float32)
        for idx in self._selected_points:
            if idx < len(selection_flags):
                selection_flags[idx] = 1.0

        # Combined mask: deleted points + slice planes
        n_pts = len(points)
        combined_mask = np.ones(n_pts, dtype=bool)
        for idx in self._deleted_points:
            if idx < n_pts:
                combined_mask[idx] = False
        slice_mask = self._slice_centroid_mask(points)
        if slice_mask is not None:
            combined_mask &= slice_mask

        # Legend colour filter (show only selected colours)
        color_mask = self._color_visibility_mask(colors)
        if color_mask is not None:
            combined_mask &= color_mask

        if not np.all(combined_mask):
            points = points[combined_mask]
            colors = colors[combined_mask]
            selection_flags = selection_flags[combined_mask]

        self._emit_rendered_count(len(points), "Points")

        try:
            return np.concatenate((points, colors, selection_flags), axis=1)
        except ValueError as exc:
            logger.error(
                "%s\n CENTROIDS %s COLORS %s TYPE %s",
                exc,
                points.shape,
                colors.shape,
                self.color_by,
            )
            return

    @classmethod
    def _shell_name(cls, shell_id: int) -> str:
        for sid, name in cls._SHELL_NAMES:
            if sid == shell_id:
                return name
        return str(shell_id)

    @classmethod
    def _shell_id(cls, name: str) -> int | None:
        for sid, n in cls._SHELL_NAMES:
            if n == name:
                return sid
        return None

    def _emit_atom_legend(self):
        """Build and emit a legend for the current Atoms/Unit Cell view."""
        if not (self._visual_data and self._visual_data.templates):
            return
        if self.color_by == "Atom":
            # One row per unique element symbol, using resolved colors
            seen: dict[str, tuple] = {}
            for tmpl in self._visual_data.templates.values():
                colors, _ = self._resolved_atom_colors_radii(tmpl)
                for sym, rgb in zip(tmpl["symbols"], colors):
                    if sym not in seen:
                        seen[sym] = tuple(float(v) for v in rgb)
            rows = [(sym, seen[sym]) for sym in sorted(seen)]
            info = {
                "color_by": "Atom",
                "colormap": self.colormap,
                "min_val": 0.0,
                "max_val": float(len(rows) - 1),
                "rows": rows,
                "mode": "atom",  # signals legend dialog to use symbol labels
            }
        else:
            # Colormap-based: build legend from visual data
            vd = self._visual_data
            if vd is None:
                return
            col_idx = self.columnLabelToIndex.get(self.color_by, 2)
            if col_idx < -1:
                col_idx = 2
            if col_idx == -1:
                rgb = np.array(
                    [
                        self.single_color.redF(),
                        self.single_color.greenF(),
                        self.single_color.blueF(),
                    ],
                    dtype=np.float32,
                )
                rows = [(None, tuple(rgb.tolist()))]
                info = {
                    "color_by": self.color_by,
                    "colormap": self.colormap,
                    "min_val": 0.0,
                    "max_val": 0.0,
                    "rows": rows,
                    "mode": "colormap",
                }
            else:
                if col_idx == 3:
                    axis_vis = np.arange(vd.n_centroids, dtype=np.float32)
                elif col_idx == 0:
                    axis_vis = vd.mol_types.astype(np.float32)
                elif col_idx == 1:
                    axis_vis = (
                        vd.mol_numbers.astype(np.float32)
                        if vd.mol_numbers is not None
                        else np.zeros(vd.n_centroids, dtype=np.float32)
                    )
                elif col_idx == 2:
                    axis_vis = (
                        vd.layers.astype(np.float32)
                        if vd.layers is not None
                        else np.zeros(vd.n_centroids, dtype=np.float32)
                    )
                elif col_idx == 6:
                    axis_vis = (
                        vd.site_numbers.astype(np.float32)
                        if vd.site_numbers is not None
                        else np.zeros(vd.n_centroids, dtype=np.float32)
                    )
                elif col_idx == 7:
                    axis_vis = (
                        vd.energies.astype(np.float32)
                        if vd.energies is not None
                        else np.zeros(vd.n_centroids, dtype=np.float32)
                    )
                else:
                    axis_vis = np.arange(vd.n_centroids, dtype=np.float32)
                if vd.layers is not None:
                    raw_layers = vd.layers.astype(float)
                    valid_layers = raw_layers[raw_layers < 99]
                    max_layers = int(np.nanmax(valid_layers)) if valid_layers.size else 1
                else:
                    max_layers = 1
                min_val = 1.0 if col_idx == 2 else float(np.nanmin(axis_vis))
                max_val = float(max_layers) if col_idx == 2 else float(np.nanmax(axis_vis))
                range_val = max_val - min_val if max_val != min_val else 1.0
                unique_vals = np.unique(axis_vis)
                norm_unique = (unique_vals - min_val) / range_val
                unique_rgb = self.availableColormaps[self.colormap](norm_unique)[:, :3]
                rows = [(float(v), tuple(c.tolist())) for v, c in zip(unique_vals, unique_rgb)]
                info = {
                    "color_by": self.color_by,
                    "colormap": self.colormap,
                    "min_val": min_val,
                    "max_val": max_val,
                    "rows": rows,
                    "mode": "colormap",
                }
        self._legend_info = info
        self.legendChanged.emit(info)

    def _emit_docking_legend(self):
        """Build and emit a legend for the current Docking/Docking Atoms view."""
        if self._docking_data is None or self._docking_data.empty:
            return

        if self.color_by == "Coordination Shell":
            rows = []
            for shell_id, default_color in self._docking_data.SHELL_COLORS.items():
                color = self._docking_shell_color_overrides.get(shell_id, default_color)
                rows.append((self._shell_name(shell_id), tuple(float(v) for v in color)))
            info = {
                "color_by": "Coordination Shell",
                "colormap": self.colormap,
                "min_val": 0.0,
                "max_val": float(len(rows) - 1),
                "rows": rows,
                "mode": "docking_shell",
            }
        elif self.color_by == "Atom Type":
            mol_types = self._docking_data.mol_types.astype(np.float32)
            min_val, max_val = float(mol_types.min()), float(mol_types.max())
            range_val = max_val - min_val if max_val != min_val else 1.0
            unique_vals = np.unique(mol_types)
            norm_unique = (unique_vals - min_val) / range_val
            unique_rgb = self.availableColormaps[self.colormap](norm_unique)[:, :3]
            rows = [(float(v), tuple(c.tolist())) for v, c in zip(unique_vals, unique_rgb)]
            info = {
                "color_by": "Atom Type",
                "colormap": self.colormap,
                "min_val": min_val,
                "max_val": max_val,
                "rows": rows,
                "mode": "colormap",
            }
        elif (
            self.color_by == "Atom"
            and self._docking_visual_data
            and self._docking_visual_data.templates
        ):
            # Atom-element coloring in Docking Atoms mode
            seen: dict[str, tuple] = {}
            for tmpl in self._docking_visual_data.templates.values():
                colors, _ = self._resolved_atom_colors_radii(tmpl)
                for sym, rgb in zip(tmpl["symbols"], colors):
                    if sym not in seen:
                        seen[sym] = tuple(float(v) for v in rgb)
            rows = [(sym, seen[sym]) for sym in sorted(seen)]
            info = {
                "color_by": "Atom",
                "colormap": self.colormap,
                "min_val": 0.0,
                "max_val": float(len(rows) - 1),
                "rows": rows,
                "mode": "atom",
            }
        else:
            return
        self._legend_info = info
        self.legendChanged.emit(info)

    def _emit_checkpoint_legend(self, atom_view: bool):
        """Build and emit a legend for the current Checkpoint / Checkpoint Atoms view.

        Row colours are computed exactly as the matching builder colours the
        centroids/atoms, so the legend colour filter matches them one-to-one.
        """
        vd = self._checkpoint_visual_data
        if vd is None or vd.n_centroids == 0:
            return
        color_by = self.color_by

        # Atom-element colouring (Checkpoint Atoms only)
        if atom_view and color_by == "Atom" and vd.templates:
            seen: dict[str, tuple] = {}
            for tmpl in vd.templates.values():
                colors, _ = self._resolved_atom_colors_radii(tmpl)
                for sym, rgb in zip(tmpl["symbols"], colors):
                    if sym not in seen:
                        seen[sym] = tuple(float(v) for v in rgb)
            rows = [(sym, seen[sym]) for sym in sorted(seen)]
            info = {
                "color_by": "Atom", "colormap": self.colormap, "min_val": 0.0,
                "max_val": float(len(rows) - 1), "rows": rows, "mode": "atom",
            }
        # Site-analysis scalar (coordination / energy / population) colormap
        elif color_by in self._SITE_METADATA_COLOR_BY and self._site_metadata.get(color_by):
            site_map = self._site_metadata.get(color_by, {})
            sites = vd.site_numbers
            vals = (
                np.array([site_map.get(int(s), np.nan) for s in sites], dtype=float)
                if sites is not None else np.zeros(0)
            )
            valid = vals[~np.isnan(vals)]
            if valid.size:
                mn, mx = float(valid.min()), float(valid.max())
                rng = mx - mn if mx != mn else 1.0
                uniq = np.unique(valid)
                rgb = self.availableColormaps[self.colormap]((uniq - mn) / rng)[:, :3]
                rows = [(float(v), tuple(c.tolist())) for v, c in zip(uniq, rgb)]
            else:
                mn = mx = 0.0
                rows = []
            info = {
                "color_by": color_by, "colormap": self.colormap, "min_val": mn,
                "max_val": mx, "rows": rows, "mode": "colormap",
            }
        # Per-tile palette (Checkpoint Atoms, default colouring)
        elif atom_view:
            n_tiles = self._checkpoint.n_tiles if self._checkpoint else 1
            palette = cm.tab10(np.linspace(0, 1, max(n_tiles, 1)))[:, :3].astype(np.float32)
            tiles = vd.tile_indices
            uniq = np.unique(tiles) if tiles is not None else np.zeros(0, dtype=int)
            rows = [
                (int(t), tuple(float(x) for x in palette[int(t) % len(palette)])) for t in uniq
            ]
            info = {
                "color_by": color_by, "colormap": self.colormap, "min_val": 0.0,
                "max_val": float(max(len(rows) - 1, 0)), "rows": rows, "mode": "colormap",
            }
        # Z-layer gradient (Checkpoint spheres) — sampled so the dialog shows a gradient
        elif color_by == "Z Layer":
            z = vd.centroids[:, 2]
            zmin, zmax = float(z.min()), float(z.max())
            ts = np.linspace(0.0, 1.0, 33)
            rows = [
                (zmin + t * (zmax - zmin), (float(t), 0.0, float(1.0 - t))) for t in ts
            ]
            info = {
                "color_by": "Z Layer", "colormap": self.colormap, "min_val": zmin,
                "max_val": zmax, "rows": rows, "mode": "colormap",
            }
        # Single solid colour (Checkpoint spheres, default)
        else:
            rgb = tuple(float(x) for x in self._single_color_rgb())
            info = {
                "color_by": color_by, "colormap": self.colormap, "min_val": 0.0,
                "max_val": 0.0, "rows": [(None, rgb)], "mode": "colormap",
            }
        self._legend_info = info
        self.legendChanged.emit(info)

    def _available_site_color_by(self) -> tuple[str, ...]:
        """Site-analysis colour fields that actually have data loaded."""
        return tuple(
            label
            for label in self._SITE_METADATA_COLOR_BY
            if self._site_metadata.get(label)
        )

    def set_site_metadata(self, maps: dict[str, dict[int, float]] | None):
        """Provide per-site colour maps from the site-analysis workflow.

        ``maps`` is {field_label: {site_number: value}}. Refreshes the view if a
        checkpoint style is active so newly-available colour modes take effect.
        """
        self._site_metadata = maps or {}
        if self.vis_mode == "Checkpoint":
            self.initGeometry()
            self.update()

    def _color_by_options_for_state(
        self, mode: str | None = None, atom_view: bool | None = None
    ) -> tuple[tuple, str]:
        """Return (options_tuple, default_option) for a (mode, atom_view) pair.

        Defaults to the widget's current visualisation mode and atom/centroid state.
        """
        if mode is None:
            mode = self.vis_mode
        if atom_view is None:
            atom_view = self._atom_view.get(mode, False)

        if mode == "Docking":
            if atom_view:
                return self._DOCKING_ATOM_COLOR_BY, "Atom"
            return self._DOCKING_COLOR_BY, "Coordination Shell"
        if mode == "Checkpoint":
            if atom_view:
                return self._CHECKPOINT_ATOM_COLOR_BY + self._available_site_color_by(), "Atom"
            return self._CHECKPOINT_COLOR_BY + self._available_site_color_by(), "Single Colour"
        if atom_view:
            return self._ATOM_COLOR_BY, "Atom"
        return self._NORMAL_COLOR_BY, "Layer"

    def _single_color_rgb(self) -> np.ndarray:
        """The user-selected single colour as a (3,) float32 RGB array in [0, 1]."""
        return np.array(
            [self.single_color.redF(), self.single_color.greenF(), self.single_color.blueF()],
            dtype=np.float32,
        )

    def _centroid_colormap_colors(self) -> np.ndarray:
        """(N, 3) float32 colours for each centroid in the current VisualData, based on color_by.

        Used in Atoms/Unit Cell/Docking Atoms/Checkpoint Atoms modes when color_by != "Atom".
        """
        vd = self._visual_data
        col_idx = self.columnLabelToIndex.get(self.color_by, 2)
        if col_idx < -1:
            col_idx = 2

        if col_idx == -1:
            return vd.colors_uniform(self._single_color_rgb())

        if col_idx == 0:
            values = vd.mol_types.astype(np.float32)
        elif col_idx == 1:
            values = (
                vd.mol_numbers.astype(np.float32)
                if vd.mol_numbers is not None
                else np.arange(vd.n_centroids, dtype=np.float32)
            )
        elif col_idx == 2:
            if vd.layers is not None:
                raw_layers = vd.layers.astype(np.float32)
                valid = raw_layers[raw_layers < 99]
                max_layers = int(np.nanmax(valid)) if valid.size else 1
                return vd.colors_by_array(
                    raw_layers, self.availableColormaps[self.colormap], 1.0, float(max_layers)
                )
            values = np.zeros(vd.n_centroids, dtype=np.float32)
        elif col_idx == 3:
            values = np.arange(vd.n_centroids, dtype=np.float32)
        elif col_idx == 6:
            values = (
                vd.site_numbers.astype(np.float32)
                if vd.site_numbers is not None
                else np.arange(vd.n_centroids, dtype=np.float32)
            )
        elif col_idx == 7:
            values = (
                vd.energies.astype(np.float32)
                if vd.energies is not None
                else np.arange(vd.n_centroids, dtype=np.float32)
            )
        else:
            values = np.arange(vd.n_centroids, dtype=np.float32)

        return vd.colors_by_array(values, self.availableColormaps[self.colormap])

    def _docking_centroid_colors(self) -> np.ndarray:
        """Return (N, 3) float32 colours for docking centroids based on color_by."""
        n = len(self._docking_data.coords)
        if self.color_by == "Atom Type":
            mol_types = self._docking_data.mol_types.astype(np.float32)
            min_val, max_val = mol_types.min(), mol_types.max()
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized = (mol_types - min_val) / range_val
            return self.availableColormaps[self.colormap](normalized)[:, :3].astype(np.float32)
        else:
            # Default: Coordination Shell — respect per-shell overrides
            colors = np.zeros((n, 3), dtype=np.float32)
            for shell_id, default_color in self._docking_data.SHELL_COLORS.items():
                color = self._docking_shell_color_overrides.get(shell_id, default_color)
                mask = self._docking_data.shells == shell_id
                colors[mask] = color
            return colors

    def _update_docking_sphere_view(self):
        """Upload docking data to the sphere renderer for the Docking style."""
        self._sync_filter_signature()
        vd = self._docking_visual_data
        if vd is None or vd.n_centroids == 0:
            if self.sphere_renderer is not None:
                self.sphere_renderer.setPoints(np.zeros((0, 7), dtype=np.float32))
            return
        self.overlay.setVisible(False)
        if self.color_by == "Atom Type":
            colors = vd.colors_by_array(
                vd.mol_types.astype(np.float32), self.availableColormaps[self.colormap]
            )
        else:
            colors = vd.colors_by_shell(
                self._docking_data.SHELL_COLORS, self._docking_shell_color_overrides
            )
        varray = vd.sphere_vertices(colors)
        self._emit_docking_legend()
        keep = np.ones(len(varray), dtype=bool)
        slice_mask = self._slice_centroid_mask(vd.centroids)
        if slice_mask is not None:
            keep &= slice_mask
        color_mask = self._color_visibility_mask(colors)
        if color_mask is not None:
            keep &= color_mask
        if not np.all(keep):
            varray = varray[keep]
        self.sphere_renderer.setPoints(varray)
        if not self.viewInitialized:
            self.camera.fitToObject(vd.centroids)
            self.viewInitialized = True
        self._emit_rendered_count(len(varray), "Points")

    # ------------------------------------------------------------------ docking
    def set_docking_data(self, docking_data):
        """Store docking data and refresh the view if a docking style is active."""
        self._docking_data = docking_data
        if docking_data is not None and not docking_data.empty:
            self._docking_visual_data = VisualData.from_docking(
                docking_data, self._mol_templates, self._mol_crystallography
            )
        else:
            self._docking_visual_data = None
        if self.vis_mode == "Docking":
            self.initGeometry()
        self.update()

    # --------------------------------------------------------------- checkpoint
    def set_checkpoint(self, checkpoint: Checkpoint):
        """Store a Checkpoint object and refresh if a Checkpoint style is active.

        If the worker already expanded the grid into a VisualData (attached as
        ``checkpoint.prebuilt_visual_data``), reuse it and only attach templates
        here — the expensive argwhere/frac_to_cart stays off the GUI thread.
        """
        self._checkpoint = checkpoint
        # Only the edges are expanded at load (fast, memory-light). The full grid —
        # edges + interior — is expanded lazily the first time middle cells are
        # switched on (see _select_checkpoint_vd), then cached for instant toggling.
        self._checkpoint_vd_full = None
        if checkpoint is not None:
            cryst = checkpoint.crystallography or self._mol_crystallography
            prebuilt = getattr(checkpoint, "prebuilt_visual_data", None)
            if prebuilt is not None:
                if prebuilt.templates is None and self._mol_templates and cryst is not None:
                    prebuilt.templates = VisualData._build_cart_templates(
                        self._mol_templates, cryst
                    )
                self._checkpoint_vd_edges = prebuilt
            elif cryst is not None:
                self._checkpoint_vd_edges = VisualData.from_checkpoint(
                    checkpoint, cryst, self._mol_templates
                )
            else:
                self._checkpoint_vd_edges = None
        else:
            self._checkpoint_vd_edges = None
        self._select_checkpoint_vd()
        if self.vis_mode == "Checkpoint":
            self.initGeometry()
        self.update()

    def _select_checkpoint_vd(self):
        """Point ``_checkpoint_visual_data`` at the full or edges-only set.

        Only chooses among already-built VisualData — it never expands the grid.
        When ``show_middle`` is on but the full set isn't cached yet, it stays on
        edges; the caller is expected to build the full set off-thread and install
        it via :meth:`apply_checkpoint_full_vd`.
        """
        edges = self._checkpoint_vd_edges
        if edges is None:
            self._checkpoint_visual_data = None
            self._checkpoint_coords = np.zeros((0, 3), dtype=np.float64)
            self._checkpoint_center = np.zeros(3, dtype=np.float64)
            return
        show_middle = bool(getattr(self._checkpoint, "show_middle", False))
        if show_middle and self._checkpoint_vd_full is not None:
            vd = self._checkpoint_vd_full
        else:
            vd = edges
        self._checkpoint_visual_data = vd
        self._checkpoint_coords = vd.centroids
        self._checkpoint_center = np.zeros(3, dtype=np.float64)

    def _checkpoint_site_colors(self, vd, label: str) -> np.ndarray:
        """Per-centroid colours from a site-analysis field (coordination, energy …)."""
        site_map = self._site_metadata.get(label, {})
        cmap_fn = self.availableColormaps[self.colormap]
        return vd.colors_by_site_metadata(site_map, cmap_fn)

    def _checkpoint_hull_points(self) -> np.ndarray | None:
        """Surface-only positions for the checkpoint convex hull.

        Interior (middle) cells always lie on the segment between their strip
        block's edge cells, so the hull of the edges-only set is identical to the
        hull of the full grid — use it even when middle cells are shown.
        """
        evd = self._checkpoint_vd_edges
        if evd is None:
            return None
        points = evd.centroids
        slice_mask = self._slice_centroid_mask(points)
        if slice_mask is not None:
            points = points[slice_mask]
        return points

    def _update_checkpoint_view(self):
        """Upload checkpoint grid points to the renderer for the active style."""
        self._sync_filter_signature()
        vd = self._checkpoint_visual_data
        if vd is None or vd.n_centroids == 0:
            if self.sphere_renderer is not None:
                self._upload_centroid_varray(None)
            return
        self.overlay.setVisible(False)
        self._emit_checkpoint_legend(atom_view=False)

        if self.render_option == "Convex Hull":
            # Only the hull mesh is drawn — skip building the (N, 7) colour array,
            # which at full-grid scale costs gigabytes for nothing.
            points = self._checkpoint_hull_points()
            self._upload_centroid_varray(None, hull_points=points)
            if not self.viewInitialized:
                self.camera.fitToObject(vd.centroids)
                self.viewInitialized = True
            self._emit_rendered_count(len(points) if points is not None else 0, "Points")
            return

        if self.color_by in self._SITE_METADATA_COLOR_BY and self._site_metadata.get(self.color_by):
            colors = self._checkpoint_site_colors(vd, self.color_by)
        elif self.color_by == "Z Layer":
            colors = vd.colors_by_z()
        else:
            colors = vd.colors_uniform(self._single_color_rgb())
        varray = vd.sphere_vertices(colors)
        keep = np.ones(len(varray), dtype=bool)
        slice_mask = self._slice_centroid_mask(vd.centroids)
        if slice_mask is not None:
            keep &= slice_mask
        color_mask = self._color_visibility_mask(colors)
        if color_mask is not None:
            keep &= color_mask
        if not np.all(keep):
            varray = varray[keep]
        self._upload_centroid_varray(varray)
        if not self.viewInitialized:
            self.camera.fitToObject(vd.centroids)
            self.viewInitialized = True
        self._emit_rendered_count(len(varray), "Points")

    def _update_checkpoint_atom_view(self):
        """Build atom/bond instances from checkpoint grid + mol templates and upload to GPU."""
        self._sync_filter_signature()
        vd = self._checkpoint_visual_data
        if vd is None or vd.n_centroids == 0:
            if self.atom_renderer is not None:
                self.atom_renderer.setPoints(np.zeros((0, 8), dtype=np.float32))
            if self.bond_renderer is not None:
                self.bond_renderer.setBonds(None)
            return
        if not vd.templates:
            logger.warning("Checkpoint Atoms: no mol templates available")
            return

        self.overlay.setVisible(False)
        use_atom_colors = self.color_by == "Atom"
        if use_atom_colors:
            centroid_colors = None
        elif self.color_by in self._SITE_METADATA_COLOR_BY and self._site_metadata.get(
            self.color_by
        ):
            centroid_colors = self._checkpoint_site_colors(vd, self.color_by)
        else:
            n_tiles = self._checkpoint.n_tiles if self._checkpoint else 1
            tile_palette = cm.tab10(np.linspace(0, 1, max(n_tiles, 1)))[:, :3].astype(np.float32)
            centroid_colors = vd.colors_by_tile(tile_palette)

        atom_arr, bond_arr = vd.atom_vertices(
            centroid_colors=centroid_colors,
            use_atom_colors=use_atom_colors,
            color_overrides=self._atom_color_overrides or None,
            radius_overrides=self._atom_radius_overrides or None,
            bond_radius=self._bond_radius,
            slice_planes=self._active_slice_planes() or None,
        )

        if len(atom_arr) == 0:
            logger.warning("Checkpoint Atoms: no instances generated — check tile/template mapping")
            return

        if not self.viewInitialized:
            self.camera.fitToObject(atom_arr[:, :3])
            self.viewInitialized = True

        self._emit_checkpoint_legend(atom_view=True)
        atom_arr, bond_arr = self._apply_atom_color_filter(atom_arr, bond_arr)
        self.atom_renderer.setPoints(atom_arr)
        self.bond_renderer.setBonds(bond_arr)
        self._emit_rendered_count(len(atom_arr), "Atoms")

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint is not None

    def request_show_middle(self, show_middle: bool) -> bool:
        """Set the middle-cell visibility; return True if a background build is needed.

        Turning middle cells *off*, or *on* when the full grid is already cached, is
        an instant swap done here. Turning them *on* for the first time needs the
        full grid expanded — too heavy for the GUI thread — so this returns True and
        leaves the view on edges; the caller expands off-thread and installs the
        result via :meth:`apply_checkpoint_full_vd`.
        """
        if self._checkpoint is None:
            return False
        self._checkpoint.show_middle = show_middle
        if show_middle and self._checkpoint_vd_full is None:
            return True
        self._select_checkpoint_vd()
        if self.vis_mode == "Checkpoint":
            self.initGeometry()
        return False

    def apply_checkpoint_full_vd(self, vd: VisualData):
        """Install a background-expanded full-grid VisualData and refresh the view."""
        if vd is None or self._checkpoint is None:
            return
        cryst = self._checkpoint.crystallography or self._mol_crystallography
        if vd.templates is None and self._mol_templates and cryst is not None:
            vd.templates = VisualData._build_cart_templates(self._mol_templates, cryst)
        self._checkpoint_vd_full = vd
        self._select_checkpoint_vd()
        if self.vis_mode == "Checkpoint":
            self.initGeometry()
        self.update()

    def _update_docking_atom_view(self):
        """Build atom/bond instances from docking centroids and upload to GPU."""
        self._sync_filter_signature()
        vd = self._docking_visual_data
        if vd is None or vd.n_centroids == 0:
            if self.atom_renderer is not None:
                self.atom_renderer.setPoints(np.zeros((0, 8), dtype=np.float32))
            if self.bond_renderer is not None:
                self.bond_renderer.setBonds(None)
            return
        if not vd.templates:
            logger.warning("Docking Atoms requested but no molecular data available")
            return

        self.overlay.setVisible(False)
        use_atom_colors = self.color_by == "Atom"
        if use_atom_colors:
            centroid_colors = None
        elif self.color_by == "Atom Type":
            centroid_colors = vd.colors_by_array(
                vd.mol_types.astype(np.float32), self.availableColormaps[self.colormap]
            )
        else:
            centroid_colors = vd.colors_by_shell(
                self._docking_data.SHELL_COLORS, self._docking_shell_color_overrides
            )

        atom_arr, bond_arr = vd.atom_vertices(
            centroid_colors=centroid_colors,
            use_atom_colors=use_atom_colors,
            color_overrides=self._atom_color_overrides or None,
            radius_overrides=self._atom_radius_overrides or None,
            bond_radius=self._bond_radius,
            slice_planes=self._active_slice_planes() or None,
        )

        if len(atom_arr) == 0:
            logger.warning("No docking atom instances generated")
            return

        if not self.viewInitialized:
            self.camera.fitToObject(atom_arr[:, :3])
            self.viewInitialized = True

        self._emit_docking_legend()
        atom_arr, bond_arr = self._apply_atom_color_filter(atom_arr, bond_arr)
        self.atom_renderer.setPoints(atom_arr)
        self.bond_renderer.setBonds(bond_arr)
        self._emit_rendered_count(len(atom_arr), "Atoms")

    def initializeGL(self):
        logger.debug("Initialized OpenGL, version info: %s", self.context().format().version())
        debug = False
        if debug:
            self.logger = QOpenGLDebugLogger(self.context())
            if self.logger.initialize():
                self.logger.messageLogged.connect(self.handleLoggedMessage)
            else:
                ext = self.context().hasExtension(QtCore.QByteArray("GL_KHR_debug"))
                logger.debug("Debug logger not initialized, have extension GL_KHR_debug: %s", ext)

        color = self.backgroundColor
        gl = self.context().extraFunctions()
        self.point_cloud_renderer = SimplePointRenderer()
        self.sphere_renderer = SphereRenderer(gl)
        self.sphere_selection_renderer = SphereSelectionRenderer()
        self.mesh_renderer = MeshRenderer(gl)
        self.line_renderer = LineRenderer(gl)
        self.axes_renderer = AxesRenderer()
        self.direction_renderer = DirectionRenderer()
        self.plane_renderer = PlaneRenderer()
        self.atom_renderer = AtomRenderer(gl)
        self.bond_renderer = BondRenderer(gl)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glClearColor(color.redF(), color.greenF(), color.blueF(), 1)

    def handleLoggedMessage(self, message):
        logger.debug(
            "Source: %s, Type: %s, Message: %s",
            message.source(),
            message.type(),
            message.message(),
        )

    def _draw_points(self, gl, uniforms):
        if self.point_cloud_renderer.numberOfPoints() <= 0:
            return
        self.point_cloud_renderer.bind()
        self.point_cloud_renderer.setUniforms(**uniforms)

        self.point_cloud_renderer.draw(gl)
        self.point_cloud_renderer.release()

    def _draw_spheres(self, gl, uniforms):
        if self.sphere_renderer.numberOfInstances() <= 0:
            return
        self.sphere_renderer.bind(gl)
        self.sphere_renderer.setUniforms(**uniforms)

        self.sphere_renderer.draw(gl)
        self.sphere_renderer.release()

    def _draw_mesh(self, gl, uniforms):
        if self.mesh_renderer.numberOfVertices() <= 0:
            return
        self.mesh_renderer.bind(gl)
        self.mesh_renderer.setUniforms(**uniforms)

        self.mesh_renderer.draw(gl)
        self.mesh_renderer.release()

    def _draw_lines(self, gl, uniforms):
        if self.line_renderer.numberOfVertices() <= 0:
            return
        self.line_renderer.bind(gl)
        self.line_renderer.setUniforms(**uniforms)

        self.line_renderer.draw(gl)
        self.line_renderer.release()

    def _draw_atoms(self, gl, uniforms):
        if self.atom_renderer is None or self.atom_renderer.numberOfInstances() <= 0:
            return
        self.atom_renderer.bind(gl)
        self.atom_renderer.setUniforms(**uniforms)
        self.atom_renderer.draw(gl)
        self.atom_renderer.release()

    def _draw_bonds(self, gl, uniforms):
        if self.bond_renderer is None or self.bond_renderer.numberOfInstances() <= 0:
            return
        self.bond_renderer.bind(gl)
        self.bond_renderer.setUniforms(**uniforms)
        self.bond_renderer.draw(gl)
        self.bond_renderer.release()

    # ------------------------------------------------------------------
    # Atom / molecule view
    # ------------------------------------------------------------------

    def set_molecular_data(self, mol_templates, crystallography):
        """Store molecule templates and rebuild VisualData with correct scaling and templates."""
        self._mol_templates = mol_templates
        self._mol_crystallography = crystallography

        rebuilt = False

        if self._checkpoint is not None:
            cryst = self._checkpoint.crystallography or crystallography
            self._checkpoint_visual_data = VisualData.from_checkpoint(
                self._checkpoint, cryst, mol_templates
            )
            rebuilt = True

        if self._docking_data is not None and not self._docking_data.empty:
            self._docking_visual_data = VisualData.from_docking(
                self._docking_data, mol_templates, crystallography
            )
            rebuilt = True

        if (
            self._visual_data is not None
            and self._visual_data.source == "xyz"
            and self._visual_data._raw is not None
        ):
            self._visual_data = VisualData.from_xyz(
                self._visual_data._raw, mol_templates, crystallography
            )
            rebuilt = True

        if not rebuilt:
            return

        self.viewInitialized = False
        self.initGeometry()

    def _resolved_atom_colors_radii(self, tmpl: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (colors, radii) arrays for a template, with any user overrides applied."""
        symbols = tmpl["symbols"]  # list[str], one per atom
        colors = tmpl["colors"].copy()
        radii = tmpl["radii"].copy()
        for j, sym in enumerate(symbols):
            if sym in self._atom_color_overrides:
                colors[j] = self._atom_color_overrides[sym]
            if sym in self._atom_radius_overrides:
                radii[j] = self._atom_radius_overrides[sym]
        return colors, radii

    def _update_atom_view(self):
        """Compute atom and bond instances from current centroids and upload to GPU."""
        self._sync_filter_signature()
        vd = self._visual_data
        if vd is None or vd.n_centroids == 0:
            if self.atom_renderer is not None:
                self.atom_renderer.setPoints(np.zeros((0, 8), dtype=np.float32))
            if self.bond_renderer is not None:
                self.bond_renderer.setBonds(None)
            return
        if not vd.templates:
            logger.warning("Atom view requested but no molecular data available")
            return

        self.overlay.setVisible(False)
        use_atom_colors = self.color_by == "Atom"
        centroid_colors = None if use_atom_colors else self._centroid_colormap_colors()

        atom_arr, bond_arr = vd.atom_vertices(
            centroid_colors=centroid_colors,
            use_atom_colors=use_atom_colors,
            color_overrides=self._atom_color_overrides or None,
            radius_overrides=self._atom_radius_overrides or None,
            bond_radius=self._bond_radius,
            selected_indices=self._selected_points or None,
            slice_planes=self._active_slice_planes() or None,
            deleted_indices=self._deleted_points or None,
        )

        if len(atom_arr) == 0:
            logger.warning("No atom instances generated — check molecule type mapping")
            return

        if not self.viewInitialized:
            self.camera.fitToObject(atom_arr[:, :3])
            self.viewInitialized = True

        self._emit_atom_legend()
        atom_arr, bond_arr = self._apply_atom_color_filter(atom_arr, bond_arr)
        self.atom_renderer.setPoints(atom_arr)
        self.bond_renderer.setBonds(bond_arr)
        self._emit_rendered_count(len(atom_arr), "Atoms")

    def set_atom_overrides(
        self,
        color_overrides: dict[str, tuple[float, float, float]],
        radius_overrides: dict[str, float],
        bond_radius: float,
    ):
        """Apply per-element color / radius overrides and bond radius, then redraw."""
        self._atom_color_overrides = color_overrides
        self._atom_radius_overrides = radius_overrides
        self._bond_radius = bond_radius
        if self.is_atom_view:
            self.initGeometry()
            self.update()

    def set_legend_element_color(self, symbol: str, color: tuple[float, float, float] | None):
        """Override or reset the colour for an element symbol, then redraw."""
        if color is None:
            self._atom_color_overrides.pop(symbol, None)
        else:
            self._atom_color_overrides[symbol] = color
        if self.is_atom_view:
            self.initGeometry()

    def set_legend_shell_color(self, shell_id: int, color: tuple[float, float, float] | None):
        """Override or reset the colour for a docking coordination shell, then redraw."""
        if color is None:
            self._docking_shell_color_overrides.pop(shell_id, None)
        else:
            self._docking_shell_color_overrides[shell_id] = color
        if self.vis_mode == "Docking":
            self.initGeometry()

    def reset_legend_colors(self):
        """Clear all legend-driven colour overrides (element and shell) and redraw."""
        self._atom_color_overrides.clear()
        self._docking_shell_color_overrides.clear()
        if self.is_atom_view or self.vis_mode == "Docking":
            self.initGeometry()

    def get_visible_elements(self) -> list[str]:
        """Return sorted list of unique element symbols in the currently loaded templates."""
        if not (self._visual_data and self._visual_data.templates):
            return []
        symbols: set[str] = set()
        for tmpl in self._visual_data.templates.values():
            symbols.update(tmpl.get("symbols", []))
        return sorted(symbols)

    def get_bond_summary(self) -> dict[tuple[str, str], int]:
        """Return {(sym_a, sym_b): count} for all bonds across all molecule templates.

        Each pair is stored in sorted order so ('C','H') not ('H','C').
        """
        counts: dict[tuple[str, str], int] = {}
        if not (self._visual_data and self._visual_data.templates):
            return counts
        for tmpl in self._visual_data.templates.values():
            syms = tmpl.get("symbols", [])
            for a1, a2 in tmpl.get("bonds", []):
                if a1 < len(syms) and a2 < len(syms):
                    pair = tuple(sorted([syms[a1], syms[a2]]))
                    counts[pair] = counts.get(pair, 0) + 1
        return counts

    def get_atom_overrides(self) -> tuple[dict, dict, float]:
        """Return current (color_overrides, radius_overrides, bond_radius)."""
        return (
            dict(self._atom_color_overrides),
            dict(self._atom_radius_overrides),
            self._bond_radius,
        )

    def draw(self, gl):
        from PySide6.QtGui import QMatrix4x4, QVector2D

        mvp = self.camera.modelViewProjectionMatrix(self.aspect_ratio)
        view = self.camera.viewMatrix()
        proj = self.camera.projectionMatrix(self.aspect_ratio)
        modelView = self.camera.modelViewMatrix()
        model = self.camera.modelMatrix()
        axes = QMatrix4x4()
        screen_size = QVector2D(*self.screen_size)

        uniforms = {
            "u_modelMat": model,
            "u_modelRotMat": self.camera.modelRotationMatrix(),
            "u_viewMat": view,
            "u_modelViewProjectionMat": mvp,
            "u_pointSize": self.point_size,
            "u_axesMat": axes,
            "u_screenSize": screen_size,
            "u_projectionMat": proj,
            "u_modelViewMat": modelView,
            "u_scale": self.camera.scale,
            "u_lineScale": 2.0,
            **self.render_settings.shader_uniforms(self.camera.perspectiveProjection),
        }

        if self.is_atom_view:
            self._draw_bonds(gl, uniforms)
            self._draw_atoms(gl, uniforms)
        elif self.render_option == "Points":
            self._draw_points(gl, uniforms)
        elif self.render_option == "Convex Hull":
            self._draw_mesh(gl, uniforms)
            if self.show_mesh_edges:
                self._draw_lines(gl, uniforms)
        else:
            self._draw_spheres(gl, uniforms)

        self.axes_renderer.bind()
        self.axes_renderer.setUniforms(**uniforms)
        self.axes_renderer.draw(gl)
        self.axes_renderer.release()

        # Draw directions and planes with alpha blending
        self._draw_directions_and_planes(gl, uniforms)

        # Draw sphere selection overlay last (transparent, on top of everything)
        if self._sphere_sel_active and self._sphere_sel_radius > 0:
            self._draw_sphere_selection(gl, uniforms)

    def _draw_directions_and_planes(self, gl, uniforms):
        """Draw crystallographic directions and planes with transparency support."""
        from OpenGL.GL import GL_CULL_FACE, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA

        has_directions = (
            self.direction_renderer is not None and self.direction_renderer.numberOfPoints() > 0
        )
        has_planes = self.plane_renderer is not None and self.plane_renderer.numberOfVertices() > 0

        if not has_directions and not has_planes:
            return

        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if has_directions:
            if self.direction_renderer.directions:
                self.direction_renderer.draw_directions(
                    gl, uniforms, self.direction_renderer.directions
                )
            else:
                self.direction_renderer.bind()
                self.direction_renderer.setUniforms(**uniforms)
                self.direction_renderer.draw(gl)
                self.direction_renderer.release()

        if has_planes:
            gl.glDisable(GL_CULL_FACE)
            self.plane_renderer.bind()
            self.plane_renderer.setUniforms(**uniforms)
            self.plane_renderer.draw(gl)
            self.plane_renderer.release()

        gl.glDisable(GL_BLEND)

    def paintGL(self):
        gl = self.context().extraFunctions()
        # Restore GL state that QPainter may have changed in the previous frame
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDisable(GL_BLEND)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.draw(gl)

        # Draw text labels using QPainter overlay
        self._draw_axis_labels()

    def _draw_axis_labels(self):
        """Draw axis labels using QPainter overlay without affecting OpenGL state."""
        if not hasattr(self, "axes_renderer") or not self.axes_renderer:
            return

        # Check if labels should be shown
        if not self.axes_renderer.show_labels:
            return

        # Begin native painting - this properly manages OpenGL state
        painter = QPainter(self)
        painter.beginNativePainting()
        painter.endNativePainting()

        # Now do 2D painting
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)

        # Get axis endpoints from axes renderer
        endpoints = self.axes_renderer.get_axis_endpoints()

        # Determine label color mode
        use_axes_color = self.axes_renderer.label_color_same_as_axes
        custom_label_color = self.axes_renderer.label_color

        # Get MVP matrix for projection
        mvp = self.camera.modelViewProjectionMatrix(self.aspect_ratio)

        # Get origin screen position for calculating label offsets
        origin_screen = self._project_to_screen((0, 0, 0), mvp)

        for endpoint in endpoints:
            # Project 3D position to screen coordinates
            pos_3d = endpoint["position"]
            screen_pos = self._project_to_screen(pos_3d, mvp)

            if screen_pos:
                # Set text color based on settings
                if use_axes_color:
                    # Use the axis color
                    color = endpoint["color"]
                else:
                    # Use custom label color (default black)
                    color = custom_label_color

                painter.setPen(
                    QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                )

                # Calculate offset direction from origin to endpoint in screen space
                # This ensures labels are positioned away from the cylinder
                offset_x, offset_y = 8, 0  # Default offset
                if origin_screen:
                    dx = screen_pos[0] - origin_screen[0]
                    dy = screen_pos[1] - origin_screen[1]
                    length = (dx * dx + dy * dy) ** 0.5
                    if length > 0.001:
                        # Normalize and scale the offset
                        offset_x = dx / length * 15
                        offset_y = dy / length * 15

                # Draw the label offset in the direction of the axis
                painter.drawText(
                    QPoint(int(screen_pos[0] + offset_x), int(screen_pos[1] + offset_y)),
                    endpoint["label"],
                )

        painter.end()

    def _project_to_screen(self, pos_3d, mvp):
        """Project a 3D position to screen coordinates."""
        from PySide6.QtGui import QVector3D, QVector4D

        # Apply rotation-only view transform (ignore camera position/target translation)
        # so axes labels respond only to orientation, not to crystal translation.
        view = self.camera.viewMatrix()
        view.setColumn(3, QVector4D(0.0, 0.0, 0.0, 1.0))
        model_rot = self.camera.modelRotationMatrix()
        world_pos = model_rot.map(QVector3D(pos_3d[0], pos_3d[1], pos_3d[2]))
        rotated = view.map(world_pos) * 0.1

        # Offset to corner (matching the geometry shader)
        screen_offset = QVector3D(0.8, 0.8, 0.0)
        rotated -= screen_offset

        # Convert to clip space
        clip_pos = QVector4D(rotated.x(), rotated.y(), rotated.z(), 1.0)

        # Get NDC coordinates
        ndc_x = clip_pos.x()
        ndc_y = clip_pos.y()

        # Convert NDC to screen coordinates
        screen_x = (ndc_x + 1.0) * 0.5 * self.width()
        screen_y = (1.0 - ndc_y) * 0.5 * self.height()

        return (screen_x, screen_y)

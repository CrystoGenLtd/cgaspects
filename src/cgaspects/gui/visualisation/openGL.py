import logging
from pathlib import Path

import numpy as np
import trimesh
from matplotlib import cm
from OpenGL.GL import (GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                       GL_DEPTH_TEST)
from PySide6 import QtCore
from PySide6.QtCore import QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QVector3D
from PySide6.QtOpenGL import QOpenGLDebugLogger, QOpenGLFramebufferObject
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from scipy.spatial import ConvexHull

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
from .sphere_renderer import SphereRenderer
from .sphere_selection_renderer import SphereSelectionRenderer
from .unit_cell_renderer import UnitCellRenderer

logger = logging.getLogger("CA:OpenGL")


class VisualisationWidget(QOpenGLWidget):
    style = "Spheres"
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
    styleChanged = Signal(str)   # Emitted when the render style changes (e.g. "Atoms", "Spheres")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lastMousePosition = QtCore.QPoint()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.camera = Camera()
        self.restrict_axis = None
        self.rotation_lock_axis = None  # "x", "y", "z", or None; toggled by menu actions
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
        self._mol_templates = None          # dict[int, MolTemplate] from structure file
        self._mol_crystallography = None    # Crystallography used to convert frac → cart
        self._mol_cart_templates = {}       # precomputed per-type: (cart_atoms, symbols, bonds)
        self.atom_renderer = None
        self.bond_renderer = None
        self.unit_cell_renderer = None

        # Per-element overrides (set via Atom Mode Settings dialog)
        self._atom_color_overrides: dict[str, tuple[float, float, float]] = {}
        self._atom_radius_overrides: dict[str, float] = {}  # absolute VdW radius in Å
        self._bond_radius: float = 0.20                     # bond cylinder radius in Å

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

        self.xyz = None
        self.crystal = None
        # self.object = 0

        self.colormap = "Viridis"
        self.color_by = "Layer"
        self.single_color = QColor(128, 128, 128)  # Default grey color
        self._legend_info = None

        self.viewInitialized = False
        self.point_size = 6.0
        self.point_type = "Point"
        self.backgroundColor = QColor(Qt.white)

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
            "Site Number": 6,
            "Particle Energy": 7,
        }

        self.availableColumns = {}

        # Site highlighting - support multiple groups
        self.highlight_groups = []  # List of (site_set, color) tuples
        self.background_color_override = None  # Background color for non-highlighted sites

    def pass_XYZ(self, xyz):
        self.xyz = xyz
        logger.debug("XYZ coordinates passed on OpenGL widget")

    def pass_XYZ_list(self, xyz_path_list):
        self.xyz_path_list = xyz_path_list
        logger.info("XYZ file paths (list) passed to OpenGL widget")

    def get_XYZ_from_list(self, value):
        if self.sim_num != value:
            self.sim_num = value
            self.crystal = CrystalCloud.from_file(self.xyz_path_list[value])
            if self.crystal.empty:
                self.showNoDataOverlay()
                return
            self.xyz = self.crystal.get_raw_frame_coords(0)
            self.initGeometry()
            self.update()

    def showNoDataOverlay(self):
        """Show an overlay message when there are no points to display."""
        self.overlay.setText("No point data available for this simulation")
        self.overlay.setVisible(True)
        self.update()

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
        if self.style in ("Atoms", "Unit Cell") and self.xyz is not None:
            from dataclasses import replace
            a = self._a_axis()
            cart_extent = self._cart_max_extent()
            dirs = [
                replace(d, origin=tuple((np.array(d.origin) * a).tolist()))
                for d in self._raw_directions
            ]
            self.direction_renderer.set_directions(dirs, self._directions_crystallography, cart_extent)
        else:
            self.direction_renderer.set_directions(
                self._raw_directions, self._directions_crystallography, self._directions_max_extent
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
        visible = [p for p in self._raw_planes if p.visible]
        if self.style in ("Atoms", "Unit Cell") and self.xyz is not None:
            from dataclasses import replace
            a = self._a_axis()
            cart_extent = self._cart_max_extent()
            converted = [
                replace(p,
                        origin=tuple((np.array(p.origin) * a).tolist()),
                        size=p.size_relative * cart_extent)
                for p in visible
            ]
            self.plane_renderer.set_planes(converted, self._planes_crystallography)
            self.update()
        else:
            self.plane_renderer.set_planes(visible, self._planes_crystallography)
            if self.xyz is not None:
                self.initGeometry()
            else:
                self.update()

    def _a_axis(self):
        """Return the a-axis length in Å, or 1.0 if unavailable."""
        if self._mol_crystallography is not None and self._mol_crystallography.cell is not None:
            return float(self._mol_crystallography.cell.a)
        return 1.0

    def _cart_max_extent(self):
        """Half-range of the crystal in Cartesian Å (used when in Atoms/Unit Cell mode).

        XYZ coordinates are in units of the a-axis length, so multiply by a.
        """
        if self.xyz is None:
            return self._directions_max_extent
        frac = self.xyz[:, 3:6].astype(np.float64)
        extents = frac.max(axis=0) - frac.min(axis=0)
        return float(extents.max()) / 2.0 * self._a_axis()

    def _frac_max_extent(self):
        """Half-range of the crystal in raw XYZ lattice units (units of a)."""
        if self.xyz is None:
            return 1.0
        frac = self.xyz[:, 3:6].astype(np.float64)
        extents = frac.max(axis=0) - frac.min(axis=0)
        return float(extents.max()) / 2.0

    _ATOM_STYLES = frozenset(("Atoms", "Unit Cell"))
    _FRAC_STYLES = frozenset(("Points", "Spheres", "Convex Hull"))

    def _rescale_camera_for_style(self, old_style, new_style):
        """Rescale camera zoom when switching between XYZ-unit and Cartesian spaces.

        XYZ coordinates are in units of the a-axis length. Toggling to Atoms mode
        scales all world positions by a, so the camera scale must divide by a
        (and multiply by a when returning to point-cloud mode).
        Preserves camera orientation (position, target, up/right vectors).
        """
        if self._mol_crystallography is None or self.xyz is None:
            return
        going_to_cart = old_style in self._FRAC_STYLES and new_style in self._ATOM_STYLES
        going_to_frac = old_style in self._ATOM_STYLES and new_style in self._FRAC_STYLES
        if not going_to_cart and not going_to_frac:
            return
        a = self._a_axis()
        if a < 1e-10:
            return
        if going_to_cart:
            self.camera.scale /= a
        else:
            self.camera.scale *= a

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

    def saveRenderDialog(self):
        # First ask user what type of export they want
        export_options = ["2D Image (PNG)", "3D Mesh"]

        # Only allow 3D mesh export if not in Points mode
        if self.style == "Points":
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
        if self.xyz is None:
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

        if self.style == "Spheres":
            # Generate mesh from sphere instances with colors
            mesh = self._generateSphereMesh(subdivision_level=subdivision_level)
        elif self.style == "Convex Hull":
            # Use the existing convex hull mesh (without colors)
            mesh = self.mesh_renderer.mesh

        if mesh is None:
            raise ValueError(f"Cannot export mesh in '{self.style}' mode")

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

        if present_and_changed("Style", self.style):
            old_style = self.style
            self.style = kwargs["Style"]
            self.styleChanged.emit(self.style)
            self._rescale_camera_for_style(old_style, self.style)
            needs_reinit = True

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

        if "Axes Thickness" in kwargs:
            if self.axes_renderer is not None:
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
                    self._sphere_sel_center_world = self.xyz[point_idx, 3:6].copy()
                    self._sphere_sel_anchor_idx = point_idx
                    self._sphere_sel_radius = 0.0
                    self._sphere_sel_active = False
            # Plain left click: camera orbit only (handled in mouseMoveEvent)

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

        self.camera.orbit(dx, dy)
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

        # Set camera position along the negative direction, looking toward target
        distance = (self.camera.position - self.camera.target).length()
        self.camera.position = self.camera.target - direction * distance
        self.camera.up = up
        self.camera.right = QVector3D.crossProduct(up, direction).normalized()
        self.update()

    # ------------------------------------------------------------------
    # Public viewport action methods (wired to menu QActions)
    # ------------------------------------------------------------------

    def align_view_x(self): self._align_view_to_axis("x")
    def align_view_y(self): self._align_view_to_axis("y")
    def align_view_z(self): self._align_view_to_axis("z")
    def align_view_a(self): self._align_view_to_axis("a")
    def align_view_b(self): self._align_view_to_axis("b")
    def align_view_c(self): self._align_view_to_axis("c")

    def reset_view(self):
        self.camera.resetOrientation()
        self.update()

    def store_view(self):
        self.camera.storeOrientation()

    def increase_point_size(self):
        self.point_size = min(self.point_size + 1.0, 20.0)
        self.pointSizeChanged.emit(int(self.point_size))
        self.update()

    def decrease_point_size(self):
        self.point_size = max(self.point_size - 1.0, 0.5)
        self.pointSizeChanged.emit(int(self.point_size))
        self.update()

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
        if self.xyz is None or len(self.xyz) == 0:
            return None, None

        ray_origin, ray_direction = self._screen_to_ray(screen_x, screen_y)

        # Get point positions (columns 3:6 contain x, y, z)
        points = self.xyz[:, 3:6]

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
        if self.xyz is None or self._sphere_sel_center_world is None:
            return

        points = self.xyz[:, 3:6]
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
        if self.xyz is None or point_index is None or point_index >= len(self.xyz):
            return None

        row = self.xyz[point_index]
        data = {
            "position": (row[3], row[4], row[5]),
            "type": row[0],
            "number": row[1],
            "layer": row[2],
        }

        # Add optional columns if available
        if self.xyz.shape[1] > 6:
            data["site"] = row[6]
        if self.xyz.shape[1] > 7:
            data["energy"] = row[7]

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

    def get_active_xyz(self):
        """Get XYZ data excluding deleted points.

        Returns:
            np.ndarray: XYZ data with deleted points removed
        """
        if self.xyz is None:
            return None

        if not self._deleted_points:
            return self.xyz.copy()

        # Create mask for non-deleted points
        mask = np.ones(len(self.xyz), dtype=bool)
        for idx in self._deleted_points:
            if idx < len(mask):
                mask[idx] = False

        return self.xyz[mask].copy()

    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.lastMousePosition.x()
        dy = event.pos().y() - self.lastMousePosition.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            if self._sphere_sel_center_world is not None:
                # Shift+drag sphere selection: update radius and selection
                self._sphere_sel_active = True
                self._sphere_sel_radius = self._compute_sphere_radius(event.pos())
                self._update_sphere_selection()
            elif self.rotation_lock_axis:
                self.rotatePointCloud(dx, self.rotation_lock_axis)
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
        if event.button() == QtCore.Qt.LeftButton:
            if self._sphere_sel_center_world is not None:
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
        if self.xyz is None:
            return

        angle = np.radians(dx * self.camera.rotationSpeed)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Define rotation matrices for X, Y, and Z axes
        if axis == "x":
            rotation_matrix = np.array(
                [[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]], dtype=np.float32
            )
        elif axis == "y":
            rotation_matrix = np.array(
                [[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], dtype=np.float32
            )
        elif axis == "z":
            rotation_matrix = np.array(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32
            )
        else:
            return

        # Apply rotation to the XYZ points
        points = self.xyz[:, 3:6]
        rotated_points = points @ rotation_matrix.T

        # Update the point cloud with rotated points
        self.xyz[:, 3:6] = rotated_points
        self.initGeometry()

    def initGeometry(self):
        # self.update()
        if self.point_cloud_renderer is None:
            return

        if self.style == "Atoms":
            self._update_atom_view()
            self._apply_planes()
            self._apply_directions()
            self.update()
            return

        if self.style == "Unit Cell":
            self._update_unit_cell_view()
            self._apply_planes()
            self._apply_directions()
            self.update()
            return

        varray = self.updatePointCloudVertices()
        self.point_cloud_renderer.setPoints(varray)
        self.sphere_renderer.setPoints(varray)

        if self.style == "Convex Hull":
            hull = ConvexHull(varray[:, :3])
            mesh = trimesh.Trimesh(vertices=varray[:, :3], faces=hull.simplices)
            # can pass vertex colors here, but I wouldn't
            self.mesh_renderer.setMesh(mesh)

            if self.show_mesh_edges:
                self.line_renderer.setLines(self.mesh_renderer.getLines())

        self.update()

    def updatePointCloudVertices(self):
        self.overlay.setVisible(False)
        logger.debug("Loading Vertices")
        logger.debug(".XYZ shape: %s", self.xyz.shape[0])
        layers = self.xyz[:, 2]
        max_layers = int(np.nanmax(layers[layers < 99]))

        # Loading the point cloud from file
        def vis_pc(xyz, color_axis):
            pcd_points = xyz[:, 3:6]
            pcd_colors = None

            if xyz.shape[1] <= 6 and color_axis >= 6:
                logger.warning(
                    "Old CrystalGrower version! %s option not available for colouring.",
                    self.color_by,
                )
                color_axis = 3

            if color_axis == 3:
                axis_vis = np.arange(0, xyz.shape[0], dtype=np.float32)
            else:
                axis_vis = xyz[:, color_axis]

            if color_axis == 2:
                min_val = 1
                max_val = max_layers
            elif color_axis == -1:
                # Single color mode - use custom color
                min_val = 0
                max_val = 0
                axis_vis = np.zeros_like(axis_vis)
                # Convert QColor to RGB values in [0, 1] range
                single_color_rgb = np.array(
                    [
                        self.single_color.redF(),
                        self.single_color.greenF(),
                        self.single_color.blueF(),
                    ],
                    dtype=np.float32,
                )
            else:
                min_val = np.nanmin(axis_vis)
                max_val = np.nanmax(axis_vis)

            # Avoid division by zero in case all values are the same
            range_val = max_val - min_val if max_val != min_val else 1

            normalized_axis_vis = (axis_vis - min_val) / range_val

            if color_axis == -1:
                # Use the custom single color for all points
                pcd_colors = np.tile(single_color_rgb, (xyz.shape[0], 1))
                legend_rows = [(None, tuple(single_color_rgb.tolist()))]
            else:
                pcd_colors = self.availableColormaps[self.colormap](normalized_axis_vis)[:, 0:3]
                unique_vals = np.unique(axis_vis)
                norm_unique = (unique_vals - min_val) / range_val
                unique_rgb = self.availableColormaps[self.colormap](norm_unique)[:, :3]
                legend_rows = [(float(v), tuple(c.tolist())) for v, c in zip(unique_vals, unique_rgb)]

            self._legend_info = {
                "color_by": self.color_by,
                "colormap": self.colormap,
                "min_val": float(min_val),
                "max_val": float(max_val),
                "rows": legend_rows,
            }
            self.legendChanged.emit(self._legend_info)

            return (pcd_points, pcd_colors)

        points, colors = vis_pc(self.xyz, self.columnLabelToIndex[self.color_by])

        if not self.viewInitialized:
            self.camera.fitToObject(points)
            self.viewInitialized = True

        points = np.asarray(points).astype("float32")
        colors = np.asarray(colors).astype("float32")

        # Apply site highlighting if any groups are defined
        if self.highlight_groups and self.xyz.shape[1] > 6:
            # Column 6 is Site Number (0-indexed)
            site_numbers = self.xyz[:, 6]

            # First, apply background color override to all sites if specified
            if self.background_color_override is not None:
                colors[:] = self.background_color_override

            # Then apply highlight colors for each group (later groups override earlier ones)
            for site_set, highlight_color in self.highlight_groups:
                # Create a mask for particles belonging to sites in this group
                mask = np.isin(site_numbers, list(site_set))
                colors[mask] = highlight_color

        # Create selection flags array (1.0 for selected, 0.0 for not selected)
        selection_flags = np.zeros((len(points), 1), dtype=np.float32)
        if self._selected_points:
            for idx in self._selected_points:
                if idx < len(selection_flags):
                    selection_flags[idx] = 1.0

        # Build combined mask: exclude deleted points and apply active slice planes
        n_pts = len(points)
        combined_mask = np.ones(n_pts, dtype=bool)

        # Deleted-points mask
        for idx in self._deleted_points:
            if idx < n_pts:
                combined_mask[idx] = False

        # Slice-plane masks (applied in translated coordinate space)
        for plane in self._raw_planes:
            if not plane.slice_enabled:
                continue
            normal = np.array(plane.normal, dtype=np.float64)
            if plane.fractional and self._planes_crystallography is not None:
                normal = self._planes_crystallography.miller_to_cart_normal(normal)
            n_len = np.linalg.norm(normal)
            if n_len < 1e-10:
                continue
            normal /= n_len
            origin = np.array(plane.origin, dtype=np.float64)
            d = (points - origin.astype(np.float32)) @ normal.astype(np.float32)
            if plane.slice_two_sided:
                combined_mask &= np.abs(d) <= plane.slice_thickness / 2.0
            else:
                combined_mask &= d >= -plane.slice_thickness

        if not np.all(combined_mask):
            points = points[combined_mask]
            colors = colors[combined_mask]
            selection_flags = selection_flags[combined_mask]

        try:
            # Concatenate: position (3) + color (3) + selection (1) = 7 floats
            attributes = np.concatenate((points, colors, selection_flags), axis=1)

            return attributes
        except ValueError as exc:
            logger.error(
                "%s\n XYZ %s POINTS %s COLORS %s TYPE %s",
                exc,
                self.xyz.shape,
                points.shape,
                colors.shape,
                self.color_by,
            )
            return

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
        self.unit_cell_renderer = UnitCellRenderer(gl)
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

    def _draw_unit_cell(self, gl, uniforms):
        if self.unit_cell_renderer is None or self.unit_cell_renderer.numberOfVertices() <= 0:
            return
        self.unit_cell_renderer.bind()
        self.unit_cell_renderer.setUniforms(**uniforms)
        self.unit_cell_renderer.draw(gl)
        self.unit_cell_renderer.release()

    # ------------------------------------------------------------------
    # Atom / molecule view
    # ------------------------------------------------------------------

    def set_molecular_data(self, mol_templates, crystallography):
        """Store molecule templates and the crystallography object for frac→cart conversion.

        Parameters
        ----------
        mol_templates : dict[int, MolTemplate]
            Keyed by molecule type (column 0 in the XYZ data).
        crystallography : Crystallography
            Used to convert fractional coordinates to Cartesian.
        """
        self._mol_templates = mol_templates
        self._mol_crystallography = crystallography
        self._mol_cart_templates = {}
        if mol_templates and crystallography:
            self._precompute_mol_templates()
        if crystallography is not None and self.unit_cell_renderer is not None:
            self.unit_cell_renderer.set_cell(crystallography)

    def _precompute_mol_templates(self):
        """Convert template fractional coords to Cartesian and compute centroids."""
        from ...utils.periodic_table import get_atom_color, get_atom_radius

        for mol_type, tmpl in self._mol_templates.items():
            if not tmpl.atoms:
                continue
            frac = np.array([a.frac for a in tmpl.atoms], dtype=np.float64)
            cart = self._mol_crystallography.frac_to_cart(frac).astype(np.float32)
            centroid = cart.mean(axis=0)
            colors = np.array(
                [get_atom_color(a.symbol) for a in tmpl.atoms], dtype=np.float32
            )
            radii = np.array(
                [get_atom_radius(a.symbol) for a in tmpl.atoms], dtype=np.float32
            )
            self._mol_cart_templates[mol_type] = {
                "cart": cart,                                          # (N_atoms, 3)
                "centroid": centroid,                                  # (3,)
                "colors": colors,                                      # (N_atoms, 3)
                "radii": radii,                                        # (N_atoms,)
                "bonds": tmpl.bonds,                                   # list[(i, j)] 0-based
                "symbols": [a.symbol for a in tmpl.atoms],            # list[str]
            }

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
        if self._mol_crystallography is None or not self._mol_cart_templates:
            logger.warning("Atom view requested but no molecular data available")
            return
        if self.xyz is None:
            return

        atom_instances = []   # each row: [x,y,z, r,g,b, selected, vdw_radius]
        bond_instances = []   # each row: [sx,sy,sz, ex,ey,ez, r,g,b, cyl_radius]

        bond_r = self._bond_radius  # Ångströms, set directly by dialog

        mol_types = self.xyz[:, 0].astype(int)
        # XYZ coordinates are in units of the a-axis length; scale to Cartesian Å.
        a = self._a_axis()
        cart_positions = (self.xyz[:, 3:6].astype(np.float64) * a).astype(np.float32)

        for i, (mol_type, centroid_pos) in enumerate(zip(mol_types, cart_positions)):
            tmpl = self._mol_cart_templates.get(mol_type)
            if tmpl is None:
                continue

            colors, radii = self._resolved_atom_colors_radii(tmpl)

            # Shift template atoms so their centroid sits on centroid_pos (both in Cartesian)
            offset = centroid_pos - tmpl["centroid"]
            atom_positions = tmpl["cart"] + offset   # (N_atoms, 3)

            sel = 1.0 if i in self._selected_points else 0.0
            sel_col = np.full((len(atom_positions), 1), sel, dtype=np.float32)

            # [x,y,z, r,g,b, selected, radius]
            block = np.hstack([
                atom_positions,     # (N,3)
                colors,             # (N,3)
                sel_col,            # (N,1)
                radii[:, None],     # (N,1)
            ])
            atom_instances.append(block)

            # Bonds: two half-cylinders per bond
            for a1, a2 in tmpl["bonds"]:
                if a1 >= len(atom_positions) or a2 >= len(atom_positions):
                    continue
                p1 = atom_positions[a1]
                p2 = atom_positions[a2]
                mid = (p1 + p2) * 0.5
                c1 = colors[a1]
                c2 = colors[a2]
                bond_instances.append(np.concatenate([p1, mid, c1, [bond_r]]))
                bond_instances.append(np.concatenate([mid, p2, c2, [bond_r]]))

        if not atom_instances:
            logger.warning("No atom instances generated — check molecule type mapping")
            return

        atom_arr = np.vstack(atom_instances).astype(np.float32)
        bond_arr = np.array(bond_instances, dtype=np.float32) if bond_instances else None

        if not self.viewInitialized:
            self.camera.fitToObject(atom_arr[:, :3])
            self.viewInitialized = True

        self.atom_renderer.setPoints(atom_arr)
        self.bond_renderer.setBonds(bond_arr)

    def _update_unit_cell_view(self):
        """Render the template molecules at their unit-cell positions (no crystal replication)."""
        if not self._mol_cart_templates:
            logger.warning("Unit cell view requested but no molecular data available")
            return

        atom_instances = []
        bond_instances = []
        bond_r = self._bond_radius

        for tmpl in self._mol_cart_templates.values():
            colors, radii = self._resolved_atom_colors_radii(tmpl)
            atom_positions = tmpl["cart"]   # already in Cartesian, no offset

            sel_col = np.zeros((len(atom_positions), 1), dtype=np.float32)
            block = np.hstack([
                atom_positions,
                colors,
                sel_col,
                radii[:, None],
            ])
            atom_instances.append(block)

            for a1, a2 in tmpl["bonds"]:
                if a1 >= len(atom_positions) or a2 >= len(atom_positions):
                    continue
                p1 = atom_positions[a1]
                p2 = atom_positions[a2]
                mid = (p1 + p2) * 0.5
                bond_instances.append(np.concatenate([p1, mid, colors[a1], [bond_r]]))
                bond_instances.append(np.concatenate([mid, p2, colors[a2], [bond_r]]))

        if not atom_instances:
            return

        atom_arr = np.vstack(atom_instances).astype(np.float32)
        bond_arr = np.array(bond_instances, dtype=np.float32) if bond_instances else None

        if not self.viewInitialized:
            self.camera.fitToObject(atom_arr[:, :3])
            self.viewInitialized = True

        self.atom_renderer.setPoints(atom_arr)
        self.bond_renderer.setBonds(bond_arr)

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
        if self.style == "Atoms":
            self._update_atom_view()
            self.update()
        elif self.style == "Unit Cell":
            self._update_unit_cell_view()
            self.update()

    def get_visible_elements(self) -> list[str]:
        """Return sorted list of unique element symbols in the currently loaded templates."""
        symbols: set[str] = set()
        for tmpl in self._mol_cart_templates.values():
            symbols.update(tmpl.get("symbols", []))
        return sorted(symbols)

    def get_bond_summary(self) -> dict[tuple[str, str], int]:
        """Return {(sym_a, sym_b): count} for all bonds across all molecule templates.

        Each pair is stored in sorted order so ('C','H') not ('H','C').
        """
        counts: dict[tuple[str, str], int] = {}
        for tmpl in self._mol_cart_templates.values():
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

    def toggle_atom_view(self):
        """Toggle between centroid (Spheres) and atom view. Shift+V shortcut."""
        if self._mol_templates is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "No Molecular Data",
                "No structure file was found.\n"
                "Load a CrystalGrower simulation folder that includes a structure file.",
            )
            return
        old_style = self.style
        self.style = "Atoms" if self.style != "Atoms" else "Spheres"
        self.styleChanged.emit(self.style)
        self._rescale_camera_for_style(old_style, self.style)
        self.initGeometry()

    def draw(self, gl):
        from PySide6.QtGui import QMatrix4x4, QVector2D

        mvp = self.camera.modelViewProjectionMatrix(self.aspect_ratio)
        view = self.camera.viewMatrix()
        proj = self.camera.projectionMatrix(self.aspect_ratio)
        modelView = self.camera.modelViewMatrix()
        axes = QMatrix4x4()
        screen_size = QVector2D(*self.screen_size)

        uniforms = {
            "u_viewMat": view,
            "u_modelViewProjectionMat": mvp,
            "u_pointSize": self.point_size,
            "u_axesMat": axes,
            "u_screenSize": screen_size,
            "u_projectionMat": proj,
            "u_modelViewMat": modelView,
            "u_scale": self.camera.scale,
            "u_lineScale": 2.0,
        }

        if self.style == "Points":
            self._draw_points(gl, uniforms)
        elif self.style == "Spheres":
            self._draw_spheres(gl, uniforms)
        elif self.style == "Convex Hull":
            self._draw_mesh(gl, uniforms)
            if self.show_mesh_edges:
                self._draw_lines(gl, uniforms)
        elif self.style == "Atoms":
            self._draw_bonds(gl, uniforms)
            self._draw_atoms(gl, uniforms)
        elif self.style == "Unit Cell":
            self._draw_bonds(gl, uniforms)
            self._draw_atoms(gl, uniforms)
            self._draw_unit_cell(gl, uniforms)

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
        from OpenGL.GL import (GL_CULL_FACE, GL_ONE_MINUS_SRC_ALPHA,
                               GL_SRC_ALPHA)

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
        rotated = view.map(QVector3D(pos_3d[0], pos_3d[1], pos_3d[2])) * 0.1

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

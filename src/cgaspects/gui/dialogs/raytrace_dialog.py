"""Dialog for exporting/rendering the 3D viewport via POV-Ray or Tachyon.

"Match GL" mirrors the live material one-to-one; "Photoreal" enables a second
settings dialog (ambient occlusion, soft shadows, reflection, depth of field).
The scene file can be exported alone, or rendered immediately if the chosen
backend is on PATH.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..visualisation import raytrace_export as rt

logger = logging.getLogger("CA:RaytraceDialog")


class PhotorealSettingsDialog(QDialog):
    """Second-level settings for the Photoreal quality mode."""

    def __init__(self, options: rt.PhotorealOptions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photoreal Settings")
        self.setModal(True)
        self._opts = options

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.ao_check = QCheckBox()
        self.ao_check.setChecked(options.ambient_occlusion)
        self.ao_check.setToolTip("Radiosity (POV-Ray) / ambient occlusion (Tachyon)")
        form.addRow("Ambient occlusion:", self.ao_check)

        self.ao_samples = QSpinBox()
        self.ao_samples.setRange(8, 256)
        self.ao_samples.setValue(options.ao_samples)
        form.addRow("AO samples:", self.ao_samples)

        self.shadow_check = QCheckBox()
        self.shadow_check.setChecked(options.soft_shadows)
        form.addRow("Soft shadows:", self.shadow_check)

        self.shadow_soft = QDoubleSpinBox()
        self.shadow_soft.setRange(0.0, 5.0)
        self.shadow_soft.setSingleStep(0.1)
        self.shadow_soft.setValue(options.shadow_softness)
        form.addRow("Shadow softness:", self.shadow_soft)

        self.reflection = QDoubleSpinBox()
        self.reflection.setRange(0.0, 1.0)
        self.reflection.setSingleStep(0.05)
        self.reflection.setValue(options.reflection)
        self.reflection.setToolTip("Mirror reflectivity of sphere surfaces (0 = matte)")
        form.addRow("Reflection:", self.reflection)

        self.blur_check = QCheckBox()
        self.blur_check.setChecked(options.focal_blur)
        self.blur_check.setToolTip("Depth of field focused on the scene centre")
        form.addRow("Focal blur (DoF):", self.blur_check)

        self.aperture = QDoubleSpinBox()
        self.aperture.setRange(0.0, 1.0)
        self.aperture.setSingleStep(0.01)
        self.aperture.setDecimals(3)
        self.aperture.setValue(options.aperture)
        form.addRow("Aperture:", self.aperture)

        self.aa = QSpinBox()
        self.aa.setRange(1, 11)
        self.aa.setValue(options.antialiasing)
        form.addRow("Anti-aliasing:", self.aa)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self) -> rt.PhotorealOptions:
        return rt.PhotorealOptions(
            ambient_occlusion=self.ao_check.isChecked(),
            ao_samples=self.ao_samples.value(),
            soft_shadows=self.shadow_check.isChecked(),
            shadow_softness=self.shadow_soft.value(),
            reflection=self.reflection.value(),
            focal_blur=self.blur_check.isChecked(),
            aperture=self.aperture.value(),
            antialiasing=self.aa.value(),
        )


class RaytraceDialog(QDialog):
    """Configure and launch a POV-Ray / Tachyon render of the current view."""

    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ray-Trace Render (POV-Ray / Tachyon)")
        self.setModal(True)
        self._widget = widget
        self._photoreal = rt.PhotorealOptions()

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("POV-Ray", "povray")
        self.backend_combo.addItem("Tachyon", "tachyon")
        self.backend_combo.currentIndexChanged.connect(self._update_backend_status)
        form.addRow("Renderer:", self.backend_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Match GL settings", "match")
        self.quality_combo.addItem("Photoreal", "photoreal")
        self.quality_combo.currentIndexChanged.connect(self._update_quality)
        form.addRow("Quality:", self.quality_combo)

        self.photoreal_button = QPushButton("Photoreal Settings…")
        self.photoreal_button.setEnabled(False)
        self.photoreal_button.clicked.connect(self._edit_photoreal)
        form.addRow("", self.photoreal_button)

        # Resolution defaults to the current viewport size.
        w = max(int(widget.width()), 320)
        h = max(int(widget.height()), 240)
        res_row = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 16384)
        self.width_spin.setValue(w)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 16384)
        self.height_spin.setValue(h)
        res_row.addWidget(self.width_spin)
        res_row.addWidget(QLabel("×"))
        res_row.addWidget(self.height_spin)
        res_row.addStretch()
        form.addRow("Resolution:", res_row)

        layout.addLayout(form)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Buttons: export-only, render, close.
        btn_group = QGroupBox()
        btn_row = QHBoxLayout(btn_group)
        self.export_button = QPushButton("Export scene file…")
        self.export_button.clicked.connect(self._export_only)
        self.render_button = QPushButton("Render to image…")
        self.render_button.clicked.connect(self._render)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        btn_row.addWidget(self.export_button)
        btn_row.addWidget(self.render_button)
        btn_row.addStretch()
        btn_row.addWidget(close_button)
        layout.addWidget(btn_group)

        self._update_backend_status()

    # ------------------------------------------------------------------
    @property
    def _backend(self) -> str:
        return self.backend_combo.currentData()

    def _update_quality(self):
        photoreal = self.quality_combo.currentData() == "photoreal"
        self.photoreal_button.setEnabled(photoreal)

    def _update_backend_status(self):
        exe = rt.find_renderer(self._backend)
        if exe:
            self.status_label.setText(f"✓ {self._backend} found: {exe}")
            self.render_button.setEnabled(True)
        else:
            self.status_label.setText(
                f"⚠ {self._backend} not found on PATH — you can still export the "
                f"scene file and render it elsewhere.")
            self.render_button.setEnabled(False)

    def _edit_photoreal(self):
        dlg = PhotorealSettingsDialog(self._photoreal, self)
        if dlg.exec() == QDialog.Accepted:
            self._photoreal = dlg.options()

    def _build_scene(self) -> rt.RTScene:
        photoreal = (self._photoreal
                     if self.quality_combo.currentData() == "photoreal" else None)
        return rt.build_scene_from_widget(
            self._widget, self.width_spin.value(), self.height_spin.value(),
            photoreal=photoreal)

    def _empty_scene_guard(self, scene) -> bool:
        if not len(scene.spheres) and not len(scene.cylinders):
            QMessageBox.warning(
                self, "Nothing to render",
                "The current view has no spheres or bonds to export.")
            return True
        return False

    # ------------------------------------------------------------------
    def _export_only(self):
        scene = self._build_scene()
        if self._empty_scene_guard(scene):
            return
        suffix = rt.SCENE_SUFFIX[self._backend]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export scene file", f"render{suffix}",
            f"Scene file (*{suffix})")
        if not path:
            return
        try:
            rt.write_scene(scene, self._backend, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(
            self, "Exported",
            f"Wrote {len(scene.spheres)} spheres and {len(scene.cylinders)} "
            f"bonds to:\n{path}")

    def _render(self):
        scene = self._build_scene()
        if self._empty_scene_guard(scene):
            return
        suffix = rt.SCENE_SUFFIX[self._backend]
        img_path, _ = QFileDialog.getSaveFileName(
            self, "Render to image", "render.png", "PNG image (*.png)")
        if not img_path:
            return
        scene_path = str(Path(img_path).with_suffix(suffix))
        self.setEnabled(False)
        self.status_label.setText(f"Rendering with {self._backend}…")
        try:
            rt.render_scene(scene, self._backend, scene_path, img_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ray-trace render failed")
            QMessageBox.critical(self, "Render failed", str(exc))
            return
        finally:
            self.setEnabled(True)
            self._update_backend_status()
        self._show_result(img_path)

    def _show_result(self, img_path: str):
        preview = QDialog(self)
        preview.setWindowTitle(Path(img_path).name)
        lay = QVBoxLayout(preview)
        label = QLabel()
        pix = QPixmap(img_path)
        if not pix.isNull():
            label.setPixmap(pix.scaled(900, 700, Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation))
        label.setAlignment(Qt.AlignCenter)
        lay.addWidget(label)
        open_btn = QPushButton("Open in default viewer")
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(img_path)))
        lay.addWidget(open_btn)
        preview.resize(920, 760)
        preview.exec()

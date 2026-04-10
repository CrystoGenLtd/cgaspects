"""Dialog for configuring and launching animation video rendering."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .keyframe import AnimationTimeline
from .render_worker import VideoRenderWorker

logger = logging.getLogger("CGA:RenderDialog")


def _imageio_ffmpeg_available() -> bool:
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


class RenderAnimationDialog(QDialog):
    """Settings dialog for rendering an animation to video or image sequence."""

    renderStarted = Signal(object)  # emits the VideoRenderWorker

    def __init__(
        self,
        timeline: AnimationTimeline,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Render Animation")
        self.setMinimumWidth(500)
        self._timeline = timeline
        self._worker: Optional[VideoRenderWorker] = None

        self._build_ui(viewport_width, viewport_height)

    def _build_ui(self, vp_w: int, vp_h: int) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Output path
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self._edit_path = QLineEdit()
        self._edit_path.setPlaceholderText("Select output file or folder…")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(self._edit_path)
        path_layout.addWidget(browse_btn)
        form.addRow("Output:", path_row)

        # Format
        self._combo_format = QComboBox()
        mp4_available = _imageio_ffmpeg_available()
        if mp4_available:
            self._combo_format.addItem("MP4 Video (.mp4)", "mp4")
        self._combo_format.addItem("PNG Sequence (folder)", "image_sequence")
        if not mp4_available:
            note = QLabel(
                "ℹ MP4 output requires imageio[ffmpeg]. "
                "Install with: <code>pip install imageio[ffmpeg]</code>"
            )
            note.setTextFormat(Qt.RichText)
            note.setWordWrap(True)
            form.addRow("", note)
        self._combo_format.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Format:", self._combo_format)

        # Resolution
        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        self._spin_width = QSpinBox()
        self._spin_width.setRange(64, 7680)
        self._spin_width.setValue(vp_w)
        self._spin_height = QSpinBox()
        self._spin_height.setRange(64, 4320)
        self._spin_height.setValue(vp_h)
        res_layout.addWidget(self._spin_width)
        res_layout.addWidget(QLabel("×"))
        res_layout.addWidget(self._spin_height)
        res_layout.addStretch()
        form.addRow("Resolution:", res_row)

        # FPS (display only, from timeline)
        self._lbl_fps = QLabel(f"{self._timeline.fps} fps")
        form.addRow("Frame rate:", self._lbl_fps)

        # Duration / frame count (display only)
        n = self._timeline.total_frames()
        self._lbl_frames = QLabel(f"{n} frames ({self._timeline.duration:.1f} s)")
        form.addRow("Frames:", self._lbl_frames)

        layout.addLayout(form)

        # Progress bar (hidden until render starts)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._lbl_status = QLabel("")
        self._lbl_status.setVisible(False)
        layout.addWidget(self._lbl_status)

        # Buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_render = QPushButton("Render")
        self._btn_render.setDefault(True)
        self._btn_render.clicked.connect(self._start_render)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_render)
        btn_layout.addWidget(self._btn_cancel)
        layout.addWidget(btn_row)

        self._on_format_changed()

    def _on_format_changed(self):
        fmt = self._combo_format.currentData()
        if fmt == "mp4":
            self._edit_path.setPlaceholderText("output.mp4")
        else:
            self._edit_path.setPlaceholderText("Select output folder…")

    def _browse(self):
        fmt = self._combo_format.currentData()
        if fmt == "mp4":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "animation.mp4", "MP4 Video (*.mp4)"
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self._edit_path.setText(path)

    def _start_render(self):
        path = self._edit_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No Output Path", "Please select an output path.")
            return
        if len(self._timeline.keyframes) < 2:
            QMessageBox.warning(
                self, "Not Enough Keyframes",
                "You need at least 2 keyframes to render an animation."
            )
            return

        fmt = self._combo_format.currentData()
        w = self._spin_width.value()
        h = self._spin_height.value()

        self._worker = VideoRenderWorker(
            timeline=self._timeline,
            output_path=path,
            resolution=(w, h),
            scale_factor=1.0,
            export_format=fmt,
            parent=self,
        )
        self._worker.progressChanged.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._btn_render.setEnabled(False)
        self._btn_cancel.setText("Stop")
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._lbl_status.setText("Rendering…")
        self._lbl_status.setVisible(True)

        self.renderStarted.emit(self._worker)
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
            self._btn_render.setEnabled(True)
            self._btn_cancel.setText("Cancel")
            self._progress.setVisible(False)
            self._lbl_status.setVisible(False)
        else:
            self.reject()

    def _on_progress(self, pct: int):
        self._progress.setValue(pct)

    def _on_finished(self, path: str):
        self._progress.setValue(100)
        self._lbl_status.setText(f"Saved: {path}")
        self._btn_render.setEnabled(True)
        self._btn_cancel.setText("Close")
        QMessageBox.information(self, "Render Complete", f"Animation saved to:\n{path}")

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._lbl_status.setVisible(False)
        self._btn_render.setEnabled(True)
        self._btn_cancel.setText("Cancel")
        QMessageBox.critical(self, "Render Error", msg)

"""Background render worker for animation video export.

The worker runs in a QThread and emits frameRequested for each animation frame.
The main thread renders the frame via OpenGL, then calls frame_ready() to
unblock the worker. This pattern avoids GL cross-thread violations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QMutex, QThread, QWaitCondition, Signal
from PySide6.QtGui import QImage

from .keyframe import AnimationTimeline, CameraSnapshot

logger = logging.getLogger("CGA:RenderWorker")


def qimage_to_numpy(img: QImage) -> np.ndarray:
    """Convert a QImage to an RGBA uint8 numpy array."""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    ptr = img.constBits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4)
    return arr.copy()


class VideoRenderWorker(QThread):
    """Iterates through animation timeline frames, requests renders from the main thread,
    and writes frames to a video file or PNG sequence."""

    # Emitted for each frame: main thread should call frame_ready() after rendering
    frameRequested = Signal(int, object, object)  # (frame_idx, CameraSnapshot, data_frame|None)
    progressChanged = Signal(int)   # 0-100
    finished = Signal(str)          # output path
    error = Signal(str)

    def __init__(
        self,
        timeline: AnimationTimeline,
        output_path: str,
        resolution: tuple[int, int],
        scale_factor: float,
        export_format: str,   # "mp4" | "image_sequence"
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._timeline = timeline
        self._output_path = output_path
        self._resolution = resolution
        self._scale_factor = scale_factor
        self._export_format = export_format

        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._current_image: Optional[QImage] = None
        self._cancelled = False

    def frame_ready(self, image: QImage) -> None:
        """Called from the main thread to deliver a rendered frame."""
        self._mutex.lock()
        self._current_image = image
        self._condition.wakeAll()
        self._mutex.unlock()

    def cancel(self) -> None:
        self._cancelled = True
        # Unblock if waiting
        self._mutex.lock()
        self._current_image = QImage()  # empty sentinel
        self._condition.wakeAll()
        self._mutex.unlock()

    def run(self) -> None:
        total = self._timeline.total_frames()
        if total < 1:
            self.error.emit("Timeline has no frames to render.")
            return

        try:
            if self._export_format == "mp4":
                self._run_mp4(total)
            else:
                self._run_image_sequence(total)
        except Exception as exc:
            logger.exception("Render worker error")
            self.error.emit(str(exc))

    def _request_frame(self, frame_idx: int) -> Optional[QImage]:
        """Request a render from the main thread and wait for the result."""
        t = self._timeline.time_at_frame(frame_idx)
        try:
            snapshot, data_frame = self._timeline.get_state_at_time(t)
        except ValueError:
            return None

        self._mutex.lock()
        self._current_image = None
        self.frameRequested.emit(frame_idx, snapshot, data_frame)
        self._condition.wait(self._mutex)
        img = self._current_image
        self._mutex.unlock()
        return img

    def _run_mp4(self, total: int) -> None:
        try:
            import imageio
        except ImportError:
            self.error.emit(
                "imageio is not installed. Run: pip install imageio[ffmpeg]"
            )
            return

        out = self._output_path
        fps = self._timeline.fps
        writer = imageio.get_writer(out, fps=fps, quality=8)

        try:
            for i in range(total):
                if self._cancelled:
                    break
                img = self._request_frame(i)
                if img is None or img.isNull():
                    continue
                arr = qimage_to_numpy(img)
                writer.append_data(arr[:, :, :3])  # drop alpha for MP4
                self.progressChanged.emit(round((i + 1) / total * 100))
        finally:
            writer.close()

        if not self._cancelled:
            self.finished.emit(out)

    def _run_image_sequence(self, total: int) -> None:
        out_dir = Path(self._output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(total):
            if self._cancelled:
                break
            img = self._request_frame(i)
            if img is None or img.isNull():
                continue
            frame_path = str(out_dir / f"frame_{i:05d}.png")
            img.save(frame_path)
            self.progressChanged.emit(round((i + 1) / total * 100))

        if not self._cancelled:
            self.finished.emit(str(out_dir))

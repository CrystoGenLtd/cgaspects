"""Keyframe timeline dock widget for the cgaspects animation system."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF, QSize
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..utils import qticons_rc  # noqa: F401 — registers the icon resources
from .history import TimelineHistory
from .keyframe import AnimationTimeline, INTERPOLATION_MODES


# ---------------------------------------------------------------------------
# Graphics items
# ---------------------------------------------------------------------------

class _KeyframeItem(QGraphicsItem):
    """Diamond-shaped keyframe marker on the timeline."""

    SIZE = 10  # half-size of the diamond

    def __init__(self, index: int, x: float, scene_height: float, view: "_TimelineView"):
        super().__init__()
        self._index = index
        self._view = view
        self.setPos(x, scene_height / 2)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(1)
        self._dragging = False
        self._range_select = False

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, v: int) -> None:
        self._index = v

    def boundingRect(self) -> QRectF:
        s = self.SIZE + 2
        return QRectF(-s, -s, s * 2, s * 2)

    def paint(self, painter: QPainter, option, widget=None):
        s = self.SIZE
        diamond = QPolygonF([
            QPointF(0, -s),
            QPointF(s, 0),
            QPointF(0, s),
            QPointF(-s, 0),
        ])
        selected = self.isSelected()
        fill = QColor(255, 200, 50) if selected else QColor(200, 160, 30)
        border = QColor(255, 255, 100) if selected else QColor(255, 200, 50)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(border, 1.5))
        painter.drawPolygon(diamond)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self._view:
            # Constrain to horizontal movement only
            new_pos = value
            new_pos.setY(self.pos().y())
            # Clamp to scene bounds
            min_x = self._view._time_to_x(0)
            max_x = self._view._time_to_x(self._view._timeline.duration)
            clamped_x = max(min_x, min(max_x, new_pos.x()))
            new_pos.setX(clamped_x)
            return new_pos
        # Do NOT emit keyframeMoved here — rebuild() during drag deletes the live item.
        # Emit is deferred to mouseReleaseEvent instead.
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        if self._range_select:
            # Swallow the release of a shift+click: QGraphicsItem's default
            # handler would collapse the fresh range selection back to just
            # this item (it does so whenever the mouse didn't move).
            self._range_select = False
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self._view and self._dragging:
            # Dragging moves every selected keyframe; commit them all at once.
            moves = [
                (it.index, self._view._x_to_time(it.x()))
                for it in self._view._items
                if it.isSelected() or it is self
            ]
            self._view.keyframesMoved.emit(moves)
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier:
            # Shift+click: select the range from the first selected keyframe to here
            self._view.select_range_to(self._index)
            self._range_select = True
            event.accept()
            return
        self._dragging = True
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._dragging = False
        self._view.keyframeDoubleClicked.emit(self._index)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        # Right-click targets this keyframe; if it is part of the current
        # multi-selection, delete/duplicate act on the whole selection.
        selection = self._view.selected_indices()
        targets = selection if self._index in selection else [self._index]
        many = len(targets) > 1

        menu = QMenu()
        edit_action = menu.addAction("Update Keyframe (capture current view)")
        duplicate_action = menu.addAction(
            f"Duplicate {len(targets)} Keyframes" if many else "Duplicate Keyframe")
        delete_action = menu.addAction(
            f"Delete {len(targets)} Keyframes" if many else "Delete Keyframe")
        menu.addSeparator()
        space_action = menu.addAction(
            "Space Selected Evenly" if len(selection) >= 2 else "Space All Evenly")

        chosen = menu.exec(event.screenPos())
        if chosen == edit_action:
            self._view.keyframeEditRequested.emit(self._index)
        elif chosen == delete_action:
            self._view.keyframesRemoved.emit(list(targets))
        elif chosen == duplicate_action:
            self._view.keyframesDuplicated.emit(list(targets))
        elif chosen == space_action:
            self._view.spaceEvenlyRequested.emit()


# ---------------------------------------------------------------------------
# Playhead item
# ---------------------------------------------------------------------------

class _PlayheadItem(QGraphicsItem):
    """Draggable vertical dashed line marking the current time position."""

    def __init__(self, x: float, ruler_height: int, track_height: int, view: "_TimelineView"):
        super().__init__()
        self._ruler_height = ruler_height
        self._track_height = track_height
        self._total_height = ruler_height + track_height
        self._view = view
        self._suppress_signal = False
        self.setPos(x, 0)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setZValue(2)
        self.setCursor(Qt.SizeHorCursor)

    def boundingRect(self) -> QRectF:
        return QRectF(-5, 0, 10, self._total_height)

    def paint(self, painter: QPainter, option, widget=None):
        pen = QPen(QColor(255, 80, 80), 1.5, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, self._ruler_height, 0, self._total_height)
        # Small triangle handle in the ruler
        painter.setBrush(QBrush(QColor(255, 80, 80)))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(QPolygonF([
            QPointF(-5, 0),
            QPointF(5, 0),
            QPointF(0, 9),
        ]))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self._view:
            new_pos = value
            new_pos.setY(0)
            min_x = self._view._time_to_x(0)
            max_x = self._view._time_to_x(self._view._timeline.duration)
            new_pos.setX(max(min_x, min(max_x, new_pos.x())))
            return new_pos
        if change == QGraphicsItem.ItemPositionHasChanged and self._view and not self._suppress_signal:
            t = self._view._x_to_time(self.x())
            self._view.playheadMoved.emit(t)
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# Timeline graphics view
# ---------------------------------------------------------------------------

class _TimelineView(QGraphicsView):
    """Custom graphics view rendering the ruler, keyframe diamonds, and segment labels."""

    keyframesMoved = Signal(list)           # [(index, new_time), ...]
    keyframesRemoved = Signal(list)         # [index, ...]
    keyframesDuplicated = Signal(list)      # [index, ...]
    keyframeEditRequested = Signal(int)     # re-capture current view into keyframe
    selectionChanged = Signal(list)         # sorted selected indices
    keyframeDoubleClicked = Signal(int)     # index double-clicked → seek to keyframe
    spaceEvenlyRequested = Signal()
    timelineClicked = Signal(float)         # bare click on ruler → preview seek
    playheadMoved = Signal(float)           # user dragged the playhead

    RULER_HEIGHT = 20
    TRACK_HEIGHT = 60

    def __init__(self, timeline: AnimationTimeline, parent=None):
        super().__init__(parent)
        self._timeline = timeline
        self._items: list[_KeyframeItem] = []
        self._playhead_time: float = 0.0
        self._playhead_item: Optional[_PlayheadItem] = None
        self._rebuilding = False
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.setRenderHint(QPainter.Antialiasing)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumHeight(self.RULER_HEIGHT + self.TRACK_HEIGHT + 4)
        self.setMaximumHeight(self.RULER_HEIGHT + self.TRACK_HEIGHT + 4)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _time_to_x(self, t: float) -> float:
        w = self.viewport().width()
        margin = 20
        usable = max(1, w - 2 * margin)
        return margin + (t / max(self._timeline.duration, 1e-9)) * usable

    def _x_to_time(self, x: float) -> float:
        w = self.viewport().width()
        margin = 20
        usable = max(1, w - 2 * margin)
        return max(0.0, min(self._timeline.duration, (x - margin) / usable * self._timeline.duration))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selected_indices(self) -> list[int]:
        return sorted(it.index for it in self._items if it.isSelected())

    def _on_scene_selection_changed(self):
        if not self._rebuilding:
            self.selectionChanged.emit(self.selected_indices())

    def select_range_to(self, index: int) -> None:
        """Select the range between the first already-selected keyframe and index."""
        current = self.selected_indices()
        a, b = sorted((current[0], index)) if current else (index, index)
        self._rebuilding = True
        try:
            for it in self._items:
                it.setSelected(a <= it.index <= b)
        finally:
            self._rebuilding = False
        self.selectionChanged.emit(self.selected_indices())

    # ------------------------------------------------------------------
    # Rebuild scene from timeline data
    # ------------------------------------------------------------------

    def rebuild(self, selected=None):
        """Rebuild the scene; `selected` is the set of indices to re-select
        (None preserves the current selection, e.g. across a resize)."""
        selected = set(self.selected_indices()) if selected is None else set(selected)
        self._rebuilding = True
        try:
            self._rebuild_scene(selected)
        finally:
            self._rebuilding = False
        self.selectionChanged.emit(self.selected_indices())

    def _rebuild_scene(self, selected: set):
        self._scene.clear()
        self._items.clear()
        tl = self._timeline
        total_h = self.RULER_HEIGHT + self.TRACK_HEIGHT

        # Draw segment interpolation labels between keyframes
        for i in range(len(tl.keyframes) - 1):
            mode = tl.interpolation[i] if i < len(tl.interpolation) else "linear"
            x_a = self._time_to_x(tl.keyframes[i].time)
            x_b = self._time_to_x(tl.keyframes[i + 1].time)
            x_mid = (x_a + x_b) / 2
            label_item = self._scene.addText(mode, QFont("Arial", 7))
            label_item.setDefaultTextColor(QColor(160, 160, 160))
            label_item.setPos(x_mid - label_item.boundingRect().width() / 2,
                              self.RULER_HEIGHT + self.TRACK_HEIGHT * 0.55)

        # Draw connecting line between keyframes
        if len(tl.keyframes) > 1:
            track_y = self.RULER_HEIGHT + self.TRACK_HEIGHT / 2
            for i in range(len(tl.keyframes) - 1):
                x_a = self._time_to_x(tl.keyframes[i].time)
                x_b = self._time_to_x(tl.keyframes[i + 1].time)
                line = self._scene.addLine(x_a, track_y, x_b, track_y,
                                           QPen(QColor(100, 100, 100), 1.5))

        # Draw keyframe diamonds
        track_y = self.RULER_HEIGHT + self.TRACK_HEIGHT / 2
        for i, kf in enumerate(tl.keyframes):
            x = self._time_to_x(kf.time)
            item = _KeyframeItem(i, x, total_h, self)
            item.setPos(x, track_y)
            self._scene.addItem(item)
            item.setSelected(i in selected)
            self._items.append(item)

        # Draw playhead
        self._playhead_item = _PlayheadItem(
            self._time_to_x(self._playhead_time),
            self.RULER_HEIGHT,
            self.TRACK_HEIGHT,
            self,
        )
        self._scene.addItem(self._playhead_item)

        self._scene.setSceneRect(0, 0, self.viewport().width(), total_h)

    def set_playhead_time(self, t: float) -> None:
        """Move the playhead to the given time without emitting playheadMoved."""
        self._playhead_time = t
        if self._playhead_item is not None:
            self._playhead_item._suppress_signal = True
            self._playhead_item.setPos(self._time_to_x(t), 0)
            self._playhead_item._suppress_signal = False

    # ------------------------------------------------------------------
    # Drawing ruler via background
    # ------------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect):
        super().drawBackground(painter, rect)
        vp = self.viewport()
        w = vp.width()

        # Ruler background
        painter.fillRect(0, 0, w, self.RULER_HEIGHT, QColor(45, 45, 45))
        # Track background
        painter.fillRect(0, self.RULER_HEIGHT, w, self.TRACK_HEIGHT, QColor(35, 35, 35))

        # Ruler ticks
        duration = self._timeline.duration
        if duration <= 0:
            return
        painter.setPen(QPen(QColor(160, 160, 160), 1))
        font = QFont("Arial", 7)
        painter.setFont(font)
        step = 1.0  # 1 second ticks
        if duration > 30:
            step = 5.0
        elif duration > 60:
            step = 10.0
        t = 0.0
        while t <= duration + 1e-6:
            x = self._time_to_x(t)
            painter.drawLine(int(x), self.RULER_HEIGHT - 6, int(x), self.RULER_HEIGHT)
            painter.drawText(int(x) - 10, 2, 20, self.RULER_HEIGHT - 6,
                             Qt.AlignCenter, f"{t:.0f}s")
            t += step

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rebuild()

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None and event.button() == Qt.LeftButton:
            self._scene.clearSelection()
            t = self._x_to_time(self.mapToScene(event.pos()).x())
            self.timelineClicked.emit(t)
        else:
            super().mousePressEvent(event)

    def keyPressEvent(self, event):
        selected = self.selected_indices()
        if selected:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self.keyframesRemoved.emit(selected)
                event.accept()
                return
            if event.key() == Qt.Key_D:
                if event.modifiers() & Qt.ShiftModifier:
                    self.keyframesDuplicated.emit(selected)
                else:
                    self.keyframesRemoved.emit(selected)
                event.accept()
                return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Main dock widget
# ---------------------------------------------------------------------------

class KeyframeTimelineWidget(QWidget):
    """Inline keyframe animation timeline panel (embedded below the OpenGL viewport)."""

    # Signals consumed by MainWindow
    keyframeAddRequested = Signal()
    keyframeEditRequested = Signal(int)  # re-capture current view into keyframe
    previewRequested = Signal(float)   # time position for preview tick
    previewStopped = Signal()
    renderRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._timeline: Optional[AnimationTimeline] = None
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._on_preview_tick)
        self._preview_time: float = 0.0
        self._selected_indices: list[int] = []
        self._history = TimelineHistory()  # in-memory until init_history() is called

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # --- Toolbar row ---
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(6)

        def icon_button(icon_name: str, tooltip: str) -> QPushButton:
            btn = QPushButton()
            btn.setIcon(QIcon(f":/material_icons/material_icons/png/{icon_name}.png"))
            btn.setToolTip(tooltip)
            btn.setFixedSize(24, 24)
            btn.setIconSize(QSize(18, 18))
            btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid rgba(128, 128, 128, 0.35);
                    border-radius: 6px;
                    background: rgba(128, 128, 128, 0.12);
                }
                QPushButton:hover { background: rgba(128, 128, 128, 0.25); }
                QPushButton:pressed, QPushButton:checked { background: rgba(128, 128, 128, 0.40); }
                QPushButton:disabled { background: transparent; }
            """)
            return btn

        self._btn_add = icon_button(
            "add", "Add keyframe (shortcut: K)")
        self._btn_add.clicked.connect(self.keyframeAddRequested)

        self._btn_edit = icon_button(
            "modify", "Modify keyframe")
        self._btn_edit.setEnabled(False)
        self._btn_edit.clicked.connect(self._on_edit_clicked)

        self._btn_delete = icon_button("bin", "Delete keyframe(s)")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(
            lambda: self._on_kfs_removed(list(self._selected_indices)))

        self._btn_duplicate = icon_button(
            "duplicate", "Duplicate keyframe(s)")
        self._btn_duplicate.setEnabled(False)
        self._btn_duplicate.clicked.connect(
            lambda: self._on_kfs_duplicated(list(self._selected_indices)))

        self._btn_space = icon_button(
            "distribute",
            "Distribute keyframes")
        self._btn_space.setEnabled(False)
        self._btn_space.clicked.connect(self._space_evenly)

        self._btn_preview = icon_button("preview", "Preview")
        self._btn_preview.setCheckable(True)
        self._btn_preview.clicked.connect(self._toggle_preview)

        lbl_dur = QLabel("Duration:")
        self._spin_duration = QDoubleSpinBox()
        self._spin_duration.setRange(0.5, 3600)
        self._spin_duration.setValue(10.0)
        self._spin_duration.setSuffix(" s")
        self._spin_duration.setFixedWidth(80)
        self._spin_duration.valueChanged.connect(self._on_duration_changed)

        lbl_fps = QLabel("FPS:")
        self._spin_fps = QSpinBox()
        self._spin_fps.setRange(1, 120)
        self._spin_fps.setValue(24)
        self._spin_fps.setFixedWidth(55)
        self._spin_fps.valueChanged.connect(self._on_fps_changed)

        self._btn_render = icon_button("render", "Render")
        self._btn_render.clicked.connect(self.renderRequested)

        toolbar_layout.addWidget(self._btn_add)
        toolbar_layout.addWidget(self._btn_edit)
        toolbar_layout.addWidget(self._btn_delete)
        toolbar_layout.addWidget(self._btn_duplicate)
        toolbar_layout.addWidget(self._btn_space)
        toolbar_layout.addWidget(self._btn_preview)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(lbl_dur)
        toolbar_layout.addWidget(self._spin_duration)
        toolbar_layout.addWidget(lbl_fps)
        toolbar_layout.addWidget(self._spin_fps)
        toolbar_layout.addWidget(self._btn_render)

        # --- Timeline view ---
        self._timeline_view = _TimelineView(AnimationTimeline())
        self._timeline_view.keyframesMoved.connect(self._on_kfs_moved)
        self._timeline_view.keyframesRemoved.connect(self._on_kfs_removed)
        self._timeline_view.keyframesDuplicated.connect(self._on_kfs_duplicated)
        self._timeline_view.keyframeEditRequested.connect(self.keyframeEditRequested)
        self._timeline_view.selectionChanged.connect(self._on_selection_changed)
        self._timeline_view.keyframeDoubleClicked.connect(self._on_kf_double_clicked)
        self._timeline_view.spaceEvenlyRequested.connect(self._space_evenly)
        self._timeline_view.timelineClicked.connect(self._on_timeline_click)
        self._timeline_view.playheadMoved.connect(self._on_playhead_moved)

        # --- Inspector row ---
        inspector = QWidget()
        inspector_layout = QHBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(8)

        inspector_layout.addWidget(QLabel("Time:"))
        self._spin_kf_time = QDoubleSpinBox()
        self._spin_kf_time.setRange(0, 3600)
        self._spin_kf_time.setSuffix(" s")
        self._spin_kf_time.setFixedWidth(80)
        self._spin_kf_time.setEnabled(False)
        self._spin_kf_time.valueChanged.connect(self._on_inspector_time_changed)

        inspector_layout.addWidget(self._spin_kf_time)
        inspector_layout.addWidget(QLabel("Data frame:"))
        self._spin_data_frame = QSpinBox()
        self._spin_data_frame.setRange(-1, 99999)
        self._spin_data_frame.setSpecialValueText("—")
        self._spin_data_frame.setValue(-1)
        self._spin_data_frame.setFixedWidth(70)
        self._spin_data_frame.setEnabled(False)
        self._spin_data_frame.valueChanged.connect(self._on_inspector_data_frame_changed)

        inspector_layout.addWidget(self._spin_data_frame)
        inspector_layout.addWidget(QLabel("Interpolation:"))
        self._combo_interp = QComboBox()
        self._combo_interp.addItems(INTERPOLATION_MODES)
        self._combo_interp.setEnabled(False)
        self._combo_interp.currentTextChanged.connect(self._on_interp_changed)
        inspector_layout.addWidget(self._combo_interp)
        inspector_layout.addStretch()

        main_layout.addWidget(toolbar)
        main_layout.addWidget(self._timeline_view)
        main_layout.addWidget(inspector)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_timeline(self, tl: AnimationTimeline) -> None:
        self._timeline = tl
        self._timeline_view._timeline = tl
        self._sync_spins()
        self._timeline_view.rebuild(set())

    def init_history(self, path) -> None:
        """Autosave the current timeline state to `path` on every change.

        The undo/redo stack itself is session-only; the autosave file is only
        a crash-recovery copy (loadable via "Load Animation…").
        """
        self._history = TimelineHistory(path)
        if self._timeline is not None:
            # Baseline for undo; skip the autosave so a previous session's
            # crash-recovery file survives until the first real edit.
            self._history.push(self._timeline.to_dict(), autosave=False)

    def refresh(self) -> None:
        """Rebuild the graphics view and record the state for undo."""
        self._timeline_view.rebuild(set())
        self._push_history()

    def _sync_spins(self) -> None:
        if self._timeline is None:
            return
        self._spin_duration.blockSignals(True)
        self._spin_duration.setValue(self._timeline.duration)
        self._spin_duration.blockSignals(False)
        self._spin_fps.blockSignals(True)
        self._spin_fps.setValue(self._timeline.fps)
        self._spin_fps.blockSignals(False)

    # ------------------------------------------------------------------
    # Undo / redo
    # ------------------------------------------------------------------

    def _push_history(self) -> None:
        if self._timeline is not None:
            self._history.push(self._timeline.to_dict())

    def undo(self) -> None:
        self._apply_history_state(self._history.undo())

    def redo(self) -> None:
        self._apply_history_state(self._history.redo())

    def _apply_history_state(self, state) -> None:
        if state is None or self._timeline is None:
            return
        self._timeline.restore(state)
        self._sync_spins()
        self._timeline_view.rebuild(set())

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _toggle_preview(self, checked: bool):
        if checked:
            self._start_preview()
        else:
            self._stop_preview()

    def _start_preview(self):
        if self._timeline is None or len(self._timeline.keyframes) < 2:
            self._btn_preview.setChecked(False)
            return
        self._preview_time = self._timeline.keyframes[0].time
        fps = self._timeline.fps or 24
        self._btn_preview.setIcon(QIcon(":/material_icons/material_icons/png/pause-preview.png"))
        self._preview_timer.start(max(1, 1000 // fps))

    def _stop_preview(self):
        self._preview_timer.stop()
        self._btn_preview.setChecked(False)
        self._btn_preview.setIcon(QIcon(":/material_icons/material_icons/png/preview.png"))
        self.previewStopped.emit()

    def stop_preview(self) -> None:
        """External stop (e.g. when play button in main window is pressed)."""
        self._stop_preview()

    def _on_preview_tick(self):
        if self._timeline is None:
            self._stop_preview()
            return
        self.previewRequested.emit(self._preview_time)
        self._timeline_view.set_playhead_time(self._preview_time)
        self._preview_time += 1.0 / max(1, self._timeline.fps)
        if self._preview_time > self._timeline.duration:
            self._preview_time = self._timeline.keyframes[0].time  # loop

    # ------------------------------------------------------------------
    # Timeline view signal handlers
    # ------------------------------------------------------------------

    def _on_selection_changed(self, indices: list):
        self._selected_indices = list(indices)
        self._update_buttons()
        self._update_inspector()

    def _update_buttons(self):
        n = len(self._selected_indices)
        self._btn_edit.setEnabled(n == 1)
        self._btn_delete.setEnabled(n >= 1)
        self._btn_duplicate.setEnabled(n >= 1)
        self._btn_space.setEnabled(
            self._timeline is not None and len(self._timeline.keyframes) >= 2)

    def _on_edit_clicked(self):
        if len(self._selected_indices) == 1:
            self.keyframeEditRequested.emit(self._selected_indices[0])

    def _rebuild_reselect(self, kf_objs: list):
        """Rebuild the view, re-selecting the given Keyframe objects wherever
        they ended up after a re-sort."""
        if self._timeline is None:
            return
        ids = {id(k) for k in kf_objs}
        indices = {i for i, k in enumerate(self._timeline.keyframes) if id(k) in ids}
        self._timeline_view.rebuild(indices)

    def _selected_keyframe_objs(self) -> list:
        if self._timeline is None:
            return []
        n = len(self._timeline.keyframes)
        return [self._timeline.keyframes[i] for i in self._selected_indices if i < n]

    def _on_kfs_moved(self, moves: list):
        if self._timeline is None:
            return
        n = len(self._timeline.keyframes)
        moves = [(i, t) for i, t in moves
                 if 0 <= i < n and abs(self._timeline.keyframes[i].time - t) > 1e-9]
        if not moves:
            return
        selected = self._selected_keyframe_objs()
        self._timeline.move_keyframes(moves)
        self._push_history()
        # Defer rebuild so we don't delete the item while still inside its mouseReleaseEvent
        QTimer.singleShot(0, lambda: self._rebuild_reselect(selected))

    def _on_kfs_removed(self, indices: list):
        if self._timeline is None or not indices:
            return
        for i in sorted(set(indices), reverse=True):
            self._timeline.remove_keyframe(i)
        self._push_history()
        # Defer: may be called from a keyframe item's own context menu event
        QTimer.singleShot(0, lambda: self._timeline_view.rebuild(set()))

    def _on_kfs_duplicated(self, indices: list):
        if self._timeline is None or not indices:
            return
        n = len(self._timeline.keyframes)
        copies = []
        for i in sorted(set(indices)):
            if 0 <= i < n:
                kf = deepcopy(self._timeline.keyframes[i])
                kf.time += 0.5
                copies.append(kf)
        for kf in copies:
            self._timeline.add_keyframe(kf)
        self._push_history()
        QTimer.singleShot(0, lambda: self._rebuild_reselect(copies))

    def _space_evenly(self):
        if self._timeline is None or len(self._timeline.keyframes) < 2:
            return
        selected = self._selected_keyframe_objs()
        indices = list(self._selected_indices) if len(self._selected_indices) >= 2 else None
        self._timeline.space_evenly(indices)
        self._push_history()
        QTimer.singleShot(0, lambda: self._rebuild_reselect(selected))

    def _on_kf_double_clicked(self, index: int):
        """Seek the viewport to the exact time of the double-clicked keyframe."""
        if self._timeline is None or index >= len(self._timeline.keyframes):
            return
        t = self._timeline.keyframes[index].time
        self._preview_time = t
        self._timeline_view.set_playhead_time(t)
        self.previewRequested.emit(t)

    def _on_playhead_moved(self, t: float):
        """User dragged the playhead — seek viewport and sync preview time."""
        self._preview_time = t
        self.previewRequested.emit(t)

    def _on_timeline_click(self, t: float):
        self._timeline_view.set_playhead_time(t)
        self.previewRequested.emit(t)

    # ------------------------------------------------------------------
    # Inspector
    # ------------------------------------------------------------------

    def _update_inspector(self):
        tl = self._timeline
        sel = self._selected_indices
        single = tl is not None and len(sel) == 1 and sel[0] < len(tl.keyframes)
        self._spin_kf_time.setEnabled(single)
        self._spin_data_frame.setEnabled(single)

        # Segment interpolation: enabled if any selected keyframe has a next segment
        seg_indices = [] if tl is None else [i for i in sel if i < len(tl.interpolation)]
        self._combo_interp.setEnabled(bool(seg_indices))

        if not single:
            self._spin_kf_time.blockSignals(True)
            self._spin_kf_time.setValue(0)
            self._spin_kf_time.blockSignals(False)
            self._spin_data_frame.blockSignals(True)
            self._spin_data_frame.setValue(-1)
            self._spin_data_frame.blockSignals(False)
        else:
            kf = tl.keyframes[sel[0]]
            self._spin_kf_time.blockSignals(True)
            self._spin_kf_time.setValue(kf.time)
            self._spin_kf_time.blockSignals(False)
            self._spin_data_frame.blockSignals(True)
            self._spin_data_frame.setValue(-1 if kf.data_frame is None else kf.data_frame)
            self._spin_data_frame.blockSignals(False)

        if seg_indices:
            # Show the first selected segment's mode
            mode = tl.interpolation[seg_indices[0]]
            self._combo_interp.blockSignals(True)
            idx = INTERPOLATION_MODES.index(mode) if mode in INTERPOLATION_MODES else 0
            self._combo_interp.setCurrentIndex(idx)
            self._combo_interp.blockSignals(False)

    def _on_inspector_time_changed(self, val: float):
        if len(self._selected_indices) != 1 or self._timeline is None:
            return
        idx = self._selected_indices[0]
        if idx >= len(self._timeline.keyframes):
            return
        kf = self._timeline.keyframes[idx]
        self._timeline.move_keyframe(idx, val)
        self._push_history()
        self._rebuild_reselect([kf])

    def _on_inspector_data_frame_changed(self, val: int):
        if len(self._selected_indices) != 1 or self._timeline is None:
            return
        idx = self._selected_indices[0]
        if idx < len(self._timeline.keyframes):
            self._timeline.keyframes[idx].data_frame = None if val == -1 else val
            self._push_history()

    def _on_interp_changed(self, mode: str):
        if self._timeline is None:
            return
        # Apply to the segment after every selected keyframe
        seg_indices = [i for i in self._selected_indices if i < len(self._timeline.interpolation)]
        if not seg_indices:
            return
        for i in seg_indices:
            self._timeline.set_interpolation(i, mode)
        self._push_history()
        self._timeline_view.rebuild(set(self._selected_indices))

    def _on_duration_changed(self, val: float):
        if self._timeline is not None:
            self._timeline.duration = val
            self._push_history()
            self._timeline_view.rebuild()

    def _on_fps_changed(self, val: int):
        if self._timeline is not None:
            self._timeline.fps = val
            self._push_history()

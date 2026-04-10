from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class _GradientWidget(QWidget):
    """Paints a horizontal colormap gradient with min/max labels."""

    def __init__(self, rows, min_val, max_val, parent=None):
        super().__init__(parent)
        self._rows = rows  # list of (value, (r, g, b))
        self._min_val = min_val
        self._max_val = max_val
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        strip_top = 4
        strip_height = 30
        strip_left = 4
        strip_right = self.width() - 4
        strip_width = strip_right - strip_left

        if strip_width <= 0 or not self._rows:
            return

        sorted_rows = sorted(self._rows, key=lambda r: r[0] if r[0] is not None else 0)
        val_range = self._max_val - self._min_val if self._max_val != self._min_val else 1.0

        gradient = QLinearGradient(strip_left, 0, strip_right, 0)
        for val, (r, g, b) in sorted_rows:
            pos = (val - self._min_val) / val_range if val is not None else 0.0
            pos = max(0.0, min(1.0, pos))
            gradient.setColorAt(pos, QColor.fromRgbF(r, g, b))

        painter.fillRect(strip_left, strip_top, strip_width, strip_height, gradient)
        painter.setPen(Qt.black)
        painter.drawRect(strip_left, strip_top, strip_width - 1, strip_height - 1)

        label_y = strip_top + strip_height + 16
        painter.drawText(strip_left, label_y, _fmt(self._min_val))
        max_text = _fmt(self._max_val)
        fm = painter.fontMetrics()
        painter.drawText(strip_right - fm.horizontalAdvance(max_text), label_y, max_text)


def _fmt(val):
    """Format a numeric legend value compactly."""
    if val is None:
        return ""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return f"{val:.4g}"


class ColorLegendDialog(QDialog):
    """Non-modal dialog showing the current point-cloud colour legend.

    In atom/shell modes the colour swatches are clickable — double-click to
    pick a new colour, right-click (or the Reset button on each row) to revert
    to the default.  A "Reset All" button clears every override at once.
    """

    _TABLE_THRESHOLD = 10  # switch to gradient above this many unique values

    # Emitted when the user picks a new colour for a legend entry.
    # Payload: (mode, key, (r, g, b))
    #   mode  — "atom" | "docking_shell"
    #   key   — element symbol (str) or shell_id (int)
    #   color — RGB tuple of floats in [0, 1], or None to reset
    colorOverrideRequested = Signal(str, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colour Legend")
        self.setModal(False)
        self.resize(300, 340)

        self._info = None
        self._user_mode = None  # None = auto, "table" or "gradient" = user override

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title
        self._title_label = QLabel("—")
        font = self._title_label.font()
        font.setBold(True)
        self._title_label.setFont(font)
        layout.addWidget(self._title_label)

        # Toggle button (table ↔ gradient)
        self._toggle_btn = QPushButton()
        self._toggle_btn.setFixedHeight(24)
        self._toggle_btn.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle_btn)

        # Body container
        self._body_container = QWidget()
        self._body_layout = QVBoxLayout(self._body_container)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._body_container, 1)

        # Reset All button — only shown in editable (atom/shell) table modes
        self._reset_all_btn = QPushButton("Reset All Colours")
        self._reset_all_btn.setFixedHeight(24)
        self._reset_all_btn.clicked.connect(self._on_reset_all)
        self._reset_all_btn.hide()
        layout.addWidget(self._reset_all_btn)

        self._body_widget = None  # current table or gradient widget

    # ------------------------------------------------------------------
    # Public slot
    # ------------------------------------------------------------------

    def update_legend(self, info: dict):
        if info is None:
            return
        self._info = info
        self._refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_editable_mode(self, info: dict) -> bool:
        """Return True when rows represent named entries users can recolour."""
        return info.get("mode") in ("atom", "docking_shell")

    def _auto_mode(self, rows):
        if not rows or rows[0][0] is None:
            return "table"
        return "table" if len(rows) <= self._TABLE_THRESHOLD else "gradient"

    def _effective_mode(self, rows):
        if not rows or rows[0][0] is None:
            return "table"
        return self._user_mode if self._user_mode is not None else self._auto_mode(rows)

    def _refresh(self):
        info = self._info
        rows = info["rows"]
        mode = self._effective_mode(rows)
        editable = self._is_editable_mode(info)

        self._title_label.setText(info.get("color_by", ""))

        is_single = (not rows) or rows[0][0] is None
        # Toggle only makes sense for non-editable (colormap) modes
        self._toggle_btn.setVisible(not is_single and not editable)
        if mode == "table":
            self._toggle_btn.setText("Switch to Gradient")
        else:
            self._toggle_btn.setText("Switch to Table")

        self._reset_all_btn.setVisible(editable)

        # Replace body widget
        if self._body_widget is not None:
            self._body_layout.removeWidget(self._body_widget)
            self._body_widget.deleteLater()
            self._body_widget = None

        if mode == "gradient" and not editable:
            self._body_widget = _GradientWidget(
                rows, info["min_val"], info["max_val"], parent=self._body_container
            )
        else:
            self._body_widget = self._build_table(rows, editable, info.get("mode"))

        self._body_layout.addWidget(self._body_widget)

    def _build_table(self, rows, editable: bool, mode: str | None):
        col_count = 3 if editable else 2
        table = QTableWidget(len(rows), col_count)
        headers = ["Label", "Colour"] + ([""] if editable else [])
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(not editable)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)

        for i, (key, rgb) in enumerate(rows):
            label = "Single Colour" if key is None else str(key)
            table.setItem(i, 0, QTableWidgetItem(label))

            swatch = QTableWidgetItem()
            swatch.setBackground(QColor.fromRgbF(*rgb))
            if editable:
                swatch.setToolTip("Double-click to change colour")
            table.setItem(i, 1, swatch)

            if editable:
                reset_btn = QPushButton("Reset")
                reset_btn.setFixedHeight(20)
                # Capture key and mode for the closure
                reset_btn.clicked.connect(
                    lambda checked=False, k=key, m=mode: self._on_reset_row(m, k)
                )
                table.setCellWidget(i, 2, reset_btn)

        if editable:
            table.cellDoubleClicked.connect(
                lambda row, col, r=rows, m=mode: self._on_cell_double_clicked(row, col, r, m)
            )

        table.resizeColumnsToContents()
        return table

    def _on_cell_double_clicked(self, row: int, col: int, rows: list, mode: str):
        if col != 1:  # only swatch column
            return
        key, rgb = rows[row]
        initial = QColor.fromRgbF(*rgb)
        color = QColorDialog.getColor(initial, self, "Pick Colour")
        if not color.isValid():
            return
        new_rgb = (color.redF(), color.greenF(), color.blueF())
        self.colorOverrideRequested.emit(mode, key, new_rgb)

    def _on_reset_row(self, mode: str, key):
        self.colorOverrideRequested.emit(mode, key, None)

    def _on_reset_all(self):
        self.colorOverrideRequested.emit("reset_all", None, None)

    def _on_toggle(self):
        if self._info is None:
            return
        rows = self._info["rows"]
        current = self._effective_mode(rows)
        self._user_mode = "gradient" if current == "table" else "table"
        self._refresh()

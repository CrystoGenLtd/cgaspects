from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
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

    Every discrete legend entry also carries a "Show" checkbox: unticking an
    entry hides all points of that colour in the 3-D view (and the point count
    in the Crystal Information panel updates to match).  The "Show All Colours"
    button clears the filter.
    """

    _TABLE_THRESHOLD = 10  # switch to gradient above this many unique values

    # Emitted when the user picks a new colour for a legend entry.
    # Payload: (mode, key, (r, g, b))
    #   mode  — "atom" | "docking_shell"
    #   key   — element symbol (str) or shell_id (int)
    #   color — RGB tuple of floats in [0, 1], or None to reset
    colorOverrideRequested = Signal(str, object, object)

    # Emitted when the visible-colour filter changes.
    # Payload: set of visible legend keys, or None to show every colour.
    filterChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Colour Legend")
        self.setModal(False)
        self.resize(300, 340)

        self._info = None
        self._user_mode = None  # None = auto, "table" or "gradient" = user override

        # Filter state — set of visible legend keys, or None (all visible).
        self._visible_keys: set | None = None
        self._filter_domain: tuple = ()  # sorted keys the current filter applies to
        self._current_keys: list = []  # keys of the rows currently displayed

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

        # Show-All-Colours button — only shown when a colour filter is active
        self._show_all_btn = QPushButton("Show All Colours")
        self._show_all_btn.setFixedHeight(24)
        self._show_all_btn.clicked.connect(self._on_show_all)
        self._show_all_btn.hide()
        layout.addWidget(self._show_all_btn)

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
        self._current_keys = [] if is_single else [key for key, _ in rows]

        # Reset the filter when the set of legend keys changes (e.g. mode switch);
        # the GL widget drops its own filter on the same event.
        domain = tuple(sorted(map(str, self._current_keys)))
        if domain != self._filter_domain:
            self._filter_domain = domain
            self._visible_keys = None

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
        self._update_filter_controls()

    def _build_table(self, rows, editable: bool, mode: str | None):
        is_single = (not rows) or rows[0][0] is None
        filterable = not is_single

        # Dynamic column layout: [Show?] Label Colour [Reset?]
        cols = (["show"] if filterable else []) + ["label", "colour"]
        if editable:
            cols.append("reset")
        col_index = {name: i for i, name in enumerate(cols)}
        header_text = {"show": "", "label": "Label", "colour": "Colour", "reset": ""}
        swatch_col = col_index["colour"]

        table = QTableWidget(len(rows), len(cols))
        table.setHorizontalHeaderLabels([header_text[c] for c in cols])
        table.horizontalHeader().setStretchLastSection(not editable)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)

        for i, (key, rgb) in enumerate(rows):
            if filterable:
                checkbox = QCheckBox()
                checkbox.setChecked(self._visible_keys is None or key in self._visible_keys)
                checkbox.setToolTip("Show points of this colour")
                checkbox.toggled.connect(
                    lambda checked, k=key: self._on_filter_toggled(k, checked)
                )
                holder = QWidget()
                hl = QHBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.setAlignment(Qt.AlignCenter)
                hl.addWidget(checkbox)
                table.setCellWidget(i, col_index["show"], holder)

            label = "Single Colour" if key is None else str(key)
            table.setItem(i, col_index["label"], QTableWidgetItem(label))

            swatch = QTableWidgetItem()
            swatch.setBackground(QColor.fromRgbF(*rgb))
            if editable:
                swatch.setToolTip("Double-click to change colour")
            table.setItem(i, swatch_col, swatch)

            if editable:
                reset_btn = QPushButton("Reset")
                reset_btn.setFixedHeight(20)
                # Capture key and mode for the closure
                reset_btn.clicked.connect(
                    lambda checked=False, k=key, m=mode: self._on_reset_row(m, k)
                )
                table.setCellWidget(i, col_index["reset"], reset_btn)

        if editable:
            table.cellDoubleClicked.connect(
                lambda row, col, r=rows, m=mode, sc=swatch_col: self._on_cell_double_clicked(
                    row, col, r, m, sc
                )
            )

        table.resizeColumnsToContents()
        return table

    def _on_cell_double_clicked(self, row: int, col: int, rows: list, mode: str, swatch_col: int):
        if col != swatch_col:  # only swatch column
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

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _on_filter_toggled(self, key, checked: bool):
        domain = set(self._current_keys)
        visible = set(domain) if self._visible_keys is None else set(self._visible_keys)
        if checked:
            visible.add(key)
        else:
            visible.discard(key)

        if visible == domain:
            self._visible_keys = None
            self.filterChanged.emit(None)
        else:
            self._visible_keys = visible
            self.filterChanged.emit(set(visible))
        self._update_filter_controls()

    def _on_show_all(self):
        if self._visible_keys is None:
            return
        self._visible_keys = None
        self.filterChanged.emit(None)
        self._refresh()  # re-tick every checkbox

    def _update_filter_controls(self):
        self._show_all_btn.setVisible(self._visible_keys is not None)

    def _on_toggle(self):
        if self._info is None:
            return
        rows = self._info["rows"]
        current = self._effective_mode(rows)
        self._user_mode = "gradient" if current == "table" else "table"
        self._refresh()

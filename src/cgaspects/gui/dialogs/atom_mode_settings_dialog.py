"""Dialog for per-element colour / size overrides in Atom view mode."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QColorDialog, QDialog, QDoubleSpinBox, QGroupBox,
                               QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QToolButton, QVBoxLayout,
                               QWidget)

DEFAULT_BOND_RADIUS = 0.20   # Ångströms


def _color_icon(color: QColor, size: int = 20) -> QIcon:
    px = QPixmap(size, size)
    px.fill(color)
    return QIcon(px)


def _qcolor_from_float(rgb: tuple) -> QColor:
    r, g, b = rgb
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def _float_from_qcolor(c: QColor) -> tuple:
    return (c.redF(), c.greenF(), c.blueF())


class _ElementRow(QWidget):
    """One row: symbol | colour button | radius spin-box (Å) | Reset button."""

    changed = Signal()

    def __init__(self, symbol: str, default_color: tuple, default_radius: float, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self._default_color = _qcolor_from_float(default_color)
        self._default_radius = default_radius
        self._color = QColor(self._default_color)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        # Symbol label
        lbl = QLabel(f"<b>{symbol}</b>")
        lbl.setFixedWidth(28)
        row.addWidget(lbl)

        # Colour swatch / picker button
        self._color_btn = QToolButton()
        self._color_btn.setIcon(_color_icon(self._color))
        self._color_btn.setFixedSize(28, 22)
        self._color_btn.setToolTip("Click to change colour")
        self._color_btn.clicked.connect(self._pick_color)
        row.addWidget(self._color_btn)

        # Radius spin-box in Ångströms
        radius_lbl = QLabel("Radius (Å):")
        radius_lbl.setFixedWidth(72)
        row.addWidget(radius_lbl)

        self._radius_spin = QDoubleSpinBox()
        self._radius_spin.setRange(0.10, 10.0)
        self._radius_spin.setSingleStep(0.05)
        self._radius_spin.setDecimals(2)
        self._radius_spin.setSuffix(" Å")
        self._radius_spin.setValue(default_radius)
        self._radius_spin.setFixedWidth(82)
        self._radius_spin.setToolTip(f"VdW radius; default {default_radius:.2f} Å")
        self._radius_spin.valueChanged.connect(lambda _: self.changed.emit())
        row.addWidget(self._radius_spin)

        # Reset button — resets both colour and radius
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(50)
        reset_btn.setToolTip("Reset colour and radius to CPK defaults")
        reset_btn.clicked.connect(self.reset)
        row.addWidget(reset_btn)

        row.addStretch()

    # ------------------------------------------------------------------
    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self, f"Colour for {self.symbol}")
        if c.isValid():
            self._color = c
            self._color_btn.setIcon(_color_icon(self._color))
            self.changed.emit()

    def reset(self):
        """Reset colour and radius to CPK/VdW defaults."""
        self._color = QColor(self._default_color)
        self._color_btn.setIcon(_color_icon(self._color))
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(self._default_radius)
        self._radius_spin.blockSignals(False)
        self.changed.emit()

    # ------------------------------------------------------------------
    def color_float(self) -> tuple:
        return _float_from_qcolor(self._color)

    def is_color_overridden(self) -> bool:
        return self._color != self._default_color

    def radius(self) -> float:
        return self._radius_spin.value()

    def is_radius_overridden(self) -> bool:
        return abs(self._radius_spin.value() - self._default_radius) > 1e-4

    def set_state(self, color_float: tuple | None, radius: float | None):
        """Restore saved state (called on populate)."""
        if color_float is not None:
            self._color = _qcolor_from_float(color_float)
            self._color_btn.setIcon(_color_icon(self._color))
        r = radius if radius is not None else self._default_radius
        self._radius_spin.blockSignals(True)
        self._radius_spin.setValue(r)
        self._radius_spin.blockSignals(False)


class AtomModeSettingsDialog(QDialog):
    """Non-modal dialog.  Emits ``settingsChanged`` whenever the user tweaks anything.

    Signal args:
        color_overrides  – {symbol: (r, g, b)}  only for overridden elements
        radius_overrides – {symbol: float (Å)}   only for overridden elements
        bond_radius      – float (Å)
    """

    settingsChanged = Signal(dict, dict, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atom Mode Settings")
        self.setModal(False)
        self.setMinimumWidth(460)

        self._rows: dict[str, _ElementRow] = {}
        self._live = True

        outer = QVBoxLayout(self)

        # ── Per-element section ────────────────────────────────────────
        elements_group = QGroupBox("Elements (currently in view)")
        elements_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        eg_layout = QVBoxLayout(elements_group)
        eg_layout.setSpacing(2)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(180)
        scroll_contents = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_contents)
        self._scroll_layout.setSpacing(0)
        self._scroll_layout.addStretch()
        scroll_area.setWidget(scroll_contents)
        eg_layout.addWidget(scroll_area)
        outer.addWidget(elements_group)

        # ── Bonds present in structure ─────────────────────────────────
        self._bonds_group = QGroupBox("Bonds in structure")
        bonds_outer = QVBoxLayout(self._bonds_group)
        bonds_outer.setSpacing(2)
        self._bonds_label = QLabel("No bond data loaded.")
        self._bonds_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._bonds_label.setWordWrap(True)
        bonds_outer.addWidget(self._bonds_label)
        outer.addWidget(self._bonds_group)

        # ── Bond radius ────────────────────────────────────────────────
        bond_group = QGroupBox("Bond radius")
        bond_layout = QHBoxLayout(bond_group)

        bond_layout.addWidget(QLabel("Radius (Å):"))

        self._bond_spin = QDoubleSpinBox()
        self._bond_spin.setRange(0.01, 5.0)
        self._bond_spin.setSingleStep(0.01)
        self._bond_spin.setDecimals(2)
        self._bond_spin.setSuffix(" Å")
        self._bond_spin.setValue(DEFAULT_BOND_RADIUS)
        self._bond_spin.setFixedWidth(82)
        self._bond_spin.setToolTip(f"Bond cylinder radius in Ångströms (default {DEFAULT_BOND_RADIUS:.2f} Å)")
        self._bond_spin.valueChanged.connect(self._emit_if_live)
        bond_layout.addWidget(self._bond_spin)

        self._bond_slider = QSlider(Qt.Horizontal)
        self._bond_slider.setRange(1, 500)   # × 0.01 Å → 0.01 … 5.0 Å
        self._bond_slider.setValue(int(DEFAULT_BOND_RADIUS * 100))
        self._bond_slider.setToolTip("Drag to adjust bond radius")
        # Keep spin and slider in sync
        self._bond_slider.valueChanged.connect(self._bond_slider_moved)
        self._bond_spin.valueChanged.connect(self._bond_spin_changed)
        bond_layout.addWidget(self._bond_slider)

        bond_reset_btn = QPushButton("Reset")
        bond_reset_btn.setFixedWidth(50)
        bond_reset_btn.setToolTip(f"Reset to default ({DEFAULT_BOND_RADIUS:.2f} Å)")
        bond_reset_btn.clicked.connect(lambda: self._bond_spin.setValue(DEFAULT_BOND_RADIUS))
        bond_layout.addWidget(bond_reset_btn)

        outer.addWidget(bond_group)

        # ── Bottom buttons ─────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._live_btn = QPushButton("Live update: ON")
        self._live_btn.setCheckable(True)
        self._live_btn.setChecked(True)
        self._live_btn.toggled.connect(self._toggle_live)
        btn_layout.addWidget(self._live_btn)

        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.setToolTip("Reset all elements and bond radius to defaults")
        reset_all_btn.clicked.connect(self.reset_all)
        btn_layout.addWidget(reset_all_btn)

        btn_layout.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setDefault(True)
        apply_btn.clicked.connect(self._emit)
        btn_layout.addWidget(apply_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)

        outer.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Bond spin / slider sync
    # ------------------------------------------------------------------

    def _bond_slider_moved(self, v: int):
        self._bond_spin.blockSignals(True)
        self._bond_spin.setValue(v / 100.0)
        self._bond_spin.blockSignals(False)
        self._emit_if_live()

    def _bond_spin_changed(self, v: float):
        self._bond_slider.blockSignals(True)
        self._bond_slider.setValue(int(v * 100))
        self._bond_slider.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(
        self,
        elements: list[str],
        color_overrides: dict,
        radius_overrides: dict,
        bond_radius: float,
        bond_summary: dict | None = None,
    ):
        """Rebuild per-element rows for the given element symbols."""
        from ...utils.periodic_table import get_atom_color, get_atom_radius

        # Remove rows for elements no longer present
        for sym in list(self._rows.keys()):
            if sym not in elements:
                row = self._rows.pop(sym)
                self._scroll_layout.removeWidget(row)
                row.deleteLater()

        # Add / update rows
        for sym in elements:
            if sym not in self._rows:
                row = _ElementRow(sym, get_atom_color(sym), get_atom_radius(sym), self)
                row.changed.connect(self._emit_if_live)
                self._scroll_layout.insertWidget(self._scroll_layout.count() - 1, row)
                self._rows[sym] = row
            self._rows[sym].set_state(
                color_overrides.get(sym),
                radius_overrides.get(sym),
            )

        # Bond radius
        self._bond_spin.blockSignals(True)
        self._bond_slider.blockSignals(True)
        self._bond_spin.setValue(bond_radius)
        self._bond_slider.setValue(int(bond_radius * 100))
        self._bond_spin.blockSignals(False)
        self._bond_slider.blockSignals(False)

        # Bond connectivity summary
        if bond_summary:
            lines = []
            for (s1, s2), count in sorted(bond_summary.items()):
                lines.append(f"  {s1} — {s2}  ×{count}")
            self._bonds_label.setText("\n".join(lines))
        else:
            self._bonds_label.setText("No bonds found in structure file.")

    def reset_all(self):
        for row in self._rows.values():
            row.reset()
        self._bond_spin.setValue(DEFAULT_BOND_RADIUS)
        self._emit_if_live()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _toggle_live(self, checked: bool):
        self._live = checked
        self._live_btn.setText("Live update: ON" if checked else "Live update: OFF")
        if checked:
            self._emit()

    def _emit_if_live(self, *_):
        if self._live:
            self._emit()

    def _emit(self):
        color_overrides = {}
        radius_overrides = {}
        for sym, row in self._rows.items():
            if row.is_color_overridden():
                color_overrides[sym] = row.color_float()
            if row.is_radius_overridden():
                radius_overrides[sym] = row.radius()
        self.settingsChanged.emit(color_overrides, radius_overrides, self._bond_spin.value())

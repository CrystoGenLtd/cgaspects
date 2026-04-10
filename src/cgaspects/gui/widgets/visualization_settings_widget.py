from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

MARGINS = (5, 5, 5, 5)


class LabelledCheckBox(QWidget):
    valueChanged = Signal()

    def __init__(self, label, state=False, parent=None):
        super().__init__(parent)

        self.label = QLabel(label)

        self.checkBox = QCheckBox()
        self.setCheckState(state)

        self.checkBox.stateChanged.connect(lambda x: self.valueChanged.emit())

        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(*MARGINS)
        layout.addWidget(self.label)
        layout.addWidget(self.checkBox)
        self.setLayout(layout)

    @property
    def value(self):
        return self.checkState() == Qt.CheckState.Checked

    def setCheckState(self, value):
        self.checkBox.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)

    def checkState(self):
        return self.checkBox.checkState()


class LabelledComboBox(QWidget):
    valueChanged = Signal()

    def __init__(self, label, options, parent=None):
        super().__init__(parent)

        self.label = QLabel(label)

        self.options = options

        self.comboBox = QComboBox()
        self.comboBox.addItems(options)

        # Connect signals to slots
        self.comboBox.currentIndexChanged.connect(self.currentIndexChanged)

        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(*MARGINS)
        layout.addWidget(self.label)
        layout.addWidget(self.comboBox)
        self.setLayout(layout)

    def setValue(self, value):
        idx = self.options.index(value)
        if idx > -1:
            self.comboBox.setCurrentIndex(idx)

    @property
    def value(self):
        return self.options[self.comboBox.currentIndex()]

    def currentIndex(self):
        return self.comboBox.currentIndex()

    def currentIndexChanged(self, idx):
        self.valueChanged.emit()

    def setOptions(self, options, default=None):
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.options = list(options)
        self.comboBox.addItems(self.options)
        if default is not None and default in self.options:
            self.comboBox.setCurrentIndex(self.options.index(default))
        elif self.options:
            self.comboBox.setCurrentIndex(0)
        self.comboBox.blockSignals(False)


class LabelledComboBoxWithColorPicker(QWidget):
    """A combo box with an optional color picker that appears on the same row."""

    valueChanged = Signal()
    colorChanged = Signal()

    def __init__(self, label, options, parent=None):
        super().__init__(parent)

        self.label = QLabel(label)
        self.options = options

        self.comboBox = QComboBox()
        self.comboBox.addItems(options)
        self.comboBox.currentIndexChanged.connect(self.currentIndexChanged)

        # Color picker button (initially hidden)
        self.colorButton = QToolButton()
        self.colorButton.clicked.connect(self.openColorDialog)
        self.colorButton.hide()

        # Default color
        self.color = QColor(128, 128, 128)
        self._updateColorIcon()

        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(*MARGINS)
        layout.addWidget(self.label)
        layout.addWidget(self.comboBox)
        layout.addWidget(self.colorButton)
        self.setLayout(layout)

    def _updateColorIcon(self):
        """Update the color button icon to match the current color."""
        self.icon = create_colored_icon(self.color)
        self.colorButton.setIcon(self.icon)

    def setColor(self, color):
        """Set the color for the color picker."""
        self.color = QColor(color)
        self._updateColorIcon()
        self.colorChanged.emit()

    def getColor(self):
        """Get the current color."""
        return self.color

    def openColorDialog(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(self.color)
        if color.isValid():
            self.setColor(color)

    def showColorPicker(self):
        """Show the color picker button."""
        self.colorButton.show()

    def hideColorPicker(self):
        """Hide the color picker button."""
        self.colorButton.hide()

    def setValue(self, value):
        idx = self.options.index(value)
        if idx > -1:
            self.comboBox.setCurrentIndex(idx)

    @property
    def value(self):
        return self.options[self.comboBox.currentIndex()]

    def currentIndex(self):
        return self.comboBox.currentIndex()

    def currentIndexChanged(self, idx):
        self.valueChanged.emit()

    def setOptions(self, options, default=None):
        self.comboBox.blockSignals(True)
        self.comboBox.clear()
        self.options = list(options)
        self.comboBox.addItems(self.options)
        if default is not None and default in self.options:
            self.comboBox.setCurrentIndex(self.options.index(default))
        elif self.options:
            self.comboBox.setCurrentIndex(0)
        self.comboBox.blockSignals(False)


def create_colored_icon(color, size=(50, 50)):
    pixmap = QPixmap(*size)
    pixmap.fill(color)
    return QIcon(pixmap)


class LabelledColorSelector(QWidget):
    valueChanged = Signal()

    def __init__(self, label, color=Qt.red, parent=None):
        super().__init__(parent)
        self.label = QLabel(label)

        self.button = QToolButton()
        self.button.clicked.connect(self.openColorDialog)

        self.setValue(color)

        layout = QHBoxLayout()
        layout.setContentsMargins(*MARGINS)
        layout.addWidget(self.label)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def setValue(self, color):
        self.color = QColor(color)
        self.color = color
        self.icon = create_colored_icon(self.color)
        self.button.setIcon(self.icon)
        self.valueChanged.emit()

    def openColorDialog(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setValue(color)

    @property
    def value(self):
        return self.color


class LabelledDoubleSlider(QWidget):
    valueChanged = Signal()

    def __init__(self, label, parent=None, vrange=(0, 1), steps=10):
        super().__init__(parent)

        self.label = QLabel(label)

        self.minimum, self.maximum = vrange
        self.steps = steps

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.steps)
        self.slider.setSingleStep(self.step)

        # Connect signals to slots
        self.slider.valueChanged.connect(self.valueChanged)

        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(*MARGINS)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    @property
    def step(self):
        return (self.maximum - self.minimum) / self.steps

    def setValue(self, value):
        slider_pos = round((value - self.minimum) / self.step)
        self.slider.setValue(int(max(0, min(self.steps, slider_pos))))

    @property
    def value(self):
        return self.minimum + self.slider.value() * self.step


class VisualizationSettingsWidget(QWidget):
    settingsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()
        layout.setSpacing(5)
        self.widgets = {}

        w = LabelledDoubleSlider("Point Size", vrange=(0.5, 20.0), steps=20, parent=self)
        w.setValue(6.0)

        self.widgets["Point Size"] = w
        w.valueChanged.connect(self.settingsChanged)

        # self.widgets["Point Type"] = LabelledComboBox("Point Type", ("Point", "Sphere"))

        # TODO avoid hard coding these
        w = LabelledComboBox(
            "Color Map",
            (
                "Viridis",
                "Cividis",
                "Plasma",
                "Inferno",
                "Magma",
                "Twilight",
                "HSV",
            ),
        )

        self.widgets["Color Map"] = w
        w.valueChanged.connect(self.settingsChanged)

        w = LabelledComboBoxWithColorPicker(
            "Color By",
            (
                "Layer",
                "Atom/Molecule Type",
                "Atom/Molecule Number",
                "Single Colour",
                "Site Number",
                "Particle Energy",
            ),
        )
        self.widgets["Color By"] = w
        w.valueChanged.connect(self._on_color_by_changed)
        w.valueChanged.connect(self.settingsChanged)
        w.colorChanged.connect(self.settingsChanged)

        w = LabelledColorSelector("Background Color", QColor(Qt.white))
        self.widgets["Background Color"] = w
        w.valueChanged.connect(self.settingsChanged)

        w = LabelledComboBox(
            "Style",
            (
                "Spheres",
                "Points",
                "Convex Hull",
                "Atoms",
                "Unit Cell",
                "Docking",
                "Docking Atoms",
            ),
        )
        self.widgets["Style"] = w
        w.valueChanged.connect(self.settingsChanged)

        w = LabelledCheckBox("Show Mesh Edges", False)
        self.widgets["Show Mesh Edges"] = w
        w.valueChanged.connect(self.settingsChanged)

        w = LabelledDoubleSlider("Frame Rate", vrange=(0.5, 50), steps=50, parent=self)
        w.setValue(15)

        self.widgets["Frame Rate"] = w
        w.valueChanged.connect(self.settingsChanged)

        for w in self.widgets.values():
            layout.addWidget(w)

        self.setLayout(layout)

    def values(self):
        return {"Point Size": self.pointSizeSlider.value}

    def fps(self):
        return self.widgets["Frame Rate"].value

    def setAvailableOptions(self, key, options):
        if key not in self.widgets:
            return
        w = self.widgets[key]
        if isinstance(w, LabelledComboBox):
            w.setOptions(options)

    def setEnabled(self, enabled=True):
        for v in self.widgets.values():
            v.setEnabled(enabled)

    def settings(self):
        settings_dict = {}
        for k, w in self.widgets.items():
            settings_dict[k] = w.value

        # Add the single color from the Color By widget
        color_by_widget = self.widgets["Color By"]
        if hasattr(color_by_widget, "getColor"):
            settings_dict["Single Color"] = color_by_widget.getColor()

        return settings_dict

    def _on_color_by_changed(self):
        """Show/hide the single color picker based on Color By selection."""
        color_by_widget = self.widgets["Color By"]
        color_by = color_by_widget.value
        if color_by == "Single Colour":
            color_by_widget.showColorPicker()
        else:
            color_by_widget.hideColorPicker()

    def setColorByOptions(self, options, default=None):
        """Replace the Color By combo options (called when render style changes)."""
        w = self.widgets.get("Color By")
        if w is not None:
            w.setOptions(options, default)
            self._on_color_by_changed()

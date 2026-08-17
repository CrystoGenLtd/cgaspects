from numbers import Real

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QSlider, QWidget)


class SimulationVariableSlider(QWidget):
    valueChanged = Signal(object)

    def __init__(self, label, values, parent=None):
        super().__init__(parent)

        self.values = list(values)
        self.numeric = all(
            isinstance(value, Real) and not isinstance(value, bool) for value in self.values
        )

        self.label = QLabel(label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(len(self.values) - 1)

        if self.numeric:
            self.editor = QDoubleSpinBox()
            self.editor.setRange(min(self.values), max(self.values))
            self.editor.valueChanged.connect(self.setValue)
        else:
            self.editor = QComboBox()
            self.editor.addItems([str(value).strip() for value in self.values])
            self.editor.currentIndexChanged.connect(self.setIndex)

        self.variableValue = self.values[0]

        self.slider.valueChanged.connect(self.setIndex)

        # Layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.editor)
        self.setLayout(layout)

    def _calculate_step(self, index):
        if len(self.values) < 2:
            return 0.0
        if index < len(self.values) - 1:
            return self.values[index + 1] - self.values[index]
        return self.values[index] - self.values[index - 1]

    def setValue(self, value):
        if value in self.values:
            self.setIndex(self.values.index(value))

    def setIndex(self, index):
        if not 0 <= index < len(self.values):
            return

        self.variableValue = self.values[index]

        with QSignalBlocker(self.slider):
            self.slider.setValue(index)

        with QSignalBlocker(self.editor):
            if self.numeric:
                self.editor.setValue(self.variableValue)
                self.editor.setSingleStep(self._calculate_step(index))
            else:
                self.editor.setCurrentIndex(index)

        self.valueChanged.emit(self.variableValue)

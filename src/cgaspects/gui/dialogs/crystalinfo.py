from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget
from . import crystalinfo_ui


class CrystalInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = crystalinfo_ui.Ui_CrystalInfoWidget()
        self.ui.setupUi(self)

        self._countLabel = QLabel("Points")
        self._countValueLabel = QLabel("N/A")
        self._countValueLabel.setAlignment(Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.ui.gridLayout.addWidget(self._countLabel, 6, 0, 1, 1)
        self.ui.gridLayout.addWidget(self._countValueLabel, 6, 1, 1, 1)

    def update(self, crystal_info):
        self.setEnabled(True)

        def fmt(val):
            return f"{val:.2f}" if val is not None else "N/A"

        self.ui.ar1ValueLabel.setText(fmt(crystal_info.aspectRatio1))
        self.ui.ar2ValueLabel.setText(fmt(crystal_info.aspectRatio2))
        self.ui.shapeClassValueLabel.setText(f"{crystal_info.shapeClass}")
        self.ui.saVolRatioValueLabel.setText(fmt(crystal_info.surfaceAreaVolumeRatio))
        self.ui.saValueLabel.setText(fmt(crystal_info.surfaceArea))
        self.ui.volValueLabel.setText(fmt(crystal_info.volume))

        count = crystal_info.pointCount
        self._countLabel.setText(getattr(crystal_info, "countLabel", "Points"))
        self._countValueLabel.setText(str(count) if count is not None else "N/A")

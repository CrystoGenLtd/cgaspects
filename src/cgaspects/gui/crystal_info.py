from PySide6.QtCore import QObject


class CrystalInfo(QObject):
    aspectRatio1 = 1.0
    aspectRatio2 = 1.0
    shapeClass = "N/A"
    surfaceAreaVolumeRatio = 1.0
    surfaceArea = 0.0
    volume = 0.0
    pointCount = None
    countLabel = "Points"

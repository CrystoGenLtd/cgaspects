# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'crystalinfo.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QLabel, QSizePolicy,
    QWidget)

class Ui_CrystalInfoWidget(object):
    def setupUi(self, CrystalInfoWidget):
        if not CrystalInfoWidget.objectName():
            CrystalInfoWidget.setObjectName(u"CrystalInfoWidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(CrystalInfoWidget.sizePolicy().hasHeightForWidth())
        CrystalInfoWidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(CrystalInfoWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.ar1Label = QLabel(CrystalInfoWidget)
        self.ar1Label.setObjectName(u"ar1Label")

        self.gridLayout.addWidget(self.ar1Label, 0, 0, 1, 1)

        self.ar1ValueLabel = QLabel(CrystalInfoWidget)
        self.ar1ValueLabel.setObjectName(u"ar1ValueLabel")
        self.ar1ValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.ar1ValueLabel, 0, 1, 1, 1)

        self.ar2Label = QLabel(CrystalInfoWidget)
        self.ar2Label.setObjectName(u"ar2Label")

        self.gridLayout.addWidget(self.ar2Label, 1, 0, 1, 1)

        self.ar2ValueLabel = QLabel(CrystalInfoWidget)
        self.ar2ValueLabel.setObjectName(u"ar2ValueLabel")
        self.ar2ValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.ar2ValueLabel, 1, 1, 1, 1)

        self.shapeClassLabel = QLabel(CrystalInfoWidget)
        self.shapeClassLabel.setObjectName(u"shapeClassLabel")

        self.gridLayout.addWidget(self.shapeClassLabel, 2, 0, 1, 1)

        self.shapeClassValueLabel = QLabel(CrystalInfoWidget)
        self.shapeClassValueLabel.setObjectName(u"shapeClassValueLabel")
        self.shapeClassValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.shapeClassValueLabel, 2, 1, 1, 1)

        self.saVolRatioLabel = QLabel(CrystalInfoWidget)
        self.saVolRatioLabel.setObjectName(u"saVolRatioLabel")

        self.gridLayout.addWidget(self.saVolRatioLabel, 3, 0, 1, 1)

        self.saVolRatioValueLabel = QLabel(CrystalInfoWidget)
        self.saVolRatioValueLabel.setObjectName(u"saVolRatioValueLabel")
        self.saVolRatioValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.saVolRatioValueLabel, 3, 1, 1, 1)

        self.saLabel = QLabel(CrystalInfoWidget)
        self.saLabel.setObjectName(u"saLabel")

        self.gridLayout.addWidget(self.saLabel, 4, 0, 1, 1)

        self.saValueLabel = QLabel(CrystalInfoWidget)
        self.saValueLabel.setObjectName(u"saValueLabel")
        self.saValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.saValueLabel, 4, 1, 1, 1)

        self.volLabel = QLabel(CrystalInfoWidget)
        self.volLabel.setObjectName(u"volLabel")

        self.gridLayout.addWidget(self.volLabel, 5, 0, 1, 1)

        self.volValueLabel = QLabel(CrystalInfoWidget)
        self.volValueLabel.setObjectName(u"volValueLabel")
        self.volValueLabel.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.volValueLabel, 5, 1, 1, 1)


        self.retranslateUi(CrystalInfoWidget)

        QMetaObject.connectSlotsByName(CrystalInfoWidget)
    # setupUi

    def retranslateUi(self, CrystalInfoWidget):
        CrystalInfoWidget.setWindowTitle(QCoreApplication.translate("CrystalInfoWidget", u"Form", None))
        self.ar1Label.setText(QCoreApplication.translate("CrystalInfoWidget", u"Aspect Ratio S:M", None))
        self.ar1ValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
        self.ar2Label.setText(QCoreApplication.translate("CrystalInfoWidget", u"Aspect Ratio M:L", None))
        self.ar2ValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
        self.shapeClassLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"Shape Class", None))
        self.shapeClassValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
        self.saVolRatioLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"Surface Area/Volume Ratio", None))
        self.saVolRatioValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
        self.saLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"Surface Area", None))
        self.saValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
        self.volLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"Volume", None))
        self.volValueLabel.setText(QCoreApplication.translate("CrystalInfoWidget", u"N/A", None))
    # retranslateUi


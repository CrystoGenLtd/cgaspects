# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'plotsavedialog.ui'
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
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpinBox, QVBoxLayout, QWidget)

class Ui_PlotSaveDialog(object):
    def setupUi(self, PlotSaveDialog):
        if not PlotSaveDialog.objectName():
            PlotSaveDialog.setObjectName(u"PlotSaveDialog")
        PlotSaveDialog.resize(400, 300)
        self.verticalLayout = QVBoxLayout(PlotSaveDialog)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox = QGroupBox(PlotSaveDialog)
        self.groupBox.setObjectName(u"groupBox")
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.fileTypeLabel = QLabel(self.groupBox)
        self.fileTypeLabel.setObjectName(u"fileTypeLabel")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.fileTypeLabel)

        self.fileTypeComboBox = QComboBox(self.groupBox)
        self.fileTypeComboBox.setObjectName(u"fileTypeComboBox")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.fileTypeComboBox)

        self.dpiLabel = QLabel(self.groupBox)
        self.dpiLabel.setObjectName(u"dpiLabel")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.dpiLabel)

        self.dpiSpinBox = QSpinBox(self.groupBox)
        self.dpiSpinBox.setObjectName(u"dpiSpinBox")
        self.dpiSpinBox.setMinimum(150)
        self.dpiSpinBox.setMaximum(1200)
        self.dpiSpinBox.setSingleStep(10)
        self.dpiSpinBox.setValue(300)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.dpiSpinBox)

        self.fileLabel = QLabel(self.groupBox)
        self.fileLabel.setObjectName(u"fileLabel")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.fileLabel)

        self.fileLineEdit = QLineEdit(self.groupBox)
        self.fileLineEdit.setObjectName(u"fileLineEdit")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.FieldRole, self.fileLineEdit)

        self.filePushButton = QPushButton(self.groupBox)
        self.filePushButton.setObjectName(u"filePushButton")

        self.formLayout.setWidget(6, QFormLayout.ItemRole.LabelRole, self.filePushButton)

        self.transparentCheckBox = QCheckBox(self.groupBox)
        self.transparentCheckBox.setObjectName(u"transparentCheckBox")
        self.transparentCheckBox.setLayoutDirection(Qt.RightToLeft)

        self.formLayout.setWidget(6, QFormLayout.ItemRole.FieldRole, self.transparentCheckBox)


        self.verticalLayout.addWidget(self.groupBox)

        self.buttonBox = QDialogButtonBox(PlotSaveDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.verticalLayout.addWidget(self.buttonBox)


        self.retranslateUi(PlotSaveDialog)
        self.buttonBox.accepted.connect(PlotSaveDialog.accept)
        self.buttonBox.rejected.connect(PlotSaveDialog.reject)

        QMetaObject.connectSlotsByName(PlotSaveDialog)
    # setupUi

    def retranslateUi(self, PlotSaveDialog):
        PlotSaveDialog.setWindowTitle(QCoreApplication.translate("PlotSaveDialog", u"Dialog", None))
        self.groupBox.setTitle(QCoreApplication.translate("PlotSaveDialog", u"Plot Export Settings", None))
        self.fileTypeLabel.setText(QCoreApplication.translate("PlotSaveDialog", u"File type", None))
#if QT_CONFIG(tooltip)
        self.dpiLabel.setToolTip(QCoreApplication.translate("PlotSaveDialog", u"<html><head/><body><p>DPI describes the resulting resolution of the output image.</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.dpiLabel.setText(QCoreApplication.translate("PlotSaveDialog", u"DPI", None))
        self.fileLabel.setText(QCoreApplication.translate("PlotSaveDialog", u"File destination", None))
        self.filePushButton.setText(QCoreApplication.translate("PlotSaveDialog", u"Browse", None))
        self.transparentCheckBox.setText(QCoreApplication.translate("PlotSaveDialog", u"Transparent background", None))
    # retranslateUi


# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settings.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QLabel,
    QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
    QWidget)

class Ui_settings_Dialog(object):
    def setupUi(self, settings_Dialog):
        if not settings_Dialog.objectName():
            settings_Dialog.setObjectName(u"settings_Dialog")
        settings_Dialog.resize(380, 380)
        font = QFont()
        font.setFamilies([u"Apple Chancery"])
        font.setPointSize(10)
        font.setItalic(True)
        settings_Dialog.setFont(font)
        self.gridLayout = QGridLayout(settings_Dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.settings_scrollArea = QScrollArea(settings_Dialog)
        self.settings_scrollArea.setObjectName(u"settings_scrollArea")
        font1 = QFont()
        font1.setFamilies([u"Apple Chancery"])
        font1.setItalic(False)
        self.settings_scrollArea.setFont(font1)
        self.settings_scrollArea.setFrameShadow(QFrame.Plain)
        self.settings_scrollArea.setLineWidth(0)
        self.settings_scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 354, 354))
        self.verticalLayout_2 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.groupBox = QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox.setObjectName(u"groupBox")
        font2 = QFont()
        font2.setFamilies([u"Arial"])
        font2.setPointSize(12)
        self.groupBox.setFont(font2)
        self.formLayout = QFormLayout(self.groupBox)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.formLayout.setLabelAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.zoom_label = QLabel(self.groupBox)
        self.zoom_label.setObjectName(u"zoom_label")
        font3 = QFont()
        font3.setFamilies([u"Arial"])
        font3.setPointSize(11)
        font3.setBold(True)
        font3.setItalic(False)
        self.zoom_label.setFont(font3)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.zoom_label)

        self.zoom_slider = QSlider(self.groupBox)
        self.zoom_slider.setObjectName(u"zoom_slider")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.zoom_slider.sizePolicy().hasHeightForWidth())
        self.zoom_slider.setSizePolicy(sizePolicy)
        font4 = QFont()
        font4.setFamilies([u"Arial"])
        font4.setPointSize(8)
        font4.setBold(True)
        font4.setItalic(False)
        self.zoom_slider.setFont(font4)
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setTickPosition(QSlider.NoTicks)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.zoom_slider)

        self.pointsize_label = QLabel(self.groupBox)
        self.pointsize_label.setObjectName(u"pointsize_label")
        self.pointsize_label.setFont(font3)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.pointsize_label)

        self.point_slider = QSlider(self.groupBox)
        self.point_slider.setObjectName(u"point_slider")
        self.point_slider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.point_slider)

        self.brightness_label = QLabel(self.groupBox)
        self.brightness_label.setObjectName(u"brightness_label")
        self.brightness_label.setEnabled(False)
        self.brightness_label.setFont(font3)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.brightness_label)

        self.brightness_slider = QSlider(self.groupBox)
        self.brightness_slider.setObjectName(u"brightness_slider")
        self.brightness_slider.setEnabled(False)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.brightness_slider)

        self.resolution_label = QLabel(self.groupBox)
        self.resolution_label.setObjectName(u"resolution_label")
        self.resolution_label.setEnabled(False)
        self.resolution_label.setFont(font3)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.resolution_label)

        self.resolution_slider = QSlider(self.groupBox)
        self.resolution_slider.setObjectName(u"resolution_slider")
        self.resolution_slider.setEnabled(False)
        self.resolution_slider.setOrientation(Qt.Horizontal)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.resolution_slider)

        self.point_type_label = QLabel(self.groupBox)
        self.point_type_label.setObjectName(u"point_type_label")
        self.point_type_label.setEnabled(True)
        self.point_type_label.setFont(font3)

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.point_type_label)

        self.pointtype_comboBox = QComboBox(self.groupBox)
        self.pointtype_comboBox.setObjectName(u"pointtype_comboBox")
        self.pointtype_comboBox.setEnabled(True)
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pointtype_comboBox.sizePolicy().hasHeightForWidth())
        self.pointtype_comboBox.setSizePolicy(sizePolicy1)
        self.pointtype_comboBox.setMinimumSize(QSize(0, 0))
        self.pointtype_comboBox.setStyleSheet(u"")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.pointtype_comboBox)

        self.colour_mode_label = QLabel(self.groupBox)
        self.colour_mode_label.setObjectName(u"colour_mode_label")
        self.colour_mode_label.setFont(font3)

        self.formLayout.setWidget(5, QFormLayout.ItemRole.LabelRole, self.colour_mode_label)

        self.colourmode_comboBox = QComboBox(self.groupBox)
        self.colourmode_comboBox.setObjectName(u"colourmode_comboBox")
        sizePolicy1.setHeightForWidth(self.colourmode_comboBox.sizePolicy().hasHeightForWidth())
        self.colourmode_comboBox.setSizePolicy(sizePolicy1)
        self.colourmode_comboBox.setMinimumSize(QSize(100, 0))
        self.colourmode_comboBox.setStyleSheet(u"")

        self.formLayout.setWidget(5, QFormLayout.ItemRole.FieldRole, self.colourmode_comboBox)

        self.colour_label = QLabel(self.groupBox)
        self.colour_label.setObjectName(u"colour_label")
        self.colour_label.setFont(font3)

        self.formLayout.setWidget(6, QFormLayout.ItemRole.LabelRole, self.colour_label)

        self.colour_comboBox = QComboBox(self.groupBox)
        self.colour_comboBox.setObjectName(u"colour_comboBox")
        sizePolicy1.setHeightForWidth(self.colour_comboBox.sizePolicy().hasHeightForWidth())
        self.colour_comboBox.setSizePolicy(sizePolicy1)
        self.colour_comboBox.setMinimumSize(QSize(100, 0))
        self.colour_comboBox.setStyleSheet(u"")

        self.formLayout.setWidget(6, QFormLayout.ItemRole.FieldRole, self.colour_comboBox)

        self.colour_label_2 = QLabel(self.groupBox)
        self.colour_label_2.setObjectName(u"colour_label_2")
        self.colour_label_2.setFont(font3)

        self.formLayout.setWidget(7, QFormLayout.ItemRole.LabelRole, self.colour_label_2)

        self.bgcolour_comboBox = QComboBox(self.groupBox)
        self.bgcolour_comboBox.setObjectName(u"bgcolour_comboBox")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(1)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.bgcolour_comboBox.sizePolicy().hasHeightForWidth())
        self.bgcolour_comboBox.setSizePolicy(sizePolicy2)
        self.bgcolour_comboBox.setMinimumSize(QSize(100, 0))
        self.bgcolour_comboBox.setStyleSheet(u"")

        self.formLayout.setWidget(7, QFormLayout.ItemRole.FieldRole, self.bgcolour_comboBox)

        self.projection_label = QLabel(self.groupBox)
        self.projection_label.setObjectName(u"projection_label")
        self.projection_label.setFont(font3)

        self.formLayout.setWidget(8, QFormLayout.ItemRole.LabelRole, self.projection_label)

        self.projection_comboBox = QComboBox(self.groupBox)
        self.projection_comboBox.setObjectName(u"projection_comboBox")
        sizePolicy.setHeightForWidth(self.projection_comboBox.sizePolicy().hasHeightForWidth())
        self.projection_comboBox.setSizePolicy(sizePolicy)

        self.formLayout.setWidget(8, QFormLayout.ItemRole.FieldRole, self.projection_comboBox)


        self.verticalLayout_2.addWidget(self.groupBox)

        self.settings_scrollArea.setWidget(self.scrollAreaWidgetContents)

        self.gridLayout.addWidget(self.settings_scrollArea, 0, 0, 1, 1)


        self.retranslateUi(settings_Dialog)

        QMetaObject.connectSlotsByName(settings_Dialog)
    # setupUi

    def retranslateUi(self, settings_Dialog):
        settings_Dialog.setWindowTitle(QCoreApplication.translate("settings_Dialog", u"Settings", None))
        self.groupBox.setTitle(QCoreApplication.translate("settings_Dialog", u"Display Options", None))
        self.zoom_label.setText(QCoreApplication.translate("settings_Dialog", u"Zoom", None))
        self.pointsize_label.setText(QCoreApplication.translate("settings_Dialog", u"Point Size", None))
        self.brightness_label.setText(QCoreApplication.translate("settings_Dialog", u"Brightness", None))
        self.resolution_label.setText(QCoreApplication.translate("settings_Dialog", u"Resolution", None))
        self.point_type_label.setText(QCoreApplication.translate("settings_Dialog", u"Point Type", None))
        self.colour_mode_label.setText(QCoreApplication.translate("settings_Dialog", u"Colour Mode", None))
        self.colour_label.setText(QCoreApplication.translate("settings_Dialog", u"Colour", None))
        self.colour_label_2.setText(QCoreApplication.translate("settings_Dialog", u"Background Colour", None))
        self.projection_label.setText(QCoreApplication.translate("settings_Dialog", u"Projection", None))
    # retranslateUi


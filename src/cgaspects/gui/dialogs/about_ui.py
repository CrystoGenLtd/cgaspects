# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'about.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
    QSizePolicy, QWidget)
from cgaspects.gui.utils import qticons_rc

class Ui_AboutCGDialog(object):
    def setupUi(self, AboutCGDialog):
        if not AboutCGDialog.objectName():
            AboutCGDialog.setObjectName(u"AboutCGDialog")
        AboutCGDialog.resize(533, 330)
        self.horizontalLayout = QHBoxLayout(AboutCGDialog)
        self.horizontalLayout.setSpacing(24)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(AboutCGDialog)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QSize(160, 160))
        self.label.setMaximumSize(QSize(160, 160))
        self.label.setPixmap(QPixmap(u":/app_icons/app_icons/CGAspects.png"))
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)

        self.horizontalLayout.addWidget(self.label)

        self.aboutLabel = QLabel(AboutCGDialog)
        self.aboutLabel.setObjectName(u"aboutLabel")
        self.aboutLabel.setWordWrap(True)
        self.aboutLabel.setOpenExternalLinks(True)

        self.horizontalLayout.addWidget(self.aboutLabel)


        self.retranslateUi(AboutCGDialog)

        QMetaObject.connectSlotsByName(AboutCGDialog)
    # setupUi

    def retranslateUi(self, AboutCGDialog):
        AboutCGDialog.setWindowTitle(QCoreApplication.translate("AboutCGDialog", u"About CGAspects", None))
        self.label.setText("")
        self.aboutLabel.setText(QCoreApplication.translate("AboutCGDialog", u"<html><head/><body><p><span style=\" font-size:24pt; font-weight:696;\">CGAspects</span></p><p>CGAspects is part of the <a href=\"https://crystogen.org/\"><span style=\" font-weight:696; text-decoration: underline; color:#007af4;\">CrystoGen</span></a> (formerly CrystalGrower) suite of programs.</p><p>Authors: <span style=\" font-weight:696;\">Alvin Jenner Walisinghe, Nathan de Bruyn and Peter R. Spackman</span></p><p>The code is publicly available on <a href=\"https://github.com/crystogenltd/cgaspects\"><span style=\" text-decoration: underline; color:#007af4;\">GItHub</span></a> and documentation is available at the website.</p><p>When using <span style=\" font-weight:696;\">CrystoGen</span> to simulate the crystal growth of your materials, please cite it in your scientific publications as follows:</p><p><a href=\"https://www.nature.com/articles/nature21684\"><span style=\" text-decoration: underline; color:#007af4;\">M. W. Anderson et al. </span></a><a href=\"https://www.nature.com/articles/nature21684\"><s"
                        "pan style=\" font-style:italic; text-decoration: underline; color:#007af4;\">Nature, </span></a><a href=\"https://www.nature.com/articles/nature21684\"><span style=\" font-weight:696; text-decoration: underline; color:#007af4;\">544</span></a><a href=\"https://www.nature.com/articles/nature21684\"><span style=\" text-decoration: underline; color:#007af4;\">, 456\u2013459 (2017)</span></a></p><p><a href=\"https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05017b#!divAbstract\"><span style=\" text-decoration: underline; color:#007af4;\">A. R. Hill et al.. </span></a><a href=\"https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05017b#!divAbstract\"><span style=\" font-style:italic; text-decoration: underline; color:#007af4;\">Chem. Sci.</span></a><a href=\"https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05017b#!divAbstract\"><span style=\" text-decoration: underline; color:#007af4;\">, </span></a><a href=\"https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05017b#!divAbstract\">"
                        "<span style=\" font-weight:696; text-decoration: underline; color:#007af4;\">12</span></a><a href=\"https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc05017b#!divAbstract\"><span style=\" text-decoration: underline; color:#007af4;\">, 1126\u20131146 (2021)</span></a></p></body></html>", None))
    # retranslateUi


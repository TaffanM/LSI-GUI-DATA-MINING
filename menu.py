# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'menu.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1353, 753)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(-4, 0, 261, 761))
        self.widget.setStyleSheet("background-color:  #2b9fa4;\n"
"color: #FFFFFF;\n"
"\n"
"\n"
"\n"
"")
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setStyleSheet("QPushButton {\n"
"    font-size: 18px;\n"
"    font-family: \"Poppins\";\n"
"    font-weight: bold;\n"
"    margin: 0px 10px;\n"
"    margin-top: 20px;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setStyleSheet("QPushButton {\n"
"    font-size: 18px;\n"
"    font-family: \"Poppins\";\n"
"    font-weight: bold;\n"
"    margin: 0px 10px;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget)
        self.pushButton_3.setStyleSheet("QPushButton {\n"
"    font-size: 18px;\n"
"    font-family: \"Poppins\";\n"
"    font-weight: bold;\n"
"    margin: 0px 10px;\n"
"}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.widget)
        self.pushButton_4.setStyleSheet("QPushButton {\n"
"    font-size: 18px;\n"
"    font-family: \"Poppins\";\n"
"    font-weight: bold;\n"
"    margin: 0px 10px;\n"
"}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 508, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.pushButton_5 = QtWidgets.QPushButton(self.widget)
        self.pushButton_5.setStyleSheet("QPushButton {\n"
"    font-size: 18px;\n"
"    font-family: \"Poppins\";\n"
"    font-weight: bold;\n"
"    margin: 0px 10px;\n"
"    margin-bottom: 20px;\n"
"}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_2.addWidget(self.pushButton_5)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(250, 0, 1111, 761))
        self.stackedWidget.setStyleSheet("background-color: rgb(76, 181, 185);")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setGeometry(QtCore.QRect(30, 30, 151, 61))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: #FFFFFF")
        self.label_2.setObjectName("label_2")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.label_3 = QtWidgets.QLabel(self.page_2)
        self.label_3.setGeometry(QtCore.QRect(30, 30, 361, 61))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: #FFFFFF")
        self.label_3.setObjectName("label_3")
        self.stackedWidget.addWidget(self.page_2)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.label_4 = QtWidgets.QLabel(self.page_5)
        self.label_4.setGeometry(QtCore.QRect(30, 30, 271, 61))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: #FFFFFF")
        self.label_4.setObjectName("label_4")
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.label_5 = QtWidgets.QLabel(self.page_6)
        self.label_5.setGeometry(QtCore.QRect(30, 30, 271, 61))
        font = QtGui.QFont()
        font.setFamily("Poppins")
        font.setPointSize(36)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: #FFFFFF")
        self.label_5.setObjectName("label_5")
        self.stackedWidget.addWidget(self.page_6)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "LSI (Latest Semantic Index)"))
        self.pushButton.setText(_translate("MainWindow", "Home"))
        self.pushButton_2.setText(_translate("MainWindow", "Preprocessing"))
        self.pushButton_3.setText(_translate("MainWindow", "Stemming"))
        self.pushButton_4.setText(_translate("MainWindow", "Temu Balik"))
        self.pushButton_5.setText(_translate("MainWindow", "Exit"))
        self.label_2.setText(_translate("MainWindow", "Home"))
        self.label_3.setText(_translate("MainWindow", "Preprocessing"))
        self.label_4.setText(_translate("MainWindow", "Stemming"))
        self.label_5.setText(_translate("MainWindow", "Temu Balik"))
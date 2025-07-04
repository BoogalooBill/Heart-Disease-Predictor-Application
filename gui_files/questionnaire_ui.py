# Form implementation generated from reading ui file 'questionnaire_ui.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_questionnaire_screen(object):
    def setupUi(self, questionnaire_screen):
        questionnaire_screen.setObjectName("questionnaire_screen")
        questionnaire_screen.resize(830, 699)
        questionnaire_screen.setMinimumSize(QtCore.QSize(830, 699))
        questionnaire_screen.setMaximumSize(QtCore.QSize(830, 699))
        self.centralwidget = QtWidgets.QWidget(parent=questionnaire_screen)
        self.centralwidget.setObjectName("centralwidget")
        self.questionnaire_scroll_area = QtWidgets.QScrollArea(parent=self.centralwidget)
        self.questionnaire_scroll_area.setGeometry(QtCore.QRect(10, 70, 811, 511))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.questionnaire_scroll_area.sizePolicy().hasHeightForWidth())
        self.questionnaire_scroll_area.setSizePolicy(sizePolicy)
        self.questionnaire_scroll_area.setToolTip("")
        self.questionnaire_scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.questionnaire_scroll_area.setWidgetResizable(True)
        self.questionnaire_scroll_area.setObjectName("questionnaire_scroll_area")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 792, 509))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollAreaWidgetContents.sizePolicy().hasHeightForWidth())
        self.scrollAreaWidgetContents.setSizePolicy(sizePolicy)
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_2 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.age_input = QtWidgets.QSpinBox(parent=self.scrollAreaWidgetContents)
        self.age_input.setGeometry(QtCore.QRect(150, 10, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.age_input.setFont(font)
        self.age_input.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.age_input.setMaximum(120)
        self.age_input.setObjectName("age_input")
        self.label_3 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_3.setGeometry(QtCore.QRect(390, 10, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.sex_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.sex_input.setGeometry(QtCore.QRect(670, 10, 69, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.sex_input.setFont(font)
        self.sex_input.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.sex_input.setEditable(False)
        self.sex_input.setMaxVisibleItems(2)
        self.sex_input.setObjectName("sex_input")
        self.sex_input.addItem("")
        self.sex_input.addItem("")
        self.label_4 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_4.setGeometry(QtCore.QRect(10, 60, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.angina_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.angina_input.setGeometry(QtCore.QRect(598, 60, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.angina_input.setFont(font)
        self.angina_input.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.angina_input.setWhatsThis("")
        self.angina_input.setEditable(False)
        self.angina_input.setMaxVisibleItems(10)
        self.angina_input.setObjectName("angina_input")
        self.angina_input.addItem("")
        self.angina_input.addItem("")
        self.angina_input.addItem("")
        self.angina_input.addItem("")
        self.label_5 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_5.setGeometry(QtCore.QRect(10, 110, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.resting_bp_input = QtWidgets.QSpinBox(parent=self.scrollAreaWidgetContents)
        self.resting_bp_input.setGeometry(QtCore.QRect(690, 110, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.resting_bp_input.setFont(font)
        self.resting_bp_input.setObjectName("resting_bp_input")
        self.resting_bp_input.setMinimum(0)
        self.resting_bp_input.setMaximum(999)
        self.label_6 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_6.setGeometry(QtCore.QRect(10, 160, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.cholesterol_input = QtWidgets.QSpinBox(parent=self.scrollAreaWidgetContents)
        self.cholesterol_input.setGeometry(QtCore.QRect(690, 160, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.cholesterol_input.setFont(font)
        self.cholesterol_input.setObjectName("cholesterol_input")
        self.cholesterol_input.setMinimum(0)
        self.cholesterol_input.setMaximum(999)
        self.label_7 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_7.setGeometry(QtCore.QRect(10, 210, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.angina_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.angina_checkbox.setGeometry(QtCore.QRect(770, 60, 16, 31))
        self.angina_checkbox.setText("")
        self.angina_checkbox.setObjectName("angina_checkbox")
        self.resting_bp_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.resting_bp_checkbox.setGeometry(QtCore.QRect(770, 110, 16, 31))
        self.resting_bp_checkbox.setText("")
        self.resting_bp_checkbox.setObjectName("resting_bp_checkbox")
        self.cholesterol_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.cholesterol_checkbox.setGeometry(QtCore.QRect(770, 160, 16, 31))
        self.cholesterol_checkbox.setText("")
        self.cholesterol_checkbox.setObjectName("cholesterol_checkbox")
        self.blood_sugar_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.blood_sugar_checkbox.setGeometry(QtCore.QRect(770, 210, 16, 31))
        self.blood_sugar_checkbox.setText("")
        self.blood_sugar_checkbox.setObjectName("blood_sugar_checkbox")
        self.blood_sugar_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.blood_sugar_input.setGeometry(QtCore.QRect(690, 210, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.blood_sugar_input.setFont(font)
        self.blood_sugar_input.setObjectName("blood_sugar_input")
        self.blood_sugar_input.addItem("")
        self.blood_sugar_input.addItem("")
        self.label_8 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_8.setGeometry(QtCore.QRect(10, 260, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.ecg_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.ecg_input.setGeometry(QtCore.QRect(500, 260, 241, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.ecg_input.setFont(font)
        self.ecg_input.setObjectName("ecg_input")
        self.ecg_input.addItem("")
        self.ecg_input.addItem("")
        self.ecg_input.addItem("")
        self.ecg_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.ecg_checkbox.setGeometry(QtCore.QRect(770, 260, 16, 31))
        self.ecg_checkbox.setText("")
        self.ecg_checkbox.setObjectName("ecg_checkbox")
        self.label_9 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_9.setGeometry(QtCore.QRect(10, 310, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.heart_rate_input = QtWidgets.QSpinBox(parent=self.scrollAreaWidgetContents)
        self.heart_rate_input.setGeometry(QtCore.QRect(690, 310, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.heart_rate_input.setFont(font)
        self.heart_rate_input.setObjectName("heart_rate_input")
        self.heart_rate_input.setMinimum(0)
        self.heart_rate_input.setMaximum(999)
        self.heart_rate_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.heart_rate_checkbox.setGeometry(QtCore.QRect(770, 310, 16, 31))
        self.heart_rate_checkbox.setText("")
        self.heart_rate_checkbox.setObjectName("heart_rate_checkbox")
        self.label_10 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_10.setGeometry(QtCore.QRect(10, 360, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.exercise_angina_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.exercise_angina_input.setGeometry(QtCore.QRect(690, 360, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.exercise_angina_input.setFont(font)
        self.exercise_angina_input.setObjectName("exercise_angina_input")
        self.exercise_angina_input.addItem("")
        self.exercise_angina_input.addItem("")
        self.exercise_angina_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.exercise_angina_checkbox.setGeometry(QtCore.QRect(770, 360, 16, 31))
        self.exercise_angina_checkbox.setText("")
        self.exercise_angina_checkbox.setObjectName("exercise_angina_checkbox")
        self.label_11 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_11.setGeometry(QtCore.QRect(10, 410, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.oldpeak_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.oldpeak_checkbox.setGeometry(QtCore.QRect(770, 410, 16, 31))
        self.oldpeak_checkbox.setText("")
        self.oldpeak_checkbox.setObjectName("oldpeak_checkbox")
        self.oldpeak_input = QtWidgets.QDoubleSpinBox(parent=self.scrollAreaWidgetContents)
        self.oldpeak_input.setGeometry(QtCore.QRect(670, 410, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.oldpeak_input.setFont(font)
        self.oldpeak_input.setObjectName("oldpeak_input")
        self.label_12 = QtWidgets.QLabel(parent=self.scrollAreaWidgetContents)
        self.label_12.setGeometry(QtCore.QRect(10, 460, 651, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.slope_input = QtWidgets.QComboBox(parent=self.scrollAreaWidgetContents)
        self.slope_input.setGeometry(QtCore.QRect(610, 460, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.slope_input.setFont(font)
        self.slope_input.setObjectName("slope_input")
        self.slope_input.addItem("")
        self.slope_input.addItem("")
        self.slope_input.addItem("")
        self.slope_checkbox = QtWidgets.QCheckBox(parent=self.scrollAreaWidgetContents)
        self.slope_checkbox.setGeometry(QtCore.QRect(770, 460, 16, 31))
        self.slope_checkbox.setText("")
        self.slope_checkbox.setObjectName("slope_checkbox")
        self.questionnaire_scroll_area.setWidget(self.scrollAreaWidgetContents)
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 811, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.questionnaire_submit_button = QtWidgets.QPushButton(parent=self.centralwidget)
        self.questionnaire_submit_button.setGeometry(QtCore.QRect(350, 590, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        self.questionnaire_submit_button.setFont(font)
        self.questionnaire_submit_button.setObjectName("nextButton")
        questionnaire_screen.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=questionnaire_screen)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 830, 21))
        self.menubar.setObjectName("menubar")
        questionnaire_screen.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=questionnaire_screen)
        self.statusbar.setObjectName("statusbar")
        questionnaire_screen.setStatusBar(self.statusbar)

        self.retranslateUi(questionnaire_screen)
        self.sex_input.setCurrentIndex(0)
        self.angina_input.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(questionnaire_screen)

    def retranslateUi(self, questionnaire_screen):
        _translate = QtCore.QCoreApplication.translate
        questionnaire_screen.setWindowTitle(_translate("questionnaire_screen", "Heart Disease Predictor by S&N Health - Questionnaire"))
        self.label_2.setText(_translate("questionnaire_screen", "Enter your age:"))
        self.label_3.setText(_translate("questionnaire_screen", "What was your sex at birth? (M/F)"))
        self.sex_input.setItemText(0, _translate("questionnaire_screen", "Male"))
        self.sex_input.setItemText(1, _translate("questionnaire_screen", "Female"))
        self.label_4.setText(_translate("questionnaire_screen", "Do you have chest pain? If so, list the type of angina you\'re experiencing. If not, select \"Asymptomatic\"."))
        self.angina_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>Typical&quot; means standard symptoms of angina such as chest pain or tightness. &quot;Atypical&quot; means that the pain/tightness only exists for a few seconds. &quot;Non-anginal&quot; means you\'re experiencing chest pain but have verified that it is not caused by the heart. &quot;Asymptomatic&quot; means that you have no chest pain.</p></body></html>"))
        self.angina_input.setItemText(0, _translate("questionnaire_screen", "Typical Angina"))
        self.angina_input.setItemText(1, _translate("questionnaire_screen", "Atypical Angina"))
        self.angina_input.setItemText(2, _translate("questionnaire_screen", "Non-anginal"))
        self.angina_input.setItemText(3, _translate("questionnaire_screen", "Asymptomatic"))
        self.label_5.setText(_translate("questionnaire_screen", "What is your resting blood pressure? Use the systolic value (if reading a blood pressure gauge, use the top number)."))
        self.resting_bp_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>Blood pressure is measured as Systolic over Diastolic. On a blood pressure gauge, Systolic is the top number, and usually the larger of the two numbers.</p></body></html>"))
        self.label_6.setText(_translate("questionnaire_screen", "What is your total cholesterol level?"))
        self.cholesterol_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>This the total level of your cholesterol, not LDL or HDL levels. This should be available on your latest lipid panel, which is part of routine physicals and bloodwork.</p></body></html>"))
        self.label_7.setText(_translate("questionnaire_screen", "Is your fasting blood sugar levels greater than 120mg/dl?"))
        self.blood_sugar_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>This is part of routine blood work, so check your latest blood work results. You can also test this for yourself by using a handheld Glucometer, which you can find at most pharmacies.</p></body></html>"))
        self.blood_sugar_input.setItemText(0, _translate("questionnaire_screen", "Yes"))
        self.blood_sugar_input.setItemText(1, _translate("questionnaire_screen", "No"))
        self.label_8.setText(_translate("questionnaire_screen", "What were the results of your latest electrocardiogram (ECG)?"))
        self.ecg_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>Your ECG results should tell you which of the options are most appropriate. If your ECG shows any abnormalities that are NOT signs of left ventricular hypertrophy, then select the second option. If it does, then select the third option. If your ECG was normal, choose the first option.</p></body></html>"))
        self.ecg_input.setItemText(0, _translate("questionnaire_screen", "Normal"))
        self.ecg_input.setItemText(1, _translate("questionnaire_screen", "ST-T Wave Abnormal"))
        self.ecg_input.setItemText(2, _translate("questionnaire_screen", "Left Ventricular hypertrophy"))
        self.label_9.setText(_translate("questionnaire_screen", "What is your maximum observed heart rate?"))
        self.heart_rate_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>This is the maximum heart rate observed by your provider during a exhaustion test. If you don\'t know, subtract your age from 220. </p></body></html>"))
        self.label_10.setText(_translate("questionnaire_screen", "Do you experience chest pain or discomfort during exercise?"))
        self.exercise_angina_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>If you experience any sort of chest pain, tightness, or discomfort that becomes onset after exercise and is relieved soon after exercise, then choose &quot;Yes&quot;. If not, choose &quot;No&quot;. You can test this for yourself by doing some quick exercise, such as push-ups or going for a short run.</p></body></html>"))
        self.exercise_angina_input.setItemText(0, _translate("questionnaire_screen", "Yes"))
        self.exercise_angina_input.setItemText(1, _translate("questionnaire_screen", "No"))
        self.label_11.setText(_translate("questionnaire_screen", "What was your Oldpeak during your last round of exercise?"))
        self.oldpeak_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>Oldpeak refers to a depression in your heart that occurs in exercise relative to your heart in rest. This is a relatively rare item for doctors to test for, so if you don\'t know, you can leave this number as 0 or check the box to the right.</p></body></html>"))
        self.label_12.setText(_translate("questionnaire_screen", "What was the slope of the peak in your last round of exercise?"))
        self.slope_input.setToolTip(_translate("questionnaire_screen", "<html><head/><body><p>Like Oldpeak, this is rarely collected. It is normally part of a complex test that doctors normally don\'t perform unless there is ample reason to do so. If you don\'t know, then simply choose the &quot;Flat&quot; option.</p></body></html>"))
        self.slope_input.setItemText(0, _translate("questionnaire_screen", "Upsloping"))
        self.slope_input.setItemText(1, _translate("questionnaire_screen", "Flat"))
        self.slope_input.setItemText(2, _translate("questionnaire_screen", "Downsloping"))
        self.label.setText(_translate("questionnaire_screen", "<html><head/><body><p>Scroll to see all questions. ALL questions must be answered before submission. If you don\'t know the answer to a particular question, check the box to the right of each question. If you have difficulty understanding the answer choices for a question, you can hover over the box of inputs for an explanation of what they mean or what to enter. When you\'re finished, click the &quot;Submit&quot; button at the bottom.</p></body></html>"))
        self.questionnaire_submit_button.setText(_translate("questionnaire_screen", "Submit"))

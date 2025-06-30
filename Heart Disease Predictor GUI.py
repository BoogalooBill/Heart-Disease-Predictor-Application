import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import joblib
import time
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QDialog
from PyQt6.QtCore import QObject, QEvent, QThread, pyqtSignal
from gui_files.start_screen_ui import Ui_start_window
from gui_files.disclaimer_screen_ui import Ui_disclaimer_window
from gui_files.questionnaire_ui import Ui_questionnaire_screen
from gui_files.calculating_ui import Ui_calculation_dialog
from gui_files.finished_calculation_dialog import Ui_finished_calculation_dialog
from gui_files.results_screen_ui import Ui_results_window

#worker thread to handle the calculation dialogs without freezing main window
class PredictionWorkerThread(QThread):
    prediction_complete = pyqtSignal(float)

    def __init__(self, model, input_data):
        super().__init__()
        self.model = model
        self.input_data = input_data
    
    def run(self):
        time.sleep(2) #self-incur time penalty to give user feeling that computer is thinking
        prediction = self.model.predict_proba(self.input_data)[0][1] * 100 #convert the prediction into a percentage
        self.prediction_complete.emit(prediction) #signal that function is complete back to the controller

#main window controller
class ScreenController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.historical_data_dict = {}
        self.historical_data = pd.read_csv("Data\\cleaned_data.csv") #load in dataset for graph creation
        self.prediction_model = joblib.load("Data\\heart_disease_predictor_model_rf.pkl") #load in model for predictions
        self.worker = None #initialize space for worker thread
        self.prediction_result = None #initialize space for the prediction result

        #translations of string values to integer values for the model
        self.value_translations = {
            "sex": {"Male": 1, "Female": 0},
            "chest pain type": {"Typical Angina": 1, "Atypical Angina": 2, "Non-anginal": 3, "Asymptomatic": 4},
            "fasting blood sugar": {"Yes": 1, "No": 0},
            "resting ecg": {"Normal": 0, "ST-T Wave Abnormal": 1, "Left Ventricular hypertrophy": 2},
            "exercise angina": {"Yes": 1, "No": 0},
            "ST slope": {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
        }

        #populate data with averages initially, will be overwritten with patient entered data (if it exists)
        self.calculate_average()

        #create data frames for later use in graphs
        for col in self.historical_data.columns:
            if col != "age" and col != "target":
                self.historical_data_dict[col] = self.historical_data[["age", col, "target"]]

        #start the main screen 
        self.load_screen(Ui_start_window)

    #function to handle generation of averages used in case the patient doesn't know the answer for a particular question
    def calculate_average(self):
        for col in list(self.historical_data.columns.values):
            if col == "age" or col == "sex":
                continue
            else:
                self.data[col] = int(math.floor(self.historical_data[col].mean())) #return the floor to account for binary values and integer values

    #function to load the given UI into the main window
    def load_screen(self, ui_class):

        #remove the old button from view
        old_button = self.findChild(QPushButton, "nextButton")
        if old_button:
            old_button.setParent(None)
            old_button.hide()


        self.setCentralWidget(None)
        self.ui = ui_class()  #creates an instance of the UI class
        self.ui.setupUi(self)

        #find and connect the next button dynamically
        next_button = self.findChild(QPushButton, "nextButton")

        if next_button:
            next_button.setParent(self)
            next_button.show()
            next_button.setEnabled(True)
            next_button.raise_()

            next_button.clicked.connect(self.handle_next)
        else:
            pass
        
        self.show()

    #function to handle loading of the results screen
    def load_results(self):
        #remove the old button from view
        old_button = self.findChild(QPushButton, "nextButton")
        if old_button:
            old_button.setParent(None)
            old_button.hide()


        self.setCentralWidget(None)
        self.ui = Ui_results_window()  #creates an instance of the UI class
        self.ui.setupUi(self)

        #run function to update labels to match user results
        self.ui.update_labels(self.prediction_result, self.data)

        self.ui.bp_graph_button.clicked.connect(lambda: self.load_graph("resting bp s", "BLOOD PRESSURE"))
        self.ui.cholesterol_graph_button.clicked.connect(lambda: self.load_graph("cholesterol", "CHOLESTEROL"))
        self.ui.blood_sugar_graph_button.clicked.connect(lambda: self.load_graph("fasting blood sugar", "BLOOD SUGAR"))
        #show the screen
        self.show()

        #function to load graph screens
    def load_graph(self, col, y_label):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        #separation of target and non-target data for plotting
        df = self.historical_data_dict[col]
        cvd_data = df[df["target"] == 1]  # Patients with CVD
        non_cvd_data = df[df["target"] == 0]  # Patients without CVD

        #plot the non-cvd data in green
        sns.regplot(x=non_cvd_data["age"], y=non_cvd_data[col], 
                    scatter_kws={"alpha": 0.6, "color": "green", "label": "No CVD"}, 
                    line_kws={"color": "darkgreen"}, ax=ax[0])

        #plot the cvd data in red
        sns.regplot(x=cvd_data["age"], y=cvd_data[col], 
                    scatter_kws={"alpha": 0.6, "color": "red", "label": "CVD"}, 
                    line_kws={"color": "darkred"}, ax=ax[0])

        #get the user's values for the graph
        user_age = self.data.get("age", None)
        user_value = self.data.get(col, None)

        #check if the values are available first
        if user_age is not None and user_value is not None:
            #highlight the user's data in blue
            ax[0].scatter(user_age, user_value, color="blue", s=150, marker="*", label="Your Data")

            #add an annotation highlighting the user's value
            ax[0].annotate(f"Your Value: {user_value}", 
                        xy=(user_age, user_value), 
                        xytext=(user_age + 2, user_value + 2),  # Offset text slightly
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), 
                        fontsize=10, color='blue')

        #labels and ax for the various graphs
        #scatterplot
        ax[0].set_xlabel("AGE")
        ax[0].set_ylabel(y_label)
        ax[0].set_title(f"{y_label} VS AGE")
        ax[0].legend()

        #histogram
        ax[1].hist(non_cvd_data[col], bins=20, color="green", alpha=0.6, label="No CVD")
        ax[1].hist(cvd_data[col], bins=20, color="red", alpha=0.6, label="CVD")
        ax[1].axvline(user_value, color="blue", linestyle="dashed", linewidth=2, label="Your Data")  # Line for user's data
        ax[1].set_title(f"Distribution of {y_label}")
        ax[1].set_xlabel(y_label)
        ax[1].set_ylabel("Count")
        ax[1].legend()

        #line plot
        avg_values_by_age = df.groupby("age")[col].mean()  # Get mean of col by age
        ax[2].plot(avg_values_by_age.index, avg_values_by_age.values, marker="o", linestyle="-", color="purple", label="Avg Trend")
        ax[2].axvline(user_age, color="blue", linestyle="dashed", linewidth=2, label="Your Age")
        ax[2].set_title(f"Average {y_label} by Age")
        ax[2].set_xlabel("Age")
        ax[2].set_ylabel(y_label)
        ax[2].legend()

        #present the plot such that text does not overlap
        plt.tight_layout()

        #show the plot
        plt.show()

    #function to save results from the questionnaire screen for use on the results screen and for making predictions
    def save_questionnaire_data(self):
        if isinstance(self.ui, Ui_questionnaire_screen):
            self.data["age"] = self.ui.age_input.value()
            self.data["sex"] = self.value_translations["sex"][self.ui.sex_input.currentText()]

            #checks to use the average value if the checkbox for "I don't know" is selected
            if not self.ui.angina_checkbox.isChecked():
                self.data["chest pain type"] = self.value_translations["chest pain type"][self.ui.angina_input.currentText()]
            
            if not self.ui.resting_bp_checkbox.isChecked():
                self.data["resting bp s"] = self.ui.resting_bp_input.value()

            if not self.ui.cholesterol_checkbox.isChecked():
                self.data["cholesterol"] = self.ui.cholesterol_input.value()

            if not self.ui.blood_sugar_checkbox.isChecked():
                self.data["fasting blood sugar"] = self.value_translations["fasting blood sugar"][self.ui.blood_sugar_input.currentText()]

            if not self.ui.ecg_checkbox.isChecked():
                self.data["resting ecg"] = self.value_translations["resting ecg"][self.ui.ecg_input.currentText()]

            if not self.ui.heart_rate_checkbox.isChecked():
                self.data["max heart rate"] = self.ui.heart_rate_input.value()

            if not self.ui.exercise_angina_checkbox.isChecked():
                self.data["exercise angina"] = self.value_translations["exercise angina"][self.ui.exercise_angina_input.currentText()]

            if not self.ui.oldpeak_checkbox.isChecked():
                self.data["oldpeak"] = float(self.ui.oldpeak_input.value())

            if not self.ui.slope_checkbox.isChecked():
                self.data["ST slope"] = self.value_translations["ST slope"][self.ui.slope_input.currentText()]

    #function to show the initial calculating results dialog
    def show_calculation_dialog(self):
        self.calculation_dialog = QDialog(self)
        ui = Ui_calculation_dialog()
        ui.setupUi(self.calculation_dialog)
        
        #list the feature columns for use in creating the dataframe for the prediction
        feature_columns = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"]

        #put the features in a specific order to feed to the machine learning model
        features = {
            "age": [self.data["age"]],
            "sex": [self.data["sex"]],
            "chest pain type": [self.data["chest pain type"]],
            "resting bp s": [self.data["resting bp s"]],
            "cholesterol": [self.data["cholesterol"]],
            "fasting blood sugar": [self.data["fasting blood sugar"]],
            "resting ecg": [self.data["resting ecg"]],
            "max heart rate": [self.data["max heart rate"]],
            "exercise angina": [self.data["exercise angina"]],
            "oldpeak": [self.data["oldpeak"]],
            "ST slope": [self.data["ST slope"]]
        }

        input_data = pd.DataFrame(features, columns=feature_columns)

        self.worker = PredictionWorkerThread(self.prediction_model, input_data)
        self.worker.prediction_complete.connect(self.handle_prediction_result)
        self.worker.start()

        self.calculation_dialog.exec()

    #function to show the finished calculation dialog
    def show_finished_calculation_dialog(self):
        self.finished_calculation_dialog = QDialog(self)
        ui = Ui_finished_calculation_dialog()
        ui.setupUi(self.finished_calculation_dialog)

        ui.pushButton.clicked.connect(self.handle_finished_calculation)

        self.finished_calculation_dialog.exec()

    #callback function to handle the retrieval of the prediction result calculation 
    def handle_prediction_result(self, result):
        self.calculation_dialog.accept()
        self.prediction_result = result

        if self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()

    #callback function to handle the closing of the finished calculation dialog
    def handle_finished_calculation(self):
        self.finished_calculation_dialog.accept()
        old_button = self.findChild(QPushButton, "nextButton")
        if old_button:
            old_button.setParent(None)
            old_button.hide()

    #function to handle when to switch to the next screen
    def handle_next(self):
        if isinstance(self.ui, Ui_questionnaire_screen):
            self.save_questionnaire_data()
            self.show_calculation_dialog()
            self.show_finished_calculation_dialog()

        if isinstance(self.ui, Ui_start_window):
            self.load_screen(Ui_disclaimer_window)
        elif isinstance(self.ui, Ui_disclaimer_window):
            self.load_screen(Ui_questionnaire_screen)
        elif isinstance(self.ui, Ui_questionnaire_screen):
            self.load_results()
        else:
            pass

#main function to call the ScreenController and begin application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = ScreenController()
    sys.exit(app.exec())
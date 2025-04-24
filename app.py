from datetime import datetime
import os
import numpy as np
import scipy.io
import scipy.signal
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import wfdb  
from collections import Counter  
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json

# --- Initialize Flask App ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

analysis_results = {}  # Global dictionary to store latest analysis results

# --- Import analysis functions from separate modules ---
from singleSelection import analyze_single, select_single_final_disease as single_select_final_disease
from doubleSelection import analyze_double , select_double_final_disease as double_select_final_disease
from multiSelection import analyze_multi, detect_leads, select_multi_final_disease as multi_select_final_disease

# --- A common plotting function for ECG segments ---
def plot_ecg_segments(ecg_signal, fs, r_peaks, seg_indices, labels, normal_label="Normal"):
    time_axis = np.arange(len(ecg_signal)) / fs
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, ecg_signal, color="blue", linewidth=1, label="ECG Signal")
    plt.scatter(r_peaks / fs, ecg_signal[r_peaks], color="red", marker="x", label="R-Peaks")
    for (start, end), disease in zip(seg_indices, labels):
        color = "blue" if disease == normal_label else "red"
        segment_time = time_axis[start:end]
        plt.plot(segment_time, ecg_signal[start:end], color=color, linewidth=2.5,
                 label=disease if disease not in plt.gca().get_legend_handles_labels()[1] else "")
        mid_time = (start + end) / (2 * fs)
        plt.text(mid_time, ecg_signal[start], disease, fontsize=9, ha="center",
                 bbox=dict(facecolor="yellow", alpha=0.6, edgecolor="black"))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude")
    plt.title("ECG Signal with Highlighted Abnormal Beats")
    plt.legend()
    plot_path = "Flaskapp/static/ecg_plot.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# --- Manual Leads Analysis Functions ---
# Global variables to store data for lead analysis
lead_names = None
normal_label = None
disease_mapping = None
analysis_results = None

def analyze_manual_leads(record_path, selected_manual, app_lead_names, app_normal_label, app_disease_mapping, app_analysis_results):
    """
    Analyze manual lead combination and generate plots.
    """
    global lead_names, normal_label, disease_mapping, analysis_results
    lead_names = app_lead_names
    normal_label = app_normal_label
    disease_mapping = app_disease_mapping
    analysis_results = app_analysis_results
    print("\n--- Analysis for Manual Lead Combination ---")
    try:
        # Validate selected_manual
        if not selected_manual or len(selected_manual) == 0:
            raise ValueError("No leads selected for manual analysis.")

        # Convert lead indices to integers if they're strings
        selected_manual = [int(lead) if isinstance(lead, str) else lead for lead in selected_manual]
        
        # Ensure lead_names has entries for all selected leads
        for lead in selected_manual:
            if str(lead) not in lead_names and lead not in lead_names:
                lead_names[lead] = f"Lead {lead}"

        # Use the multi_detect_leads function directly
        lead_predictions_manual, fs_manual = detect_leads(record_path, selected_manual, window_size_wavelet=250)
        num_leads_manual = len(selected_manual)

        if num_leads_manual == 0:
            raise ValueError("No valid leads found for manual analysis.")

        final_disease_manual = {}
        final_disease_manual1, confidence1 = multi_select_final_disease(lead_predictions_manual, selected_manual)
        print(f"Final Disease (Multi Label): {final_disease_manual1}, Confidence: {confidence1}")
        final_disease_manual2, confidence2 = double_select_final_disease(lead_predictions_manual, selected_manual)
        print(f"Final Disease (Double Label): {final_disease_manual2}, Confidence: {confidence2}")
        final_disease_manual3, confidence3 = single_select_final_disease(lead_predictions_manual, selected_manual)
        print(f"Final Disease (Single Label): {final_disease_manual3}, Confidence: {confidence3}")
        # Select the final disease based on confidence
        if confidence1 > confidence2 and confidence1 >= confidence3:
            final_disease_manual = final_disease_manual1
            confidence = confidence1
            print("MULTI LABEL COMBINATION")
            label_file_path = "models\labelMulti.json"
        elif confidence2 >= confidence1 and confidence2 >= confidence3:
            final_disease_manual = final_disease_manual2
            confidence = confidence2
            print("DOUBLE LABEL COMBINATION")
            label_file_path = "models\labelDouble.json"
        else:
            final_disease_manual = final_disease_manual3
            confidence = confidence3
            print("SINGLE LABEL COMBINATION")
            label_file_path = "models\labelSingle.json"

        fig_manual, axes_manual = plt.subplots(num_leads_manual, 1, figsize=(15, 3 * num_leads_manual), sharex=True)
        if num_leads_manual == 1:
            axes_manual = [axes_manual]

        fig_manual.suptitle("ECG Signals with Highlighted Abnormal Beats", fontsize=18)

        # Iterate over each lead and plot its data
        for i, lead in enumerate(selected_manual):
            ax = axes_manual[i]
            data = lead_predictions_manual.get(lead)
            if data is None:
                # Use string representation of lead as key if integer key doesn't work
                lead_key = lead
                if lead not in lead_names:
                    lead_key = str(lead)
                
                lead_name = lead_names.get(lead_key, f"Lead {lead}")
                ax.set_title(f"{lead_name}: No data")
                continue
                
            # Use string representation of lead as key if integer key doesn't work
            lead_key = lead
            if lead not in lead_names:
                lead_key = str(lead)
                
            lead_name = lead_names.get(lead_key, f"Lead {lead}")

            # Plot the ECG signal for the current lead
            time_axis = np.arange(len(data["signal"])) / fs_manual
            ax.plot(time_axis, data["signal"], color="blue", linewidth=1, label="ECG Signal")

            # Plot the segments and add labels for the current lead
            for (start, end), label_index in zip(data["seg_indices"], data["pred_labels"]):
                # Load the label mapping from the JSON file
                # label_file_path = "models\labelMulti.json"
                with open(label_file_path, "r") as label_file:
                    label_mapping = json.load(label_file)  # Parse the JSON file into a Python list

                # Map the label index to the corresponding label
                try:
                    label = label_mapping[int(label_index)]  # Use the label index to get the label
                except (ValueError, IndexError, TypeError):
                    label = "Unknown"  # Default to "Unknown" if mapping fails

                # Set the color for the segment
                color = "blue" if label == normal_label else "red"

                # Plot the segment
                segment_time = np.arange(start, end) / fs_manual
                ax.plot(segment_time, data["signal"][start:end], color=color, linewidth=2.5)

                # Add the label as text in the middle of the segment
                mid_time = (start + end) / (2 * fs_manual)
                ax.text(mid_time, data["signal"][start], label, fontsize=8, ha="center",
                        bbox=dict(facecolor="yellow", alpha=0.6, edgecolor="black"))

            # Add R-peaks and other plot details
            ax.scatter(data["r_peaks"] / fs_manual, data["signal"][data["r_peaks"]],
                       color="black", marker="x", label="R-Peaks")
            ax.set_ylabel(lead_name)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True)

        # Add x-axis label to the last subplot
        axes_manual[-1].set_xlabel("Time (seconds)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the manual leads plot
        plot_path = "static\ecg_leads.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

        if final_disease_manual is not None:
            print("\nFinal Diagnosis based on Manual Lead Combination:")
            print(f"  Disease: {final_disease_manual}")
            return {
                "final_disease": final_disease_manual,
                "confidence": confidence,
                "lead_predictions": lead_predictions_manual,
                "fs": fs_manual,
                "plot_path": plot_path
            }
        else:
            print("No valid segments were detected for the Manual Lead Combination.")
            return None
    except ValueError as ve:
        print(f"Validation Error: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# --- PDF Report Generation ---
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# Define the logo path
LOGO_PATH = "static\SET.png"  # Replace with actual logo path

class PDFReport:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pdf = canvas.Canvas(file_path, pagesize=letter)
        self.width, self.height = letter
        self.y_position = self.height - 100  # Initial Y position for content
        self.page_number = 1

    def add_header(self):
        """Adds the hospital logo, name, and contact details to the report."""
        if os.path.exists(LOGO_PATH):
            self.pdf.drawImage(LOGO_PATH, 20, self.height - 70, width=60, height=50)

        self.pdf.setFont("Helvetica-Bold", 14)
        self.pdf.drawString(100, self.height - 50, "THE HITV.G HOSPITAL")
        self.pdf.setFont("Helvetica-Bold", 12)
        self.pdf.drawString(100, self.height - 65, "Accurate | Caring | Instant")

        self.pdf.setFont("Helvetica", 10)
        self.pdf.drawString(400, self.height - 50, "Phone: 040-XXXXXXXXX / +91 XX XXX XXX")
        self.pdf.drawString(400, self.height - 65, "Email: sgbhospital@gmail.com")

        self.pdf.setFont("Helvetica-Bold", 10)
        self.pdf.drawString(120, self.height - 90, ".")

        # Blue & Red separator lines
        self.pdf.setFillColorRGB(0, 0.47, 0.75)  # Blue
        self.pdf.rect(20, self.height - 100, self.width - 40, 3, fill=True, stroke=False)

        self.pdf.setFillColorRGB(0.89, 0.12, 0.14)  # Red
        self.pdf.rect(20, self.height - 103, self.width - 40, 3, fill=True, stroke=False)

        # Reset text color to black
        self.pdf.setFillColorRGB(0, 0, 0)

        # Date & Time
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        self.pdf.setFont("Helvetica", 10)
        self.pdf.drawString(self.width - 150, self.height - 115, f"Date: {current_date}")
        self.pdf.drawString(self.width - 150, self.height - 130, f"Time: {current_time}")

    def add_footer(self):
        """Adds the footer with page number and contact details."""
        self.pdf.setFillColorRGB(0.39, 0.58, 0.93)  # Medium blue
        self.pdf.rect(0, 10, self.width, 20, fill=True, stroke=False)

        self.pdf.setFont("Helvetica", 8)
        self.pdf.setFillColorRGB(1, 1, 1)  # White text
        self.pdf.drawString(50, 20, f" Thank you || THE G HOSPITAL || EMERGENCY CONTACT - +91 XXXXXXXXXX")

        self.page_number += 1

    def add_space(self, space=20):
        """Adds vertical spacing between sections and creates a new page when necessary."""
        if self.y_position - space < 50:  # Check if there's space left
            self.pdf.showPage()
            self.add_header()
            self.y_position = self.height - 100  # Reset position
        self.y_position -= space

    def add_text(self, text, font="Helvetica", size=12, bold=False, space=20):
        """Adds text to the PDF with automatic new page handling."""
        if bold:
            self.pdf.setFont(f"{font}-Bold", size)
        else:
            self.pdf.setFont(font, size)

        self.pdf.drawString(50, self.y_position, text)
        self.add_space(space)

    def add_list(self, items, font="Helvetica", size=10):
        """Adds a bullet-point list with automatic page breaks."""
        self.pdf.setFont(font, size)
        for item in items:
            self.pdf.drawString(50, self.y_position, f"• {item}")
            self.add_space(25)

    def add_ecg_image(self, image_path,image_width = 400,image_height = 400, x_position = 100):
        """Adds the ECG image to the PDF with a fixed size and position."""
        if os.path.exists(image_path):
            # Define fixed dimensions for the image
            # image_width = 400  # Fixed width
            # image_height = 400  # Fixed height
            # x_position = 100  # Centered horizontally
            y_position = self.y_position - image_height - 20  # Adjust position below current content

            # Draw the image on the PDF
            self.pdf.drawImage(image_path, x_position, y_position, width=image_width, height=image_height)

            # Update the y_position after adding the image
            self.y_position = y_position - 20

    def save_pdf(self):
        """Finalizes and saves the PDF."""
        self.add_footer()
        self.pdf.save()


@app.route("/generate_report", methods=["POST"])
def generate_report():
    try:
        if not analysis_results or not analysis_results.get("final_disease"):
            return jsonify({"success": False, "error": "No analysis results found. Please analyze an ECG first."})

        # Proceed with report generation...
        report = PDFReport("static\ecg_report.pdf")
        report.add_header()
        # **REPORT TITLE**
        report.add_space(30)
        report.add_text("Cardiac Functional ECG Report", size=16, bold=True, space=40)

        # **PATIENT & DOCTOR DETAILS**
        report.add_text("Patient Information:", bold=True)
        report.add_text(f"Name: {analysis_results['patient_name']}")
        report.add_text(f"Age: {analysis_results['patient_age']}")
        report.add_text(f"Gender: {analysis_results['patient_gender']}")
        report.add_text("Doctor Information:", bold=True)
        report.add_text(f"Name: {analysis_results['doctor_name']}")
        report.add_text(f"Designation: {analysis_results['doctor_designation']}")

        # **ECG ANALYSIS RESULTS**
        report.add_text("ECG Analysis Results:", bold=True)
        report.add_ecg_image(analysis_results["plot_path"],image_width=550,x_position=30)
        report.add_space(40)

        # **PREDICTED DISEASE**
        final_disease_list = analysis_results.get("final_disease", [])
        if not isinstance(final_disease_list, list):
            final_disease_list = list(final_disease_list)  # Ensure it's a list

        report.add_text("Predicted Conditions:", bold=True)
        for disease in final_disease_list:
            report.add_text(f"- {disease}", bold = True , size=11)

        # **DISEASE DESCRIPTION**
        for disease in final_disease_list:
            disease_details = get_disease_info(disease)
            report.add_text(f"Causes of {disease}:", bold=True, size = 10)
            report.add_list(disease_details["Causes"])

            report.add_text(f"Symptoms of {disease}:", bold=True , size = 10)
            report.add_list(disease_details["Symptoms"])

        # **RECOMMENDED TESTS**
        report.add_text("Recommended Tests:", bold=True)
        for disease in final_disease_list:
            tests = get_recommended_tests(disease)
            disease_title = f"{disease}:"
            report.add_text(disease_title, bold=True , size=10)
            report.add_list(tests)

        # **PRECAUTIONS**
        report.add_text("Precautions for the Patient:", bold=True)
        for disease in final_disease_list:
            precautions = get_precautions([disease])  # Get precautions for the specific disease
            if precautions:
                disease_title = f"For {disease}:"
                report.add_text(disease_title, bold=True, size=10)
                report.add_list(precautions)

        # **DISEASE IMAGES**
        image_found = False
        for disease in final_disease_list:
            image_path = f"static/images/{disease}.jpg"
            if os.path.exists(image_path):
                if not image_found:
                    report.add_text("Disease Images:", bold=True, space=30)
                    image_found = True
                report.add_text(f"Anatomical Changes for {disease}:", bold=True, size=10)
                report.add_ecg_image(image_path, image_height=240, image_width=280)
                report.add_space(20)

        report.add_space(150)    
        # **DISCLAIMER**
        report.add_text("Disclaimer:", bold=True, space=30)
        report.add_text("The detected regions in the ECG are indicative of possible cardiac diseases.", size=10)
        report.add_text("Clinical correlation with the patient's medical history and further diagnostic tests is necessary for accurate evaluation.", size=10)

        report.add_space(50)
        report.add_text("Doctor's Comments:", bold=True, space=30)
        report.add_text("_______________________________", bold=True)
        report.add_space(20)
        report.add_text("_________________________", bold=True)
        report.add_text("Dr. Srinivasan", bold=True, space=30)
        report.add_text("Cardiologist", bold=True)
        report.add_text("THE .G HOSPITAL", bold=True)
        report.add_text(".ASRV LLP, Avishkaran, NIPER, Balanagar,", bold=True)
        report.add_text("Hyderabad, Telangana, 500037,", bold=True)
        report.add_text("Date: ", bold=True)
        report.save_pdf()

        return jsonify({
            "success": True,
            "report_url": url_for('static', filename='ecg_report.pdf'),
            "patient_name": analysis_results["patient_name"],
            "disease": final_disease_list
        })

    except Exception as e:
        print(f"Error in generate_report: {e}")
        return jsonify({"success": False, "error": str(e)})



# Helper functions for test recommendations and precautions based on detected condition
def get_recommended_tests(disease):
    recommendations = {
        "1 degree atrioventricular block": [
            "24-hour Holter monitoring to assess heart rhythm abnormalities.",
            "Electrocardiogram (ECG) for periodic evaluation.",
            "Consult a cardiologist to determine if further monitoring is needed."
        ],
        "2 degree atrioventricular block": [
            "Echocardiogram to check heart function and structure.",
            "Electrophysiology study to assess electrical conduction pathways.",
            "24-hour Holter monitoring for continuous heart rhythm observation."
        ],
        "2 degree atrioventricular block (Type one)": [
            "Regular ECG evaluations to track changes in heart conduction.",
            "Stress test to observe heart behavior under physical exertion.",
            "Consultation with a cardiologist for medication or pacemaker evaluation."
        ],
        "2 degree atrioventricular block (Type two)": [
            "Immediate electrophysiology study to assess conduction block severity.",
            "Echocardiogram to examine heart function and detect any underlying conditions.",
            "Pacemaker implantation if heart block is severe."
        ],
        "3 degree atrioventricular block": [
            "Emergency medical assessment with ECG to confirm diagnosis.",
            "Pacemaker implantation for proper heart rhythm management.",
            "Blood tests to detect electrolyte imbalances or underlying causes."
        ],
        "Atrial Bigeminy": [
            "24-hour Holter monitoring to detect frequent abnormal beats.",
            "Echocardiogram to assess heart function.",
            "Electrolyte level testing to rule out imbalances."
        ],
        "Axis Left Shift": [
            "Echocardiogram to assess heart size and structure.",
            "Monitor for underlying conditions such as hypertension or heart disease.",
            "Follow-up with a cardiologist if symptoms persist."
        ],
        "Atrial Premature Beats": [
            "24-hour Holter monitoring to track frequency of premature beats.",
            "Electrolyte level testing to identify potential imbalances.",
            "Lifestyle modification recommendations to reduce triggers."
        ],
        "Abnormal Q Wave": [
            "Electrocardiogram (ECG) to monitor heart activity.",
            "Cardiac MRI for detailed imaging of heart muscle damage.",
            "Coronary angiography to check for blocked arteries."
        ],
        "Axis Right Shift": [
            "Echocardiogram to detect heart or lung-related abnormalities.",
            "Chest X-ray to check for lung diseases affecting heart positioning.",
            "Consultation with a cardiologist for further testing."
        ],
        "Atrioventricular Block": [
            "ECG to assess the severity of the conduction block.",
            "Holter monitoring to evaluate long-term heart rhythm.",
            "Electrophysiology study for in-depth analysis of electrical activity."
        ],
        "Counterclockwise Rotation": [
            "Echocardiogram to assess heart function.",
            "Regular ECG monitoring to track heart rotation changes.",
            "Cardiac MRI if structural abnormalities are suspected."
        ],
        "Clockwise Rotation": [
            "ECG to determine any associated heart abnormalities.",
            "Monitor for symptoms such as dizziness or breathlessness.",
            "Echocardiogram to assess heart function and structure."
        ],
        "Early Repolarization of the Ventricles": [
            "ECG to monitor for any progression of early repolarization.",
            "Echocardiogram to rule out underlying heart disease.",
            "Cardiac MRI if further evaluation is needed."
        ],
        "fQRS Wave": [
            "Echocardiogram to evaluate cardiac function.",
            "Electrophysiology study to assess conduction disturbances.",
            "Regular ECG monitoring for long-term assessment."
        ],
        "Interior Differences Conduction": [
            "Electrocardiogram (ECG) for diagnosis and monitoring.",
            "Electrophysiology study to assess severity.",
            "Follow-up with a cardiologist for management strategies."
        ],
        "Intraventricular Block": [
            "Echocardiogram to evaluate the underlying cause.",
            "24-hour Holter monitoring to assess rhythm abnormalities.",
            "Electrophysiology study if conduction issues persist."
        ],
        "Junctional Escape Beat": [
            "ECG monitoring to track rhythm irregularities.",
            "Electrolyte testing to detect imbalances.",
            "Cardiac MRI if structural heart disease is suspected."
        ],
        "Junctional Premature Beat": [
            "24-hour Holter monitoring to evaluate premature beats.",
            "Electrophysiology study for detailed rhythm analysis.",
            "Consultation with a cardiologist for treatment recommendations."
        ],
        "Left Bundle Branch Block": [
            "Echocardiogram to assess left ventricular function.",
            "Cardiac MRI or CT scan to evaluate structural abnormalities.",
            "Stress test to determine if blood flow is compromised."
        ],
        "Left Back Bundle Branch Block": [
            "ECG to track conduction abnormalities.",
            "Echocardiogram to assess left ventricular function.",
            "Electrophysiology study for further evaluation."
        ],
        "Left Front Bundle Branch Block": [
            "Regular ECG monitoring to observe progression.",
            "Echocardiogram to detect any associated heart conditions.",
            "Cardiac MRI if structural issues are suspected."
        ],
        "Left Ventricle Hypertrophy": [
            "Echocardiogram to measure heart muscle thickness.",
            "Blood pressure monitoring to prevent further strain on the heart.",
            "Cardiac MRI for detailed heart imaging."
        ],
        "Right Bundle Branch Block": [
            "Echocardiogram to assess right heart function.",
            "Pulmonary function tests if lung disease is suspected.",
            "Regular ECG evaluations to monitor progression."
        ],
        "Sinus Bradycardia": [
            "Electrocardiogram (ECG) to assess heart rhythm.",
            "Holter monitoring to evaluate daily fluctuations.",
            "Consultation with a cardiologist if symptoms persist."
        ],
        "Sinus Tachycardia": [
            "ECG to assess heart rate patterns.",
            "Thyroid function tests to rule out hyperthyroidism.",
            "Electrolyte level testing for imbalances."
        ],
        "Atrial Fibrillation": [
            "24-hour Holter monitoring to track irregular rhythms.",
            "Echocardiogram to assess structural heart abnormalities.",
            "Blood tests for thyroid function and electrolyte levels."
        ],
        "Atrial Flutter": [
            "Echocardiogram to assess heart function.",
            "Chest X-ray to check for lung or heart abnormalities.",
            "Electrophysiology study to determine treatment options."
        ],
        "Ventricular Tachycardia": [
            "Cardiac MRI to evaluate structural heart disease.",
            "Stress test to assess exercise-induced arrhythmias.",
            "Electrophysiology study to evaluate severity."
        ],
        "Ventricular Fibrillation": [
            "Echocardiogram to assess heart function.",
            "Genetic testing for inherited heart conditions.",
            "Electrophysiology study to evaluate electrical conduction."
        ],
        "Wandering in the Atrioventricular Node": [
            "24-hour Holter monitoring to track heart rate changes.",
            "Electrolyte level testing to rule out imbalances.",
            "Follow-up with a cardiologist for monitoring."
        ],
        "WPW Syndrome": [
            "Electrophysiology study to assess the need for catheter ablation.",
            "Echocardiogram to evaluate structural abnormalities.",
            "ECG monitoring to detect recurrent episodes."
        ]
    }

    tests = recommendations.get(disease, ["Comprehensive cardiac evaluation", "24-hour Holter monitoring", "Echocardiogram"])
    return [f"{test}" for test in tests]


def get_precautions(diseases):
    precautions = {
        "1 degree atrioventricular block": [
            "Regular cardiac check-ups are necessary to monitor heart function.",
            "Avoid excessive caffeine and alcohol as they can affect heart rhythm.",
            "Maintain a healthy diet and engage in light physical activity."
        ],
        "2 degree atrioventricular block": [
            "Monitor heart rate regularly and seek medical attention if symptoms worsen.",
            "Avoid dehydration and ensure proper electrolyte balance.",
            "Follow up with a cardiologist to determine if a pacemaker is needed."
        ],
        "2 degree atrioventricular block (Type one)": [
            "Regular ECG monitoring is advised to track heart rhythm changes.",
            "Avoid stimulant medications unless prescribed by a doctor.",
            "Report any episodes of dizziness or fainting to a healthcare provider."
        ],
        "2 degree atrioventricular block (Type two)": [
            "Immediate medical evaluation is necessary as this condition may worsen.",
            "Avoid strenuous activities that may stress the heart.",
            "A pacemaker may be required if the condition progresses."
        ],
        "3 degree atrioventricular block": [
            "Seek immediate medical attention, as this condition can be life-threatening.",
            "Pacemaker implantation is often required to regulate heart rhythm.",
            "Avoid activities that may lead to dizziness or fainting."
        ],
        "Atrial Bigeminy": [
            "Reduce caffeine and alcohol intake to prevent heart rhythm disturbances.",
            "Manage stress levels through relaxation techniques such as meditation.",
            "Follow up regularly with a cardiologist for heart monitoring."
        ],
        "Axis Left Shift": [
            "Maintain a heart-healthy diet and exercise routine.",
            "Monitor blood pressure regularly and manage hypertension effectively.",
            "Consult a cardiologist if you experience chest pain or breathlessness."
        ],
        "Atrial Premature Beats": [
            "Limit caffeine and alcohol intake as they may trigger irregular beats.",
            "Stay hydrated and maintain electrolyte balance.",
            "Monitor heart rhythm and consult a doctor if irregular beats persist."
        ],
        "Abnormal Q Wave": [
            "Routine ECG check-ups are advised to monitor changes in heart activity.",
            "Maintain a heart-healthy lifestyle to reduce the risk of complications.",
            "Avoid smoking and excessive alcohol consumption."
        ],
        "Axis Right Shift": [
            "Monitor for underlying lung or heart conditions that may contribute to this condition.",
            "Follow up with a cardiologist if experiencing shortness of breath or fatigue.",
            "Maintain a balanced diet and regular exercise."
        ],
        "Atrioventricular Block": [
            "Routine ECG monitoring is necessary to track heart conduction issues.",
            "Avoid excessive alcohol, smoking, and stress.",
            "Follow prescribed medications and attend follow-up appointments."
        ],
        "Counterclockwise Rotation": [
            "Monitor for any symptoms such as dizziness or breathlessness.",
            "Maintain a heart-healthy lifestyle with a balanced diet and exercise.",
            "Consult a doctor if experiencing abnormal heart rhythms."
        ],
        "Clockwise Rotation": [
            "Regular ECG monitoring is required for any progression of heart changes.",
            "Follow a heart-friendly diet and lifestyle.",
            "Consult a cardiologist if symptoms like dizziness occur."
        ],
        "Early Repolarization of the Ventricles": [
            "Regular heart check-ups are advised to monitor any progression.",
            "Avoid excessive alcohol, caffeine, and stimulant drugs.",
            "Report any unusual chest pain or discomfort to a doctor."
        ],
        "fQRS Wave": [
            "Routine ECG evaluations are recommended to monitor for heart disease.",
            "Manage risk factors like hypertension and diabetes effectively.",
            "Consult a cardiologist for further assessment if symptoms develop."
        ],
        "Interior Differences Conduction": [
            "Monitor for arrhythmias and report symptoms like dizziness to a doctor.",
            "Avoid excessive exertion and high-stress activities.",
            "Follow up with a cardiologist for regular check-ups."
        ],
        "Intraventricular Block": [
            "Routine ECG check-ups are advised to monitor the condition.",
            "Avoid excessive alcohol, smoking, and caffeine intake.",
            "Follow prescribed medications and a heart-healthy lifestyle."
        ],
        "Junctional Escape Beat": [
            "Maintain a healthy electrolyte balance and hydration.",
            "Regular monitoring for any underlying heart conditions is necessary.",
            "Seek medical advice if experiencing persistent symptoms."
        ],
        "Junctional Premature Beat": [
            "Reduce caffeine and alcohol intake to minimize premature beats.",
            "Practice stress management techniques like meditation and yoga.",
            "Follow up with a cardiologist for heart function evaluation."
        ],
        "Left Bundle Branch Block": [
            "Routine heart monitoring is necessary to track any disease progression.",
            "Manage high blood pressure and diabetes effectively.",
            "Consult a cardiologist for further assessment if symptoms worsen."
        ],
        "Left Back Bundle Branch Block": [
            "Regular ECG evaluations are necessary to assess heart function.",
            "Avoid excessive exertion and high-stress activities.",
            "Follow prescribed medications and a heart-healthy lifestyle."
        ],
        "Left Front Bundle Branch Block": [
            "Routine ECG monitoring is recommended to detect any progression.",
            "Maintain a heart-healthy lifestyle with proper diet and exercise.",
            "Consult a doctor if experiencing dizziness or shortness of breath."
        ],
        "Left Ventricle Hypertrophy": [
            "Control high blood pressure through medications and lifestyle changes.",
            "Avoid excessive salt and processed foods in the diet.",
            "Follow up with a cardiologist to monitor heart function."
        ],
        "Right Bundle Branch Block": [
            "Routine ECG monitoring is required to detect any worsening condition.",
            "Manage high blood pressure and other cardiovascular risk factors.",
            "Avoid excessive exertion and high-stress activities."
        ],
        "Sinus Bradycardia": [
            "Monitor heart rate regularly and seek medical advice if symptoms appear.",
            "Avoid excessive alcohol, caffeine, and stimulant drugs.",
            "Regular physical activity can help maintain heart health."
        ],
        "Sinus Tachycardia": [
            "Limit caffeine and alcohol consumption to prevent rapid heart rate.",
            "Stay hydrated and maintain electrolyte balance.",
            "Manage stress through relaxation techniques and meditation."
        ],
        "Atrial Fibrillation": [
            "Take prescribed anticoagulants as directed.",
            "Monitor pulse regularly and report irregularities.",
            "Avoid excessive caffeine, alcohol, and smoking."
        ],
        "Atrial Flutter": [
            "Take medications as prescribed to control heart rate.",
            "Avoid stimulants like caffeine and alcohol.",
            "Follow up regularly with a cardiologist."
        ],
        "Ventricular Tachycardia": [
            "Avoid strenuous exercise until cleared by a doctor.",
            "Take all prescribed medications regularly.",
            "Seek immediate medical attention if symptoms worsen."
        ],
        "Ventricular Fibrillation": [
            "Emergency medical intervention is required in case of sudden collapse.",
            "Family members should learn CPR and be prepared for emergencies.",
            "Follow up with an electrophysiologist regularly."
        ],
        "Wandering in the Atrioventricular Node": [
            "Regular heart check-ups are necessary to monitor heart conduction.",
            "Avoid excessive alcohol, smoking, and caffeine intake.",
            "Follow a heart-healthy diet and maintain a proper lifestyle."
        ],
        "WPW Syndrome": [
            "Avoid strenuous activities and monitor heart rate regularly.",
            "Consult a cardiologist for further evaluation.",
            "Consider catheter ablation if symptoms persist."
        ]
    }



    # Default precautions for unknown diseases
    default_precautions = [
        "Take all medications as prescribed by your physician",
        "Follow up with a cardiologist promptly",
        "Monitor and record any symptoms (palpitations, dizziness, chest pain, etc.)",
        "Avoid excessive physical exertion until cleared by your doctor",
        "Maintain a heart-healthy diet low in sodium and saturated fats",
        "Report any new or worsening symptoms immediately",
        "Avoid alcohol, caffeine, and tobacco products"
    ]

    # Track diseases without specific precautions
    diseases_without_precautions = []

    # Collect precautions for each disease
    all_precautions = []
    for disease in diseases:
        if disease in precautions:
            # all_precautions.append(f"For {disease}:")
            all_precautions.extend(precautions[disease])
        else:
            diseases_without_precautions.append(disease)

    # If more than two diseases lack specific precautions, add default precautions just once
    if len(diseases_without_precautions) >= 2:
        all_precautions.append("Default Precautions:")
        all_precautions.extend(default_precautions)

    return all_precautions

# Helper function for retrieving causes and symptoms of detected disease
def get_disease_info(disease):
    disease_info = {
        "1 degree atrioventricular block": {
            "Causes": [
                "Aging-related degeneration of heart conduction pathways",
                "Medications such as beta-blockers or calcium channel blockers",
                "Electrolyte imbalances",
                "Heart infections like myocarditis"
            ],
            "Symptoms": [
                "Usually asymptomatic",
                "Occasional mild dizziness",
                "Fatigue",
                "Slow heart rate in some cases"
            ]
        },
        "2 degree atrioventricular block": {
            "Causes": [
                "Heart disease or coronary artery disease",
                "Inflammation of the heart muscle (myocarditis)",
                "Side effects of cardiac medications",
                "Electrolyte imbalances"
            ],
            "Symptoms": [
                "Skipped or dropped heartbeats",
                "Dizziness or fainting",
                "Shortness of breath",
                "Chest discomfort"
            ]
        },
        "2 degree atrioventricular block (Type one)": {
            "Causes": [
                "Increased vagal tone (e.g., in athletes)",
                "Certain medications (beta-blockers, digoxin)",
                "Inferior myocardial infarction",
                "Electrolyte disturbances"
            ],
            "Symptoms": [
                "Occasional dizziness",
                "Lightheadedness",
                "Usually asymptomatic"
            ]
        },
        "2 degree atrioventricular block (Type two)": {
            "Causes": [
                "Advanced conduction system disease",
                "Severe coronary artery disease",
                "Myocardial infarction",
                "Cardiomyopathy"
            ],
            "Symptoms": [
                "Sudden fainting",
                "Extreme fatigue",
                "Severe dizziness",
                "Shortness of breath"
            ]
        },
        "3 degree atrioventricular block": {
            "Causes": [
                "Heart attack",
                "Congenital heart defects",
                "Degeneration of conduction pathways",
                "Side effects of cardiac medications"
            ],
            "Symptoms": [
                "Severe dizziness",
                "Extreme fatigue",
                "Fainting (syncope)",
                "Shortness of breath",
                "Very slow heart rate"
            ]
        },
        "Atrial Bigeminy": {
            "Causes": [
                "Electrolyte imbalances",
                "Excessive caffeine or alcohol intake",
                "Heart disease",
                "Increased stress or anxiety"
            ],
            "Symptoms": [
                "Palpitations",
                "Skipped heartbeats",
                "Occasional dizziness",
                "Fatigue"
            ]
        },
        "Atrioventricular Block": {
            "Causes": [
                "Aging and degeneration of conduction pathways",
                "Heart disease or prior heart attack",
                "Electrolyte imbalances",
                "Certain medications (beta-blockers, digoxin)"
            ],
            "Symptoms": [
                "Slow heart rate",
                "Dizziness or fainting",
                "Shortness of breath",
                "Fatigue"
            ]
        },
        "Left Bundle Branch Block": {
            "Causes": [
                "Hypertension",
                "Coronary artery disease",
                "Cardiomyopathy",
                "Heart valve disorders"
            ],
            "Symptoms": [
                "Often asymptomatic",
                "Shortness of breath",
                "Fatigue",
                "Chest pain (if underlying coronary artery disease is present)"
            ]
        },
        "Right Bundle Branch Block": {
            "Causes": [
                "Heart attack",
                "Lung diseases (pulmonary embolism, COPD)",
                "Congenital heart disease",
                "Heart infections (myocarditis)"
            ],
            "Symptoms": [
                "Usually asymptomatic",
                "Fatigue",
                "Dizziness",
                "Shortness of breath (if underlying condition is present)"
            ]
        },
        "Myocardial Infarction": {
            "Causes": [
                "Blocked coronary arteries due to plaque buildup",
                "Blood clot formation in coronary arteries",
                "Severe coronary artery spasm",
                "High blood pressure leading to arterial damage"
            ],
            "Symptoms": [
                "Severe chest pain or discomfort",
                "Pain radiating to the arm, jaw, or back",
                "Shortness of breath",
                "Sweating and nausea",
                "Lightheadedness or fainting"
            ]
        },
        "Sinus Bradycardia": {
            "Causes": [
                "Normal in well-trained athletes",
                "Aging-related heart changes",
                "Certain medications (beta-blockers, digoxin)",
                "Electrolyte imbalances"
            ],
            "Symptoms": [
                "Fatigue",
                "Dizziness or lightheadedness",
                "Fainting",
                "Shortness of breath"
            ]
        },
        "Sinus Tachycardia": {
            "Causes": [
                "Exercise",
                "Fever or infections",
                "Anxiety or stress",
                "Dehydration or blood loss"
            ],
            "Symptoms": [
                "Rapid heartbeat",
                "Shortness of breath",
                "Dizziness",
                "Fatigue"
            ]
        },
        "Atrial Fibrillation": {
            "Causes": [
                "High blood pressure",
                "Heart valve disease",
                "Coronary artery disease",
                "Excessive alcohol or caffeine consumption",
                "Overactive thyroid (hyperthyroidism)",
                "Sleep apnea",
                "Lung diseases"
            ],
            "Symptoms": [
                "Irregular heartbeat (palpitations)",
                "Shortness of breath",
                "Dizziness or lightheadedness",
                "Fatigue",
                "Chest pain or discomfort"
            ]
        },
        "Atrial Flutter": {
            "Causes": [
                "Scarring of the heart due to previous surgery",
                "High blood pressure",
                "Heart failure",
                "Thyroid disorders",
                "Chronic lung disease",
                "Congenital heart defects"
            ],
            "Symptoms": [
                "Rapid heartbeat (tachycardia)",
                "Shortness of breath",
                "Fatigue",
                "Dizziness or fainting",
                "Chest discomfort"
            ]
        },
        "Ventricular Tachycardia": {
            "Causes": [
                "Coronary artery disease",
                "Heart attack (myocardial infarction)",
                "Heart failure",
                "Electrolyte imbalances (low potassium or magnesium)",
                "Use of stimulant drugs (cocaine, methamphetamine)"
            ],
            "Symptoms": [
                "Palpitations",
                "Chest pain",
                "Dizziness or fainting",
                "Shortness of breath",
                "Loss of consciousness in severe cases"
            ]
        },
        "Ventricular Fibrillation": {
            "Causes": [
                "Severe heart disease",
                "Heart attack",
                "Electrocution accidents",
                "Drowning",
                "Drug overdose"
            ],
            "Symptoms": [
                "Sudden collapse",
                "Loss of consciousness",
                "No detectable pulse",
                "No breathing",
                "Fatal if untreated within minutes"
            ]
        },
        "ST Elevation": {
            "Causes": [
                "Acute heart attack",
                "Pericarditis (inflammation of the heart lining)",
                "Coronary artery blockage",
                "Spontaneous coronary artery dissection"
            ],
            "Symptoms": [
                "Severe chest pain",
                "Pain radiating to arm, jaw, or back",
                "Shortness of breath",
                "Sweating and nausea",
                "Lightheadedness or fainting"
            ]
        },
        "ST Depression": {
            "Causes": [
                "Partial blockage of coronary arteries",
                "Severe anemia",
                "High blood pressure",
                "Stress-induced heart problems"
            ],
            "Symptoms": [
                "Chest pain (angina)",
                "Fatigue",
                "Shortness of breath",
                "Dizziness"
            ]
        }
    }
    # Return causes & symptoms if disease is found, else return general info
    return disease_info.get(disease, {
        "Causes": ["Unknown cause – Might be an abnormality of the main disease."],
        "Symptoms": ["No specific symptoms. Clinical correlation is necessary."]
    })



# Utility function to convert non-JSON serializable objects to JSON serializable types
def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return obj

from imgToMat import convert_ecg_image_to_12_leads
from pdfToImg import extract_ecg_from_pdf

@app.route("/analyze_manual_leads", methods=["POST"])
def analyze_manual_leads_route():
    global analysis_results  # Ensure we are updating the global dictionary
    print(f"Request data: {request.form}")  # Debugging
    
    valid_extensions = (".mat", ".png", ".jpg", ".jpeg", ".bmp", ".tiff",".pdf")  # Add valid image formats
    mat_file_path =""
    record_path = os.path.join(app.config["UPLOAD_FOLDER"], request.form.get("record_path"))
    if not os.path.exists(record_path):
        return jsonify({"success": False, "message": f"File not found: {record_path}"}), 400

    if not record_path.lower().endswith(valid_extensions):
        return jsonify({"success": False, "message": "Invalid file type. Please upload a .mat file or an image file."}), 400

    try:
        if record_path.lower().endswith(".mat"):
            # Process .mat file directly
            mat_file_path = record_path
            print("Processing .mat file...")
        
        elif record_path.lower().endswith(".pdf"):
        # Convert .pdf to .png
            print("Processing .pdf file...")
            ecg_pp = extract_ecg_from_pdf(record_path)
            print(ecg_pp)
            print("Processing image file...")
            mat_file_path = convert_ecg_image_to_12_leads(
                image_path=ecg_pp,
                sampling_rate=500,  # Adjust sampling rate as needed
                duration=10.0,      # Adjust duration as needed
                patient_id=request.form.get("patient_id", "Anonymous"),
                age=int(request.form.get("patient_age", 0)),
                gender=request.form.get("patient_gender", "Unknown")
            )

        else:
            # Process image file and convert it to a .mat file
            print("Processing image file...")
            mat_file_path = convert_ecg_image_to_12_leads(
                image_path=record_path,
                sampling_rate=500,  # Adjust sampling rate as needed
                duration=10.0,      # Adjust duration as needed
                patient_id=request.form.get("patient_id", "Anonymous"),
                age=int(request.form.get("patient_age", 0)),
                gender=request.form.get("patient_gender", "Unknown")
            )
            print(f"Image converted to .mat file: {mat_file_path}")

        # Proceed with the existing .mat file processing logic
        selected_manual = json.loads(request.form.get("selected_leads", "[]"))
        print(f"Selected Manual Leads: {selected_manual}")  # Debugging

        # Parse JSON strings for dictionaries
        app_lead_names = json.loads(request.form.get("lead_names", "{}"))
        app_normal_label = request.form.get("normal_label", default="Normal")
        app_disease_mapping = json.loads(request.form.get("disease_mapping", "{}"))

        # Ensure lead_names is a dictionary
        if not isinstance(app_lead_names, dict):
            app_lead_names = {}

        # Ensure disease_mapping is a dictionary
        if not isinstance(app_disease_mapping, dict):
            app_disease_mapping = {}

        # Create default lead names if not provided
        for lead in selected_manual:
            lead_key = str(lead)
            if lead_key not in app_lead_names:
                app_lead_names[lead_key] = f"Lead {lead}"

        app_analysis_results = {}

        # Call the analyze_manual_leads function
        result = analyze_manual_leads(
            mat_file_path, selected_manual, app_lead_names, app_normal_label, app_disease_mapping, app_analysis_results
        )

        if result is None:
            return jsonify({"success": False, "message": "No valid leads or segments detected for the manual lead combination."}), 400

        # Make the result JSON serializable
        result = make_json_serializable(result)
        # Full Disease Mapping: Short Form -> Full Form
        disease_mapping = {
            "1AVB": "1 degree atrioventricular block",
            "2AVB": "2 degree atrioventricular block",
            "2AVB1": "2 degree atrioventricular block (Type one)",
            "2AVB2": "2 degree atrioventricular block (Type two)",
            "3AVB": "3 degree atrioventricular block",
            "ABI": "Atrial Bigeminy",
            "ALS": "Axis Left Shift",
            "APB": "Atrial Premature Beats",
            "AQW": "Abnormal Q Wave",
            "ARS": "Axis Right Shift",
            "AVB": "Atrioventricular Block",
            "CCR": "Counterclockwise Rotation",
            "CR": "Clockwise Rotation",
            "ERV": "Early Repolarization of the Ventricles",
            "FQRS": "fQRS Wave",
            "IDC": "Interior Differences Conduction",
            "IVB": "Intraventricular Block",
            "JEB": "Junctional Escape Beat",
            "JPT": "Junctional Premature Beat",
            "LBBB": "Left Bundle Branch Block",
            "LBBBB": "Left Back Bundle Branch Block",
            "LFBBB": "Left Front Bundle Branch Block",
            "LVH": "Left Ventricle Hypertrophy",
            "LVQRSAL": "Lower Voltage QRS in All Lead",
            "LVQRSCL": "Lower Voltage QRS in Chest Lead",
            "LVQRSLL": "Lower Voltage QRS in Limb Lead",
            "MI": "Myocardial Infarction",
            "MIBW": "Myocardial Infarction in Back Wall",
            "MIFW": "Myocardial Infarction in the Front Wall",
            "MILW": "Myocardial Infarction in the Lower Wall",
            "MISW": "Myocardial Infarction in the Side Wall",
            "PRIE": "PR Interval Extension",
            "PWC": "P Wave Change",
            "QTIE": "QT Interval Extension",
            "RAH": "Right Atrial Hypertrophy",
            "RBBB": "Right Bundle Branch Block",
            "RVH": "Right Ventricle Hypertrophy",
            "STDD": "ST Drop Down",
            "STE": "ST Extension",
            "STTC": "ST-T Change",
            "STTU": "ST Tilt Up",
            "TWC": "T Wave Change",
            "TWO": "T Wave Opposite",
            "UW": "U Wave",
            "VB": "Ventricular Bigeminy",
            "VEB": "Ventricular Escape Beat",
            "VFW": "Ventricular Fusion Wave",
            "VPB": "Ventricular Premature Beat",
            "VPE": "Ventricular Preexcitation",
            "VET": "Ventricular Escape Trigeminy",
            "WAVN": "Wandering in the Atrioventricular Node",
            "WPW": "WPW Syndrome",
            "SB": "Sinus Bradycardia",
            "SR": "Sinus Rhythm",
            "AFIB": "Atrial Fibrillation",
            "ST": "Sinus Tachycardia",
            "AF": "Atrial Flutter",
            "SA": "Sinus Irregularity",
            "SVT": "Supraventricular Tachycardia",
            "AT": "Atrial Tachycardia",
            "AVNRT": "Atrioventricular Node Reentrant Tachycardia",
            "AVRT": "Atrioventricular Reentrant Tachycardia",
            "SAAWR": "Sinus Atrium to Atrial Wandering Rhythm"
        }

        # Ensure disease_name is properly deserialized
        disease_name = result["final_disease"]

        # If disease_name is a string, convert it to a set or list
        if isinstance(disease_name, str):
            # Remove the curly braces and split the string into individual codes
            disease_name = disease_name.strip("{}").replace("'", "").split(", ")
            disease_name = set(disease_name)  # Convert to a set for further processing

        # Map the disease codes to their full forms using disease_mapping
        if isinstance(disease_name, (set, list)):
            disease_names = [disease_mapping.get(code.strip(), code.strip()) for code in disease_name]
        else:
            # Handle unexpected data types gracefully
            disease_names = [disease_mapping.get(disease_name.strip(), disease_name.strip())]

        # Convert disease_names to a list to make it JSON serializable
        disease_names = list(disease_names)

        # Print the updated disease_name for debugging
        print(disease_names)

        # Get additional disease information if available
        disease_info = [get_disease_info(disease) for disease in disease_names]

        # Update the global analysis_results dictionary
        analysis_results.update({
            "final_disease": disease_names,  # Ensure this is a list
            "confidence": result["confidence"],
            "disease_info": disease_info,
            "plot_path": result["plot_path"],
            "lead_predictions": result["lead_predictions"],
            "fs": result["fs"],
            "patient_name": request.form.get("patient_name", "Patient X"),
            "patient_age": request.form.get("patient_age", "50"),
            "patient_gender": request.form.get("patient_gender", "Male"),
            "doctor_name": request.form.get("doctor_name", "Dr. Srinivasan"),
            "doctor_designation": request.form.get("doctor_designation", "Cardiologist"),
        })

        return jsonify({
            "success": True,
            "message": "Manual lead analysis complete.",
            "final_disease": disease_names,  # Ensure this is a list
            "confidence": result["confidence"],
            "disease_info": disease_info,
            "plot_url": url_for('static', filename=os.path.basename(result["plot_path"])),
        })
    except Exception as e:
        print(f"Error in analyze_manual_leads_route: {e}")
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"}), 400


@app.route("/")
def home():
    return render_template("index.html", disease_mapping=disease_mapping)

if __name__ == "__main__":
    app.run(debug=True)

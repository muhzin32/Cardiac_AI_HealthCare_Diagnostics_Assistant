# Cardiac Wellness ECG Analysis System

A Flask-based web application for analyzing ECG signals and detecting cardiac abnormalities.

## Key Features
- ECG signal processing and analysis
- Multiple lead selection options (single, double, multi)
- Automated disease detection using machine learning
- PDF report generation with detailed findings
- Web interface for easy interaction

## Dependencies
The application requires the following Python packages:
- Flask (web framework)
- Keras (deep learning)
- TensorFlow (machine learning)
- scikit-learn (machine learning)
- OpenCV (image processing)
- Pandas (data analysis)
- NumPy (numerical computing)
- PyWavelets (signal processing)
- WFDB (ECG processing)
- Matplotlib (visualization)
- ReportLab (PDF generation)

## Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
Start the development server:
```bash
python Cardiac_Flask_April4/Flaskapp/app.py
```
The application will be available at `http://localhost:5000`

## ECG Analysis Types
The system supports three types of ECG analysis:

### 1. Single Lead Analysis
- Analyzes one selected ECG lead
- Basic detection of abnormalities
- Quick preliminary results

### 2. Double Lead Analysis
- Compares two selected ECG leads
- Enhanced detection accuracy
- Better localization of abnormalities

### 3. Multi-Lead Analysis
- Processes multiple ECG leads simultaneously
- Most comprehensive analysis
- Highest detection accuracy
- Generates detailed PDF report

## API Endpoints
- `/` - Main application interface
- `/analyze_manual_leads` - POST endpoint for manual lead analysis
- `/generate_report` - POST endpoint for PDF report generation

## Workflow
1. Upload ECG file (.mat, .pdf or image format)
2. Select leads for analysis
3. System processes the ECG signals
4. View detected abnormalities
5. Generate detailed PDF report

## Report Contents
The generated PDF report includes:
- Patient and doctor information
- ECG plot with highlighted abnormalities
- Detected conditions with confidence levels
- Disease descriptions (causes and symptoms)
- Recommended tests
- Precautions for the patient
- Disease images (when available)

## Model Architecture
The analysis uses a combination of:
- Wavelet transforms for signal processing
- Deep learning models for feature extraction
- Rule-based systems for final diagnosis
- Ensemble methods for improved accuracy (not yet updated, as of now multi-label classification is used)

## File Structure
- `Flaskapp/` - Main application directory
  - `app.py` - Flask application entry point
  - `static/` - Static files (images, reports)
  - `templates/` - HTML templates
- `models/` - Machine learning models
- `utils/` - Utility functions and helpers


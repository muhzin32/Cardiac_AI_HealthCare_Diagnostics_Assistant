import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import datetime
import scipy.io as sio

def process_single_lead(roi, sampling_rate=500, duration=10.0):
    """
    Process a single lead ROI from an ECG image.
    Returns the resampled ECG signal.
    """
    # Convert ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply binary thresholding (invert so the trace becomes white)
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Find grid lines using morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_grid = cv2.erode(binary, horizontal_kernel)
    horizontal_grid = cv2.dilate(horizontal_grid, horizontal_kernel)
    vertical_grid = cv2.erode(binary, vertical_kernel)
    vertical_grid = cv2.dilate(vertical_grid, vertical_kernel)

    # Combine grid lines and remove them to isolate the ECG signal
    grid = cv2.bitwise_or(horizontal_grid, vertical_grid)
    signal = cv2.bitwise_and(binary, cv2.bitwise_not(grid))

    # Extract the ECG trace: for each column, find the row index of the trace
    trace_rows = []
    trace_cols = []
    rows, cols = signal.shape
    for col in range(cols):
        col_values = signal[:, col]
        if np.sum(col_values) > 0:  # if any signal is present in the column
            row_indices = np.where(col_values > 0)[0]
            if len(row_indices) > 0:
                # Use the average row index as the trace location for this column
                row = int(np.mean(row_indices))
                trace_rows.append(row)
                trace_cols.append(col)

    if len(trace_rows) == 0:
        raise ValueError("No ECG trace detected in this ROI.")

    # Ensure ordering by column
    sorted_indices = np.argsort(trace_cols)
    trace_rows = np.array(trace_rows)[sorted_indices]

    # Convert pixel rows to voltage values (scaled to -1.0 to 1.0)
    voltage_values = (rows - trace_rows) / rows * 2.0 - 1.0

    # Smooth the extracted trace to reduce jaggedness
    voltage_values = savgol_filter(voltage_values, 15, 3)

    # Resample the signal to the target sampling rate
    num_samples = int(sampling_rate * duration)
    resampled_values = np.interp(
        np.linspace(0, len(voltage_values) - 1, num_samples),
        np.arange(len(voltage_values)),
        voltage_values
    )
    # Scale to a typical ECG amplitude (in mV, here assumed -1mV to 1mV)
    resampled_values *= 1.0
    return resampled_values

# Example load_ecg function matching your format
def load_ecg(record_path, lead_index=0):
    mat_file = record_path if record_path.endswith(".mat") else record_path + ".mat"
    mat_data = sio.loadmat(mat_file)
    signal = mat_data['val'][lead_index]
    return signal, 500

def convert_ecg_image_to_12_leads(image_path, sampling_rate=500, duration=10.0,
                                  patient_id="Anonymous", age=0, gender="Unknown"):
    """
    Convert an ECG image into 12 separate leads and compile them into a single .mat file.
    The image is assumed to be arranged in a 4-row by 3-column grid with the following lead order:
      Row 1: Lead I, Lead II, Lead III
      Row 2: aVR, aVL, aVF
      Row 3: V1, V2, V3
      Row 4: V4, V5, V6
    Returns the combined .mat file path.
    """
    # Load the full ECG image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img_height, img_width, _ = image.shape
    num_rows, num_cols = 4, 3  # fixed segmentation
    roi_height = img_height // num_rows
    roi_width = img_width // num_cols

    # Define the standard lead order
    lead_order = [
        "Lead I", "Lead II", "Lead III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6"
    ]
    
    lead_signals = []
    for lead_index, lead in enumerate(lead_order):
        row = lead_index // num_cols
        col = lead_index % num_cols
        x = col * roi_width
        y = row * roi_height
        roi = image[y:y+roi_height, x:x+roi_width]
        
        try:
            signal = process_single_lead(roi, sampling_rate, duration)
        except Exception as e:
            print(f"Error processing {lead}: {e}")
            # If processing fails, fill with zeros
            signal = np.zeros(int(sampling_rate * duration))
        lead_signals.append(signal)
        print(f"Processed {lead}")

    # Create a 2D NumPy array: each row corresponds to one lead
    signals_array = np.array(lead_signals)
    
    # Compile combined data into a single dictionary for .mat file
    combined_data = {
        'val': signals_array,  # 2D array with shape (12, num_samples)
        'fs': sampling_rate,
        'duration': duration,
        'patient_id': patient_id,
        'age': age,
        'gender': gender,
        'source_image': os.path.basename(image_path),
        'conversion_date': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    
    # Save the combined .mat file
    output_dir = os.path.dirname(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    combined_mat_path = os.path.join(output_dir, f"{base_filename}.mat")
    sio.savemat(combined_mat_path, combined_data)
    print(f"Combined .mat file saved as: {combined_mat_path}")
    return combined_mat_path

if __name__ == "__main__":
    pass
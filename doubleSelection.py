import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from scipy.signal import butter, filtfilt, find_peaks, resample
import pywt

# ========== Custom Attention Layer ==========
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(name="attention_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        weighted = x * a
        return tf.reduce_sum(weighted, axis=1)

# --- Utility Functions ---
def bandpass_filter(signal, lowcut=0.4, highcut=50.0, fs=500, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def adaptive_wavelet_denoise(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in denoised_coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(signal)]

def load_ecg(record_path, lead_index=0):
    mat_file = record_path if record_path.endswith(".mat") else record_path + ".mat"
    mat_data = sio.loadmat(mat_file)
    signal = mat_data['val'][lead_index]
    return signal, 500

def detect_r_peaks(ecg_signal, fs):
    threshold = np.mean(ecg_signal) + np.std(ecg_signal)
    peaks, _ = find_peaks(ecg_signal, height=threshold, distance=fs//2)
    return peaks

def segment_ecg_by_r_peaks(ecg_signal, r_peaks, window_size=250):
    segments = []
    seg_indices = []
    for peak in r_peaks:
        start = max(0, peak - window_size // 2)
        end = start + window_size
        if end <= len(ecg_signal):
            segments.append(ecg_signal[start:end])
            seg_indices.append((start, end))
    return np.array(segments), seg_indices

# ========== Preprocessing ==========
def preprocess_ecg_basic(record_path, lead_index=0):
    signal, fs = load_ecg(record_path, lead_index)
    filtered_signal = bandpass_filter(signal, lowcut=0.4, highcut=50.0, fs=fs)
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    r_peaks = detect_r_peaks(normalized_signal, fs)
    return normalized_signal, fs, r_peaks

def preprocess_ecg_wavelet(record_path, lead_index=0):
    signal, fs = load_ecg(record_path, lead_index)
    filtered_signal = bandpass_filter(signal, lowcut=0.4, highcut=50.0, fs=fs)
    denoised_signal = adaptive_wavelet_denoise(filtered_signal)
    normalized_signal = (denoised_signal - np.mean(denoised_signal)) / np.std(denoised_signal)
    normalized_signal = normalized_signal.astype(np.float32)
    r_peaks = detect_r_peaks(normalized_signal, fs)
    return normalized_signal, fs, r_peaks

# ========== Load Models and Label Encoders ==========
models_basic = []
label_encoders_basic = []
for lead in range(6):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DOUBLE_LABEL", "models", f"model_lead_{lead}.h5")
    model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    models_basic.append(model)
    label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DOUBLE_LABEL", "labels", f"labels_lead_{lead}.pkl")
    with open(label_path, "rb") as f:
        le = pickle.load(f)
    label_encoders_basic.append(le)
    print("Loaded model and label encoder for lead", lead)

models_wavelet = {}
label_encoders_wavelet = {}
for lead in range(6, 12):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DOUBLE_LABEL", "models", f"model_lead_{lead}.h5")
    label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "DOUBLE_LABEL", "labels", f"labels_lead_{lead}.pkl")
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    models_wavelet[lead] = loaded_model
    with open(label_path, "rb") as f:
        le = pickle.load(f)
    label_encoders_wavelet[lead] = le
    print("Loaded model and label encoder for lead", lead)

# ========== Detection Functions ==========
def detect_leads(record_path, selected_leads, window_size_basic=250, window_size_wavelet=250):
    """
    Processes only the selected leads.
    For leads 0–5, basic preprocessing is applied.
    For leads 6–11, wavelet-based preprocessing is applied.
    """
    lead_predictions_all = {}
    fs = None
    for lead in selected_leads:
        if lead < 6:
            normalized_signal, fs, r_peaks = preprocess_ecg_basic(record_path, lead_index=lead)
            segments, seg_indices = segment_ecg_by_r_peaks(normalized_signal, r_peaks, window_size=window_size_basic)
            if len(segments) == 0:
                print(f"No valid segments extracted from lead {lead}.")
                lead_predictions_all[lead] = {
                    "pred_labels": [],
                    "seg_indices": [],
                    "signal": normalized_signal,
                    "r_peaks": r_peaks
                }
                continue
            segments = segments.reshape(-1, window_size_basic, 1)
            predictions = models_basic[lead].predict(segments, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            try:
                pred_labels = label_encoders_basic[lead].inverse_transform(pred_classes)
            except AttributeError:
                pred_labels = np.array([label_encoders_basic[lead][idx] for idx in pred_classes])
            lead_predictions_all[lead] = {
                "pred_labels": pred_labels,
                "seg_indices": seg_indices,
                "signal": normalized_signal,
                "r_peaks": r_peaks
            }
        else:
            normalized_signal, fs, r_peaks = preprocess_ecg_wavelet(record_path, lead_index=lead)
            segments, seg_indices = segment_ecg_by_r_peaks(normalized_signal, r_peaks, window_size=window_size_wavelet)
            if len(segments) == 0:
                print(f"⚠️ No valid segments for lead {lead}.")
                lead_predictions_all[lead] = {
                    "pred_labels": [],
                    "seg_indices": [],
                    "signal": normalized_signal,
                    "r_peaks": r_peaks
                }
                continue
            target_length = 26 * 11
            resampled_segments = []
            for seg in segments:
                resampled_seg = resample(seg, target_length)
                resampled_seg = resampled_seg.reshape(26, 11)
                resampled_segments.append(resampled_seg)
            resampled_segments = np.array(resampled_segments, dtype=np.float32)
            predictions = models_wavelet[lead].predict(resampled_segments, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            try:
                pred_labels = label_encoders_wavelet[lead].inverse_transform(pred_classes)
            except AttributeError:
                pred_labels = np.array([label_encoders_wavelet[lead][idx] for idx in pred_classes])
            lead_predictions_all[lead] = {
                "pred_labels": pred_labels,
                "seg_indices": seg_indices,
                "signal": normalized_signal,
                "r_peaks": r_peaks
            }
    return lead_predictions_all, fs

# def select_double_final_disease(lead_predictions_all, selected_leads, primary_leads=range(0,6), secondary_leads=range(6,12)):
#     """
#     Custom algorithm to select the final disease.
#     Each lead contributes based on its label distribution and confidence.
#     Primary leads (I–aVF; indices 0–5) are given twice the weight of secondary leads (V1–V6; indices 6–11).
#     """
#     disease_scores = {}
#     for lead in selected_leads:
#         data = lead_predictions_all.get(lead)
#         if data is None or len(data["pred_labels"]) == 0:
#             continue
#         seg_count = len(data["pred_labels"])
#         labels, counts = np.unique(data["pred_labels"], return_counts=True)
#         max_idx = np.argmax(counts)
#         lead_label = labels[max_idx]
#         # Convert numpy types to Python native types
#         if isinstance(lead_label, (np.integer, np.int64, np.int32)):
#             lead_label = int(lead_label)
#         elif isinstance(lead_label, (np.floating, np.float64, np.float32)):
#             lead_label = float(lead_label)
#         elif isinstance(lead_label, np.ndarray):
#             lead_label = lead_label.tolist()
#         confidence_factor = counts[max_idx] / seg_count
#         weight = 2 if lead in primary_leads else 1
#         score = weight * counts[max_idx] * confidence_factor
#         disease_scores[lead_label] = disease_scores.get(lead_label, 0) + score
#         # confidence_factor = round(max(disease_scores[lead_label]) * 100)
#     if not disease_scores:
#         return None
    
#     final_disease = max(disease_scores, key=disease_scores.get)
#     confidence_factor = round(disease_scores[final_disease] / sum(disease_scores.values()) * 100, 2)
#     label_file_path = "models\labelDouble.json"
#     with open(label_file_path, "r") as label_file:
#         label_data = label_file.read()
#         # Parse the text file content into a list
#         label_mapping = eval(label_data)  # Use eval to parse the list-like string

#     # Handle final_disease as an index to the list
#     try:
#         full_disease_name = label_mapping[int(final_disease)]
#     except (ValueError, IndexError, TypeError):
#         full_disease_name = "Unknown Disease"

#     return full_disease_name, confidence_factor

def select_double_final_disease(lead_predictions_all, selected_leads, primary_leads=range(0,6), secondary_leads=range(6,12)):
    total_weighted_counts = {}
    for lead in selected_leads:
        data = lead_predictions_all.get(lead)
        if data is None or len(data["pred_labels"]) == 0:
            continue
        weight = 2 if lead in primary_leads else 1
        labels, counts = np.unique(data["pred_labels"], return_counts=True)
        for label, count in zip(labels, counts):
            # Convert numpy types to Python native types for consistency
            if isinstance(label, (np.integer, np.int64, np.int32)):
                label = int(label)
            elif isinstance(label, (np.floating, np.float64, np.float32)):
                label = float(label)
            elif isinstance(label, np.ndarray):
                label = label.tolist()
            # Initialize count if label not yet in dictionary
            if label not in total_weighted_counts:
                total_weighted_counts[label] = 0
            # Add weighted count
            total_weighted_counts[label] += weight * count

    if not total_weighted_counts:
        return None
    final_disease = max(total_weighted_counts, key=total_weighted_counts.get)
    confidence = round(total_weighted_counts[final_disease] / sum(total_weighted_counts.values()) * 100, 2)
    label_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "labelDouble.json")
    with open(label_file_path, "r") as label_file:
        label_data = label_file.read()
        # Parse the text file content into a list
        label_mapping = eval(label_data)
        # Use eval to parse the list-like string
    # Handle final_disease as an index to the list
    try:
        full_disease_name = label_mapping[int(final_disease)]
    except (ValueError, IndexError, TypeError):
        full_disease_name = "Unknown Disease"
    # Add the full disease name to the dictionary
    # total_weighted_counts[full_disease_name] = total_weighted_counts.pop(final_disease)
    # Convert the final_disease to a string for consistency
    final_disease = str(full_disease_name)
    return final_disease, confidence


def analyze_double(file_path, selected_leads=None):
    if selected_leads is None:
        selected_leads = list(range(12))
    base_path = file_path.rstrip(".mat")
    lead_predictions_all, fs = detect_leads(base_path, selected_leads, window_size_wavelet=250)
    final_disease , confidence = select_double_final_disease(lead_predictions_all, selected_leads)
    print(final_disease)
    print(selected_leads)
    # print(lead_predictions_all)
    print(confidence)

    return {"final_disease": final_disease, "lead_predictions": lead_predictions_all, "fs": fs , "confidence": confidence}



if __name__ == "__main__":
    pass
    # def analyze_double(file_path, selected_leads=None):
    #     if selected_leads is None:
    #         selected_leads = list(range(12))
    #     base_path = file_path.rstrip(".mat")
    #     lead_predictions_all, fs = detect_leads(base_path, selected_leads, window_size_wavelet=250)
    #     final_disease , confidence = select_double_final_disease(lead_predictions_all, selected_leads)
    #     print(final_disease)
    #     print(selected_leads)
    #     # print(lead_predictions_all)
    #     print(confidence)

    #     return {"final_disease": final_disease, "lead_predictions": lead_predictions_all, "fs": fs , "confidence": confidence}
    
    # analyze_double("D:/SETV/Cardiac_Wellness_April9th/Cardiac_Wellness_April8th/Cardiac_Wellness/Cardiac_Flask_April/Flaskapp/uploads/JS08246", selected_leads=[1,2,3,4,7,8,9])

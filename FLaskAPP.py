import os
import numpy as np
import scipy.io
import tensorflow as tf
from flask import Flask, request, jsonify
from scipy.signal import butter, filtfilt, find_peaks
import pickle
from werkzeug.utils import secure_filename

# --- Initialize Flask App ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Custom Attention Layer Definition ---
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        weighted = x * a
        return tf.reduce_sum(weighted, axis=1)

# --- Load Models ---
abnormality_model = tf.keras.models.load_model('Abnormality_Cardiac.h5')
single_model = tf.keras.models.load_model('model_single.h5', custom_objects={'AttentionLayer': AttentionLayer})
double_model = tf.keras.models.load_model('model_double.h5', custom_objects={'AttentionLayer': AttentionLayer})
multi_model = tf.keras.models.load_model('model_multi.h5', custom_objects={'AttentionLayer': AttentionLayer})

# --- Load Labels ---
with open('labels_single.pkl', 'rb') as f:
    labels_single = pickle.load(f)
with open('labels_double.pkl', 'rb') as f:
    labels_double = pickle.load(f)
with open('label_multi.npy', 'rb') as f:
    labels_multi = np.load(f, allow_pickle=True)

# --- Bandpass Filter Function ---
def bandpass_filter(signal, lowcut=0.4, highcut=50.0, fs=500, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Preprocess ECG Data ---
def preprocess_ecg(file_path):
    mat_data = scipy.io.loadmat(file_path)
    signal = mat_data['val'][0]
    fs = 500
    filtered_signal = bandpass_filter(signal, lowcut=0.4, highcut=50.0, fs=fs)
    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    r_peaks = find_peaks(normalized_signal, height=np.mean(normalized_signal) + np.std(normalized_signal), distance=fs//2)[0]
    return normalized_signal, fs, r_peaks

# --- Segment ECG Beats ---
def segment_ecg_by_r_peaks(ecg_signal, r_peaks, window_size=250):
    segments = [ecg_signal[max(0, p - window_size // 2): p + window_size // 2] for p in r_peaks if p + window_size // 2 <= len(ecg_signal)]
    return np.array(segments)

# --- Analyze ECG Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze_ecg():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided."})
    
    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".mat"):
        return jsonify({"success": False, "error": "Invalid file type."})

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(file_path)

    ecg_signal, fs, r_peaks = preprocess_ecg(file_path)
    segments = segment_ecg_by_r_peaks(ecg_signal, r_peaks, window_size=250)
    if len(segments) == 0:
        return jsonify({"success": False, "error": "No valid ECG segments extracted."})
    
    segments = segments.reshape(-1, 250, 1)
    abnormal_flags = abnormality_model.predict(segments, verbose=0).flatten() > 0.5

    single_preds = single_model.predict(segments, verbose=0)
    double_preds = double_model.predict(segments, verbose=0)
    multi_preds = multi_model.predict(segments, verbose=0)

    final_labels = []
    for i, flag in enumerate(abnormal_flags):
        if not flag:
            final_labels.append("Normal")
        else:
            confidences = [max(single_preds[i]), max(double_preds[i]), max(multi_preds[i])]
            best_model = np.argmax(confidences)
            if best_model == 0:
                best_label = labels_single[np.argmax(single_preds[i])]
            elif best_model == 1:
                best_label = labels_double[np.argmax(double_preds[i])]
            else:
                indices = np.where(multi_preds[i] > 0.5)[0]
                best_label = ", ".join([labels_multi[j] for j in indices]) if len(indices) > 0 else "Unclassified"
            final_labels.append(best_label)

    response = {"success": True, "time": list(np.arange(len(ecg_signal)) / fs), "signal": list(ecg_signal), "r_peaks": list(r_peaks), "segment_labels": final_labels}
    return jsonify(response)

# --- Run Flask App ---
if __name__ == "__main__":
    app.run(debug=True)

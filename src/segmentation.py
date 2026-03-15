import numpy as np
import pandas as pd
import neurokit2 as nk
from wfdb import processing
from src.preprocessing import bandpass_filter

def detect_r_peaks(signal, fs, method='wfdb', filtered=True):
    if len(signal) < 3:
        return np.array([])
    if not filtered:
        signal = bandpass_filter(signal, 0.5, 45.0, fs)

    if method == 'wfdb':
        xqrs = processing.XQRS(sig=signal, fs=fs)
        xqrs.detect()
        candidates = xqrs.qrs_inds
    else:
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        candidates = info.get('ECG_R_Peaks', [])

    radius = int(0.05 * fs)
    refined = []
    for idx in candidates:
        start = max(0, idx - radius)
        end = min(len(signal), idx + radius)
        window = signal[start:end]
        if window.size == 0: continue
        refined.append(start + np.argmax(np.abs(window)))

    refined = np.unique(refined)
    if refined.size < 2:
        return refined

    rr = np.diff(refined) / fs
    final = [refined[0]]
    for i, dt in enumerate(rr):
        if 0.3 < dt < 2.0:
            final.append(refined[i + 1])
    return np.array(final)

def get_sliding_windows(signal, r_peaks, fs, num_peaks=3, pre_sec=0.1, post_sec=0.2, min_sec=0.3):
    """
    Fungsi dinamis untuk mengambil sliding windows. 
    Ubah parameter num_peaks menjadi 3, 5, atau 10 sesuai kebutuhan (3R, 5R, 10R).
    """
    windows = []
    waves = {}
    n = len(r_peaks)
    if n < num_peaks:
        return windows, waves

    try:
        clean = nk.ecg_clean(signal, sampling_rate=fs)
        _, waves = nk.ecg_delineate(clean, r_peaks, sampling_rate=fs, method="dwt", show=False)
    except Exception as e:
        waves = {}

    pre_samples  = int(pre_sec  * fs)
    post_samples = int(post_sec * fs)
    min_samples  = int(min_sec  * fs)
    offset = num_peaks - 1

    for i in range(n - offset):
        r_start = r_peaks[i]
        r_end = r_peaks[i + offset]

        # P-onset
        p_on_list = waves.get("ECG_P_Onsets", [])
        if i < len(p_on_list) and not pd.isna(p_on_list[i]):
            p_on = int(p_on_list[i])
        else:
            p_on = max(0, r_start - pre_samples)

        # T-offset
        t_off_list = waves.get("ECG_T_Offsets", [])
        if i+offset < len(t_off_list) and not pd.isna(t_off_list[i+offset]):
            t_off = int(t_off_list[i+offset])
        else:
            t_off = min(len(signal), r_end + post_samples)

        start = np.clip(p_on, 0, len(signal)-1)
        end   = np.clip(t_off, 0, len(signal))

        if (end - start) >= min_samples:
            win = signal[start:end]
            windows.append((start, end, win))

    return windows, waves

def resample_signal(signal, target_len=600):
    original_len = len(signal)
    if original_len == target_len: return signal
    elif original_len > target_len: return signal[:target_len]
    else:
        padding = target_len - original_len
        return np.pad(signal, (0, padding), 'constant')

def normalize_per_window(X):
    return (X - np.mean(X, axis=1, keepdims=True)) / (np.std(X, axis=1, keepdims=True) + 1e-8)

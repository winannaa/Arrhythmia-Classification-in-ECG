import numpy as np
import neurokit2 as nk
from wfdb import processing


def detect_r_peaks(signal, fs, method='wfdb', filtered=True):
    if len(signal) < 3:
        return np.array([])

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
        if window.size == 0:
            continue

        refined.append(start + np.argmax(window))

    return np.array(refined)


def sliding_windows_by_pqst(signal, r_peaks, fs,
                            pre_sec=0.1,
                            post_sec=0.2,
                            min_sec=0.3):

    windows = []
    waves = {}
    n = len(r_peaks)

    if n < 3:
        return windows, waves

    try:
        clean = nk.ecg_clean(signal, sampling_rate=fs)
        _, waves = nk.ecg_delineate(
            clean,
            r_peaks,
            sampling_rate=fs,
            method="dwt",
            show=False
        )
    except Exception:
        waves = {}

    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)

    for i in range(n - 2):

        start = r_peaks[i] - pre_samples
        end = r_peaks[i + 2] + post_samples

        if start < 0 or end > len(signal):
            continue

        window = signal[start:end]

        if len(window) < int(min_sec * fs):
            continue

        windows.append(window)

    return windows, waves

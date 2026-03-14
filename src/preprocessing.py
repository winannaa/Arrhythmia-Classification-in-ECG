import os
import random
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def bandpass_filter(signal, fs, lowcut=0.5, highcut=45.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered


def zscore_normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std

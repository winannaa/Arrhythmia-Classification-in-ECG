# Arrhythmia Classification in ECG Signals Using GRU and BiGRU

This repository contains my undergraduate thesis project in Biomedical Engineering. It implements a Deep Learning approach to automatically classify cardiac arrhythmias from Electrocardiogram (ECG) signals. Utilizing the MIT-BIH Arrhythmia Database, the project evaluates the effectiveness of Gated Recurrent Unit (GRU) and Bidirectional GRU (BiGRU) architectures, combined with R-peak-based sliding window segmentation and hyperparameter tuning.

## What this project does
This system processes ECG signals and then classifies eight different heart rhythm annotations
- AFIB (Atrial Fibrillation)
- AFL (Atrial Flutter)
- P (Paced Rhythm)
- B (Ventricular Bigeminy)
- VT (Ventricular Tachycardia)
- SVTA (Supraventricular Tachyarrhythmia)
- NOD (Nodal (A-V Junctional) Rhythm)
- N (Normal Rhythm)

The following is a block diagram of the system stages developed to classify arrhythmias based on ECG signals from the MIT-BIH Arrhythmia Database.

<img width="1966" height="576" alt="Diagram Alir Aktivitas Penelitian-Diagram Blok Desain Sistem drawio (1)" src="https://github.com/user-attachments/assets/2bd94a13-07e6-4a92-9488-b2e4372b25d5" />

As illustrated in the system block diagram above, the project follows a structured workflow:
1. **Data Preparation & EDA:** Extracting the MIT-BIH Arrhythmia Database and performing Exploratory Data Analysis to understand the signal characteristics and class distributions.
2. **Data Pre-Processing:** Cleaning the continuous ECG signals using a 4th-order Butterworth bandpass filter (0.5 - 45 Hz) to remove baseline wander and high-frequency noise, followed by Z-score normalization per window.
3. **Sliding Window:** Segmenting the long signals into fixed-length segments based on R-peak counts (3R, 5R, and 10R-peak windows) to capture dynamic temporal features.
4. **Classification Model:** Feeding the processed segments into two main Deep Learning architectures for comparison: GRU Variants (GRU0-GRU4) and Bidirectional GRU (BiGRU0-BiGRU4).
5. **Evaluation:** Assessing the models performance using comprehensive metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

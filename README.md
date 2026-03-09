# Arrhythmia Classification in ECG Signals Using GRU and BiGRU

This repository contains my undergraduate thesis project in Biomedical Engineering. It implements a Deep Learning approach to automatically classify cardiac arrhythmias from Electrocardiogram (ECG) signals. Utilizing the MIT-BIH Arrhythmia Database, the project evaluates the effectiveness of Gated Recurrent Unit (GRU) and Bidirectional GRU (BiGRU) architectures, combined with R-peak-based sliding window segmentation and hyperparameter tuning.

## What this project does
This system processes raw ECG signals and classifies them into eight distinct cardiac rhythms:
- **AFIB** (Atrial Fibrillation) 
- **AFL** (Atrial Flutter) 
- **P** (Paced Rhythm) 
- **B** (Ventricular Bigeminy) 
- **VT** (Ventricular Tachycardia) 
- **SVTA** (Supraventricular Tachyarrhythmia) 
- **NOD** (Nodal / A-V Junctional Rhythm) 
- **N** (Normal Sinus Rhythm) 

The block diagram below illustrates the end-to-end pipeline developed for this arrhythmia classification using the MIT-BIH Arrhythmia Database.

<img width="1966" height="576" alt="Diagram Alir Aktivitas Penelitian-Diagram Blok Desain Sistem drawio (1)" src="https://github.com/user-attachments/assets/2bd94a13-07e6-4a92-9488-b2e4372b25d5" />

As illustrated in the system block diagram above, the project follows a structured workflow.
1. **Data Preparation & EDA** involves extracting the MIT-BIH Arrhythmia Database and performing Exploratory Data Analysis to understand the signal characteristics and class distributions.
2. **Data Pre-Processing** cleans the continuous ECG signals using a 4th-order Butterworth bandpass filter (0.5 - 45 Hz) to remove baseline wander and high-frequency noise, followed by Z-score normalization per window.
3. **Sliding Window** segments the long signals into fixed-length segments based on R-peak counts (3R, 5R, and 10R-peak windows) to capture dynamic temporal features.
4. **Classification Model** feeds the processed segments into two main Deep Learning architectures for comparison, namely GRU Variants (GRU0-GRU4) and Bidirectional GRU (BiGRU0-BiGRU4).
5. **Evaluation** assesses the model performance using comprehensive metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Classification Model
#### 1. GRU Variants (Custom Implementations)
The custom GRU variants (GRU1 - GRU4) implemented in the `src/` directory are heavily adapted and optimized from the foundational concepts presented by [A. Kumarsinha](https://github.com/abhaskumarsinha/GRU-varients).

To make these raw architectures suitable for robust training on complex physiological time-series data (ECG) and fully compatible with the modern TensorFlow/Keras ecosystem, several major engineering improvements were applied to the original implementations:

* **Tensor Operation Optimization:** Replaced memory-intensive `tf.einsum` operations and manual bias matrices with highly efficient `tf.matmul` operations, leveraging TensorFlow's native broadcasting mechanisms to speed up computation.
* **Keras Serialization:** Integrated the `@register_keras_serializable()` decorator and explicitly defined `self.state_size`. This ensures the custom layers are fully compatible with the Keras API, allowing the trained models to be saved and loaded successfully for deployment.
* **Regularization (Dropout):** Introduced a `dropout_rate` parameter directly within the cell state calculations to mitigate overfitting, which was critical for achieving high validation accuracy.
* **Weight Initialization:** Switched the initialization method from standard `random_uniform` to `glorot_uniform` (Xavier Initialization) to stabilize gradient flow and accelerate model convergence during training.

#### 2. Custom Bidirectional Architecture (BiGRU)
This project implements a **Dual-RNN Bidirectional pipeline from scratch**:
* **Forward Path:** A standard RNN layer that processes the ECG sequence in chronological order.
* **Backward Path:** A parallel RNN layer that processes the sequence in reverse using the `go_backwards=True` argument. To ensure perfect temporal alignment with the forward path, a custom `tf.reverse` operation is applied to the output.
* **Feature Fusion:** The learned temporal representations from both directions are cleanly merged using `tf.keras.layers.Concatenate(axis=-1)` before being passed to the dense classification layers.

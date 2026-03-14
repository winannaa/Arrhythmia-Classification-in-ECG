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

### Experiments
To find the most optimal model for ECG arrhythmia classification, this project employed a comprehensive **Grid Search** approach. A robust training pipeline was designed to systematically evaluate and loop through multiple combinations of data segmentation strategies, architectures, and hyperparameters.

The findings from this exhaustive grid search are categorized into three main experiments:

#### 1. **Hyperparameter Tuning** 
- Involved a systematic search across units (32, 64, 128), dropout rates (0.2, 0.5), and learning rates (0.001, 0.0001).
- The best configuration was **128 units, a dropout rate of 0.2, and a learning rate of 0.001**. This configuration was retained for further experimentation.

#### 2. **Architecture Comparison** 
- Compared conventional GRU (GRU0) with four mathematical variants (GRU1-4) and five bidirectional architectures (BiGRU0-4). The results showed that conventional GRU0 emerged as the best architecture. Below is a table of the 3 best architectures.
  
| No. | Architecture | Accuracy | Precision | Recall | F1-Score | AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **GRU0 (Conventional)** | **95.99%** | **95.82%** | **95.99%** | **95.80%** | **0.99** |
| 2 | GRU2 (Variant) | 95.78% | 95.63% | 95.78% | 95.63% | 0.99 |
| 3 | BiGRU0 (Bidirectional)| 95.73% | 95.66% | 95.73% | 95.61% | 0.99 |

#### 3. **Sliding Window Variation** 
- Analyzed the effect of different segmentation lengths based on R-peak counts (3R, 5R, and 10R windows) on the winning architecture. The following are the performance results of the three segmentation lengths.

| Window Size | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **3R-Peak** | **95.99%** | **95.82%** | **95.99%** | **95.80%** | **0.99** |
| 5R-Peak | 95.55% | 95.53% | 95.55% | 95.42% | 0.99 |
| 10R-Peak | 94.62% | 94.67% | 94.62% | 94.64% | 0.98 |

## Getting Started
Follow the steps below to set up the environment and reproduce the experiments in this repository.

### 1. Clone the Repository
```bash
git clone https://github.com/winanannaa/Arrhythmia-Classification-in-ECG.git
cd Arrhythmia-Classification-in-ECG
```
### 2. Create a Virtual Environment
Creating a virtual environment helps avoid dependency conflicts.
```bash
python -m venv venv
```
Activate the environment:
- Windows
  ```bash
  venv\Scripts\activate
  ```
- Linux / Mac
  ```bash
  source venv/bin/activate
  ```
### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
This project uses the MIT-BIH Arrhythmia Database from PhysioNet. Download the dataset from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/). After downloading, place the dataset inside the project directory:
```text
data/
└── mit-bih-arrhythmia-database/
```
### Running the Notebooks
You can run the experiments using Jupyter Notebook or Visual Studio Code (VS Code).
### 1. Using Jupyter Notebook
Install Jupyter Notebook if it is not already installed:
```bash
pip install notebook
```
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Your browser will open a local Jupyter interface. Navigate to the repository folder and open one of the experiment notebooks. Run the cells sequentially to reproduce the experiments.

### 2. Using Visual Studio Code
Open the repository folder in VS Code and open any .ipynb notebook file. Make sure the Python and Jupyter extensions are installed. Then run the notebook cells directly inside Visual Studio Code.

## Repository Structure
```text
Arrhythmia-Classification-in-ECG
│
├── data/
│   └── mit-bih-arrhythmia-database/     # MIT-BIH ECG dataset
│
├── notebooks/                           # Jupyter notebooks for experiments
│   ├── GRU_Variants_3R.ipynb
│   ├── GRU_Variants_5R.ipynb
│   ├── GRU_Variants_10R.ipynb
│   ├── BiGRU_Variants_3R.ipynb
│   ├── BiGRU_Variants_5R.ipynb
│   └── BiGRU_Variants_10R.ipynb
│
├── src/                                 # Custom GRU / BiGRU implementations
│
├── requirements.txt                     # Python dependencies
│
└── README.md                            # Project documentation
```

## Experiment Pipeline
Each notebook follows the same experimental workflow:
1. Load ECG recordings from the MIT-BIH Arrhythmia Database
2. Apply ECG preprocessing using Butterworth bandpass filtering (0.5–45 Hz) and Z-score normalization.
3. Detect R-peaks
4. Perform sliding window segmentation (3R, 5R, 10R)
5. Train GRU / BiGRU models
6. Evaluate model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC 

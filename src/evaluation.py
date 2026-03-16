import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay
from itertools import cycle
import os

def plot_roc_multiclass(y_true_enc, y_pred_probs, model_name="Model"):
    n_classes = y_true_enc.shape[1]
    fpr = dict(); tpr = dict(); roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_enc[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'teal'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_and_save_training_history(hist, model_name, output_dir):
    if not hist: return
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['accuracy'], label='Train')
    plt.plot(hist['val_accuracy'], label='Val')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Train')
    plt.plot(hist['val_loss'], label='Val')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_acc_loss.png"))
    plt.close()

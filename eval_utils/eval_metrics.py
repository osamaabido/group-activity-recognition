import numpy as np
from sklearn.metrics import accuracy_score , f1_score , precision_score , recall_score , classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime 
import seaborn as sns
import os
import torch

def get_f1_score(y_true, y_pred, average='weighted', report=False):
    if report:
        print("Classification Report:\n")
        print(classification_report(y_true, y_pred, zero_division=1))
    else:
        f1 = f1_score(y_true, y_pred, average=average)
        print(f"F1 Score: {f1:.4f}")
        return f1
    
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)

    return fig
    
def plot_learning_curves(train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, save_path):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Losses')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracies')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_f1, label='Train F1')
    plt.plot(val_f1, label='Val F1')
    plt.title('F1 Scores')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def save_classification_report(labels, predictions, class_names, save_path):
    """Generate and save classification report to a text file."""
    report = classification_report(labels, predictions, target_names=class_names)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f'Classification report saved at {save_path}')

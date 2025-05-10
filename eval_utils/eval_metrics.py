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

def compute_metrics(labels, predictions):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_micro': f1_score(labels, predictions, average='micro'),
    }
    return metrics
    


def model_eval(model , data_loader , criterion = None ,device =None ,  path ="" , classes_name = None ):
    model.eval()
    y_true=[]
    y_pred=[]
    loss = 0.0
    with torch.no_grad():
        for inuputs , targets in data_loader:
            inputs , targets = inputs.to(device) , targets.to(device)
            outputs = model(inputs)
            if criterion:
                loss += criterion(outputs , targets).item()
            _, predicted = outputs.max(1)
            _, target_class = targets.max(1)
            y_true.extend(target_class.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    report_dict = classification_report(y_true, y_pred, target_names=classes_name, output_dict=True)
    if isinstance(report_dict, dict):
        accuracy = report_dict["accuracy"] * 100
 
    avg_loss = loss / len(data_loader) if criterion else None
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\n" + "=" * 50)
    print("=" * 50)
    print(f"Accuracy : {accuracy:.2f}%")
    if criterion:
        print(f"Average Loss: {avg_loss:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes_name))

    if classes_name:
        save_path = f"{path}/Group Activity Test Set Classification Report_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, class_names=classes_name, save_path=save_path)
    
    metrics = {
        "accuracy": accuracy,
        "avg_loss": avg_loss,
        "f1_score": f1,
        "classification_report": report_dict,
    }
    return metrics
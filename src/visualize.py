"""Visualization utilities for training and evaluation."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          save_path="models/confusion_matrix.png"):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         save_path="models/training_curves.png"):
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Training curves saved → {save_path}")


def plot_feature_importance(model, feature_names=None,
                            save_path="models/feature_importance.png"):
    """Plot weight magnitudes of first layer as feature importance."""
    first_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first_layer = module
            break

    if first_layer is None:
        return

    import torch
    weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(weights)), weights, alpha=0.7)
    ax.set_xlabel('Feature Index (landmark * 3 + coord)')
    ax.set_ylabel('Mean |Weight|')
    ax.set_title('Feature Importance (First Layer Weights)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🔍 Feature importance saved → {save_path}")

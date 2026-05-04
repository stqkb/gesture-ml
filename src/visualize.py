"""Visualization utilities."""
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="models/confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved -> {save_path}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="models/training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses)+1)
    ax1.plot(epochs, train_losses, 'b-', label='Train'); ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_accs, 'b-', label='Train'); ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Training curves saved -> {save_path}")
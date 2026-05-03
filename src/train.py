"""Training engine for gesture recognition model."""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.data import get_dataloaders
from src.model import build_model
from src.utils import load_config, set_seed, get_device, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc=f"🚂 Epoch {epoch:03d}", file=sys.stdout, leave=False)
    for batch_idx, (X_batch, y_batch) in enumerate(pbar):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def train(config_path: str = "configs/config.yaml"):
    """Full training pipeline."""
    config = load_config(config_path)

    # Reproducibility
    set_seed(config['train']['seed'])
    device = get_device(config['train']['device'])
    print(f"🖥️  Device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Model
    model = build_model(config).to(device)
    print(f"🧠 Model: {count_parameters(model):,} trainable parameters")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    save_path = Path(config['output']['model_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_interval = config['output']['log_interval']

    print(f"\n🏋️ Training for {config['train']['epochs']} epochs...\n")
    start_time = time.time()

    for epoch in range(1, config['train']['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, log_interval
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        # Log
        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"  Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"LR: {lr_now:.6f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"  ✅ Saved best model → {save_path} (val_acc={val_acc:.3f})")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training completed in {elapsed:.1f}s")
    print(f"🏆 Best Val Accuracy: {best_val_acc:.3f}")

    # Final test evaluation
    print("\n🧪 Final test evaluation...")
    # Load best model
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"   Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")

    return best_val_acc, test_acc


if __name__ == "__main__":
    train()

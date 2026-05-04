"""Training engine for gesture recognition model."""

import sys, time, logging
import torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from src.data import get_dataloaders, normalize_landmarks
from src.model import build_model
from src.utils import load_config, set_seed, get_device, count_parameters, resolve_path

logger = logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}", file=sys.stdout, leave=False)
    for bi, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
        if (bi+1) % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")
    return total_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss/total, correct/total


def train(config_path="configs/config.yaml"):
    config = load_config(config_path)
    set_seed(config['train']['seed'])
    device = get_device(config['train']['device'])
    tl, vl, testl = get_dataloaders(config)

    if config['model'].get('name', 'mlp') == 'xgb':
        return train_xgb(config, tl, vl, testl)
    return train_pytorch(config, tl, vl, testl, device)


def train_pytorch(config, tl, vl, testl, device):
    model = build_model(config).to(device)
    logger.info(f"Model: {count_parameters(model):,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = config['train'].get('patience', 10)
    patience_counter = 0
    save_path = Path(resolve_path(config['output']['model_path']))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    li = config['output']['log_interval']
    epochs = config['train']['epochs']

    logger.info(f"Training for {epochs} epochs (early stopping patience={patience})...")
    t0 = time.time()

    for ep in range(1, epochs+1):
        tl_loss, tl_acc = train_one_epoch(model, tl, criterion, optimizer, device, ep, li)
        vl_loss, vl_acc = evaluate(model, vl, criterion, device)
        scheduler.step(vl_acc)

        lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Epoch {ep:03d} | Train Loss: {tl_loss:.4f} Acc: {tl_acc:.3f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.3f} | LR: {lr:.6f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'val_acc': vl_acc, 'val_loss': vl_loss}, save_path)
            logger.info(f"  Saved best model -> {save_path} (val_acc={vl_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {ep}")
                break

    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed:.1f}s, best val_acc={best_val_acc:.3f}")

    logger.info("Final test evaluation...")
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    te_loss, te_acc = evaluate(model, testl, criterion, device)
    logger.info(f"  Test Loss: {te_loss:.4f} | Test Accuracy: {te_acc:.3f}")

    export_onnx(model, config)
    return best_val_acc, te_acc


def train_xgb(config, tl, vl, testl):
    import numpy as np
    model = build_model(config)
    logger.info("Training XGBoost model...")

    Xtr = np.concatenate([X.numpy() for X, _ in tl])
    ytr = np.concatenate([y.numpy() for _, y in tl])
    Xv = np.concatenate([X.numpy() for X, _ in vl])
    yv = np.concatenate([y.numpy() for _, y in vl])
    Xte = np.concatenate([X.numpy() for X, _ in testl])
    yte = np.concatenate([y.numpy() for _, y in testl])

    t0 = time.time()
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    logger.info(f"XGBoost done in {time.time()-t0:.1f}s")

    va = model.score(Xv, yv)
    te = model.score(Xte, yte)
    logger.info(f"  Val Acc: {va:.3f} | Test Acc: {te:.3f}")

    import pickle
    sp = Path(resolve_path(config['output']['model_path'])).with_suffix('.xgb.pkl')
    with open(sp, 'wb') as f:
        pickle.dump({'model': model, 'config': config}, f)
    logger.info(f"  Saved -> {sp}")
    return va, te


def export_onnx(model, config):
    try:
        p = Path(resolve_path(config['output']['model_path'])).with_suffix('.onnx')
        dummy = torch.randn(1, config['model']['input_dim'])
        model.cpu().eval()
        torch.onnx.export(model, dummy, str(p), input_names=["landmarks"], output_names=["logits"], dynamic_axes={"landmarks": {0: "batch"}}, opset_version=17)
        logger.info(f"ONNX exported -> {p}")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")


def main():
    train()

if __name__ == "__main__":
    main()
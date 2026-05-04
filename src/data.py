"""Data loading and preprocessing for gesture recognition."""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.utils import resolve_path

logger = logging.getLogger(__name__)


class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(labels)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_landmarks(features):
    single = features.ndim == 1
    if single:
        features = features.reshape(1, -1)
    result = features.copy()
    for i in range(len(result)):
        wrist = result[i, :3].copy()
        for j in range(21):
            result[i, j*3:(j+1)*3] -= wrist
        max_dist = np.max(np.linalg.norm(result[i].reshape(21, 3), axis=1))
        if max_dist > 0:
            result[i] /= max_dist
    return result.squeeze() if single else result


def augment_landmarks(features):
    augmented = features.copy()
    augmented *= np.random.uniform(0.9, 1.1)
    augmented += np.random.normal(0, 0.01, augmented.shape)
    if np.random.random() > 0.5:
        for i in range(21):
            augmented[i*3] = -augmented[i*3]
    if np.random.random() > 0.5:
        augmented = random_rotation(augmented)
    if np.random.random() > 0.5:
        augmented = random_translation(augmented)
    if np.random.random() > 0.3:
        augmented = random_occlusion(augmented)
    return augmented


def random_rotation(features, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    result = features.copy()
    for i in range(21):
        x, y = result[i*3], result[i*3+1]
        result[i*3] = cos_a*x - sin_a*y
        result[i*3+1] = sin_a*x + cos_a*y
    return result


def random_translation(features, max_shift=0.05):
    shift = np.random.uniform(-max_shift, max_shift, size=3)
    result = features.copy()
    for i in range(21):
        result[i*3:(i+1)*3] += shift
    return result


def random_occlusion(features, prob=0.1):
    result = features.copy()
    for i in range(21):
        if np.random.random() < prob:
            result[i*3:(i+1)*3] = 0.0
    return result


def generate_demo_data(n_samples=2000, n_features=63, n_classes=10, seed=42):
    np.random.seed(seed)
    X_all, y_all = [], []
    spc = n_samples // n_classes
    for digit in range(n_classes):
        base = np.random.randn(n_features) * 0.3
        base[digit*6:(digit+1)*6] += 2.0
        for _ in range(spc):
            X_all.append(base + np.random.randn(n_features)*0.15)
            y_all.append(digit)
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    idx = np.random.permutation(len(X_all))
    return X_all[idx], y_all[idx]


def load_collected_data(raw_path):
    X_all, y_all = [], []
    raw_dir = resolve_path(raw_path)
    for digit in range(10):
        d = raw_dir / str(digit)
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".npy"):
                X_all.append(np.load(d / f))
                y_all.append(digit)
    if not X_all:
        return None
    return np.array(X_all, dtype=np.float32), np.array(y_all, dtype=np.int64)


def get_dataloaders(config):
    pp = resolve_path(config['data']['processed_path'])
    ff, lf = pp/"features.npy", pp/"labels.npy"

    collected = load_collected_data(config['data']['raw_path'])
    if collected is not None:
        X, y = collected
        logger.info(f"Loaded {len(X)} real collected samples")
        pp.mkdir(parents=True, exist_ok=True)
        np.save(ff, X); np.save(lf, y)
    elif ff.exists() and lf.exists():
        X, y = np.load(ff), np.load(lf)
        logger.info(f"Loaded {len(X)} processed samples")
    else:
        logger.info("No real data found, generating demo dataset...")
        X, y = generate_demo_data(n_features=config['model']['input_dim'], n_classes=config['model']['num_classes'])
        pp.mkdir(parents=True, exist_ok=True)
        np.save(ff, X); np.save(lf, y)

    X = normalize_landmarks(X)
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=y)
    vr = config['data']['val_size'] / (1 - config['data']['test_size'])
    Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=vr, random_state=config['data']['random_state'], stratify=yt)

    logger.info(f"Data split: train={len(Xtr)} | val={len(Xv)} | test={len(Xte)}")
    bs = config['train']['batch_size']
    return (
        DataLoader(GestureDataset(Xtr, ytr), batch_size=bs, shuffle=True),
        DataLoader(GestureDataset(Xv, yv), batch_size=bs),
        DataLoader(GestureDataset(Xte, yte), batch_size=bs),
    )
"""Data loading and preprocessing for gesture recognition."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class GestureDataset(Dataset):
    """Dataset for hand gesture landmarks."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.FloatTensor(features)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def normalize_landmarks(features: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks: center at wrist (point 0) and scale to unit range.
    features: (N, 63) or (63,)
    """
    single = features.ndim == 1
    if single:
        features = features.reshape(1, -1)

    result = features.copy()
    for i in range(len(result)):
        # Center at wrist (first landmark)
        wrist = result[i, :3].copy()
        for j in range(21):
            result[i, j*3:(j+1)*3] -= wrist
        # Scale to max distance from wrist = 1
        max_dist = np.max(np.linalg.norm(result[i].reshape(21, 3), axis=1))
        if max_dist > 0:
            result[i] /= max_dist

    return result.squeeze() if single else result


def augment_landmarks(features: np.ndarray) -> np.ndarray:
    """Apply random augmentation to landmark features."""
    augmented = features.copy()

    # Random scale (0.9 ~ 1.1)
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale

    # Random noise (small jitter)
    noise = np.random.normal(0, 0.01, augmented.shape)
    augmented += noise

    # Random horizontal flip (swap left/right hand features)
    if np.random.random() > 0.5:
        for i in range(21):
            augmented[i * 3] = -augmented[i * 3]  # flip x

    return augmented


def generate_demo_data(n_samples: int = 2000, n_features: int = 63,
                       n_classes: int = 10, seed: int = 42) -> tuple:
    """
    Generate synthetic gesture data for demo purposes.
    Each digit gets a distinct pattern in the landmark space.
    """
    np.random.seed(seed)
    X_all, y_all = [], []

    samples_per_class = n_samples // n_classes

    for digit in range(n_classes):
        # Create a distinct base pattern for each digit
        base = np.random.randn(n_features) * 0.3
        # Add digit-specific offset to distinguish classes
        base[digit * 6:(digit + 1) * 6] += 2.0

        for _ in range(samples_per_class):
            sample = base + np.random.randn(n_features) * 0.15
            X_all.append(sample)
            y_all.append(digit)

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X_all))
    return X_all[idx], y_all[idx]


def get_dataloaders(config: dict) -> tuple:
    """
    Load data and return train/val/test DataLoaders.
    If no real data exists, generates demo data.
    """
    import os

    processed_path = config['data']['processed_path']
    features_file = os.path.join(processed_path, "features.npy")
    labels_file = os.path.join(processed_path, "labels.npy")

    if os.path.exists(features_file) and os.path.exists(labels_file):
        # Load real data
        X = np.load(features_file)
        y = np.load(labels_file)
    else:
        # Generate demo data
        print("⚡ No real data found, generating demo dataset...")
        X, y = generate_demo_data(
            n_features=config['model']['input_dim'],
            n_classes=config['model']['num_classes']
        )
        os.makedirs(processed_path, exist_ok=True)
        np.save(features_file, X)
        np.save(labels_file, y)
        print(f"   Saved demo data to {processed_path}/ ({len(X)} samples)")

    # Normalize
    X = normalize_landmarks(X)

    # Split: train / val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )

    val_ratio = config['data']['val_size'] / (1 - config['data']['test_size'])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=config['data']['random_state'],
        stratify=y_trainval
    )

    print(f"📊 Data split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

    bs = config['train']['batch_size']
    return (
        DataLoader(GestureDataset(X_train, y_train), batch_size=bs, shuffle=True),
        DataLoader(GestureDataset(X_val, y_val), batch_size=bs),
        DataLoader(GestureDataset(X_test, y_test), batch_size=bs),
    )

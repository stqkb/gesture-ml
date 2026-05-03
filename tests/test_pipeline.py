"""Tests for the gesture recognition pipeline."""

import pytest
import numpy as np
import torch


def test_dataset_creation():
    """Test GestureDataset can be created and indexed."""
    from src.data import GestureDataset

    X = np.random.randn(100, 63).astype(np.float32)
    y = np.random.randint(0, 10, 100).astype(np.int64)
    ds = GestureDataset(X, y)

    assert len(ds) == 100
    x_sample, y_sample = ds[0]
    assert x_sample.shape == (63,)
    assert isinstance(y_sample.item(), int)


def test_normalize_landmarks():
    """Test landmark normalization centers at wrist."""
    from src.data import normalize_landmarks

    features = np.random.randn(63).astype(np.float32)
    normalized = normalize_landmarks(features)

    # After normalization, first landmark should be at origin
    assert np.allclose(normalized[:3], 0.0, atol=1e-6)
    assert normalized.shape == (63,)


def test_normalize_batch():
    """Test normalization works on batches."""
    from src.data import normalize_landmarks

    batch = np.random.randn(10, 63).astype(np.float32)
    normalized = normalize_landmarks(batch)
    assert normalized.shape == (10, 63)


def test_augment_landmarks():
    """Test augmentation returns correct shape."""
    from src.data import augment_landmarks

    features = np.random.randn(63).astype(np.float32)
    augmented = augment_landmarks(features)
    assert augmented.shape == features.shape


def test_generate_demo_data():
    """Test demo data generation."""
    from src.data import generate_demo_data

    X, y = generate_demo_data(n_samples=100, n_classes=10)
    assert X.shape == (100, 63)
    assert y.shape == (100,)
    assert set(y) == set(range(10))


def test_model_forward():
    """Test model forward pass."""
    from src.model import GestureMLP

    model = GestureMLP(input_dim=63, hidden_dims=[64, 32], num_classes=10)
    x = torch.randn(4, 63)
    out = model(x)

    assert out.shape == (4, 10)


def test_model_build():
    """Test model building from config."""
    from src.model import build_model

    config = {
        'model': {
            'input_dim': 63,
            'hidden_dims': [128, 64],
            'num_classes': 10,
            'dropout': 0.3,
        }
    }
    model = build_model(config)
    x = torch.randn(2, 63)
    assert model(x).shape == (2, 10)


def test_predictor_init():
    """Test predictor initialization."""
    from src.utils import load_config
    from src.model import build_model

    # Build and save a model
    config = {
        'model': {
            'input_dim': 63,
            'hidden_dims': [64, 32],
            'num_classes': 10,
            'dropout': 0.3,
        },
        'data': {'test_size': 0.2, 'random_state': 42},
        'train': {'batch_size': 32, 'seed': 42, 'device': 'cpu'},
    }
    model = build_model(config)

    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "test_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_acc': 0.95,
    }, path)

    from src.predict import GesturePredictor
    predictor = GesturePredictor(model_path=path, device="cpu")
    result = predictor.predict(np.random.randn(63).astype(np.float32))
    assert 0 <= result <= 9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

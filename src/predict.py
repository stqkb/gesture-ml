"""Inference / prediction module for gesture recognition."""

import torch
import numpy as np
from src.model import build_model
from src.data import normalize_landmarks
from src.utils import load_config, get_device


class GesturePredictor:
    """Load a trained model and run predictions."""

    def __init__(self, model_path: str = "models/best_model.pt",
                 device: str = "cpu"):
        self.device = get_device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        self.config = checkpoint['config']
        self.val_acc = checkpoint.get('val_acc', None)

        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.class_names = [str(i) for i in range(self.config['model']['num_classes'])]

        if self.val_acc:
            print(f"📦 Model loaded (val_acc={self.val_acc:.3f}) on {self.device}")

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> int:
        """Predict digit from raw landmark features (63-dim array)."""
        features = normalize_landmarks(np.array(features, dtype=np.float32))
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        return self.model(x).argmax(1).item()

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> dict:
        """Predict with probability distribution over all classes."""
        features = normalize_landmarks(np.array(features, dtype=np.float32))
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1).squeeze()
        return {self.class_names[i]: probs[i].item()
                for i in range(len(probs))}

    @torch.no_grad()
    def predict_batch(self, features_batch: np.ndarray) -> list:
        """Predict a batch of samples."""
        features_batch = normalize_landmarks(
            np.array(features_batch, dtype=np.float32)
        )
        x = torch.FloatTensor(features_batch).to(self.device)
        preds = self.model(x).argmax(1)
        return preds.cpu().tolist()


def demo():
    """Quick demo of the predictor."""
    predictor = GesturePredictor()

    # Fake 63-dim landmark vector
    dummy = np.random.randn(63).astype(np.float32)
    digit = predictor.predict(dummy)
    proba = predictor.predict_proba(dummy)

    print(f"\n🔮 Predicted digit: {digit}")
    print("📊 Class probabilities:")
    for cls, prob in sorted(proba.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {cls}: {bar} {prob:.3f}")


if __name__ == "__main__":
    demo()

"""Inference / prediction module for gesture recognition."""

import logging, pickle
import torch, numpy as np
from pathlib import Path
from src.model import build_model, XGBPredictor
from src.data import normalize_landmarks
from src.utils import load_config, get_device, resolve_path

logger = logging.getLogger(__name__)


class GesturePredictor:
    def __init__(self, model_path="models/best_model.pt", device="cpu"):
        self.device = get_device(device)
        resolved = resolve_path(model_path)
        xgb_path = resolved.with_suffix('.xgb.pkl')

        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                data = pickle.load(f)
            self.config = data['config']
            self.model = XGBPredictor(data['model'], self.config)
            self.val_acc = None
            self.is_xgb = True
            logger.info("XGBoost model loaded")
        elif resolved.exists():
            ckpt = torch.load(str(resolved), map_location=self.device, weights_only=False)
            self.config = ckpt['config']
            self.val_acc = ckpt.get('val_acc', None)
            self.model = build_model(self.config).to(self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.is_xgb = False
            if self.val_acc:
                logger.info(f"PyTorch model loaded (val_acc={self.val_acc:.3f}) on {self.device}")
        else:
            raise FileNotFoundError(f"Model not found: {resolved}. Run 'python -m src.train' first.")

        self.class_names = [str(i) for i in range(self.config['model']['num_classes'])]

    def _prep(self, features):
        f = np.array(features, dtype=np.float32)
        f = np.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0)
        f = np.clip(f, -10, 10)
        return normalize_landmarks(f)

    @torch.no_grad()
    def predict(self, features):
        f = self._prep(features)
        if self.is_xgb:
            return self.model.predict(f)
        x = torch.FloatTensor(f).unsqueeze(0).to(self.device)
        return self.model(x).argmax(1).item()

    @torch.no_grad()
    def predict_proba(self, features):
        f = self._prep(features)
        if self.is_xgb:
            return self.model.predict_proba(f)
        x = torch.FloatTensor(f).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1).squeeze()
        return {self.class_names[i]: probs[i].item() for i in range(len(probs))}

    @torch.no_grad()
    def predict_batch(self, features_batch):
        fb = np.array(features_batch, dtype=np.float32)
        fb = np.nan_to_num(fb, nan=0.0, posinf=10.0, neginf=-10.0)
        fb = normalize_landmarks(fb)
        if self.is_xgb:
            return self.model.predict_batch(fb)
        x = torch.FloatTensor(fb).to(self.device)
        return self.model(x).argmax(1).cpu().tolist()


def demo():
    predictor = GesturePredictor()
    dummy = np.random.randn(63).astype(np.float32)
    digit = predictor.predict(dummy)
    proba = predictor.predict_proba(dummy)
    logger.info(f"Predicted digit: {digit}")
    for cls, prob in sorted(proba.items(), key=lambda x: -x[1]):
        logger.info(f"   {cls}: {'#'*int(prob*30)} {prob:.3f}")

if __name__ == "__main__":
    demo()
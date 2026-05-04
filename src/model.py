"""Model definitions for gesture recognition."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GestureMLP(nn.Module):
    def __init__(self, input_dim=63, hidden_dims=None, num_classes=10, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_model(config):
    name = config['model'].get('name', 'mlp')
    if name == 'mlp':
        mc = config['model']
        return GestureMLP(input_dim=mc['input_dim'], hidden_dims=mc['hidden_dims'], num_classes=mc['num_classes'], dropout=mc['dropout'])
    elif name == 'xgb':
        return build_xgb_model(config)
    raise ValueError(f"Unknown model: {name}")


def build_xgb_model(config):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, objective='multi:softprob', num_class=config['model']['num_classes'], eval_metric='mlogloss', random_state=config['train']['seed'], verbosity=0)


class XGBPredictor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.class_names = [str(i) for i in range(config['model']['num_classes'])]

    def predict(self, features):
        return int(self.model.predict(features.reshape(1, -1))[0])

    def predict_proba(self, features):
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        return {self.class_names[i]: float(probs[i]) for i in range(len(probs))}

    def predict_batch(self, features_batch):
        return [int(x) for x in self.model.predict(features_batch)]
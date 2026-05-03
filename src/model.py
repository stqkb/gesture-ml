"""Model definitions for gesture recognition."""

import torch
import torch.nn as nn


class GestureMLP(nn.Module):
    """Multi-layer perceptron for gesture classification."""

    def __init__(self, input_dim: int = 63, hidden_dims: list = None,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def build_model(config: dict) -> nn.Module:
    """Build model from config dict."""
    model_cfg = config['model']
    return GestureMLP(
        input_dim=model_cfg['input_dim'],
        hidden_dims=model_cfg['hidden_dims'],
        num_classes=model_cfg['num_classes'],
        dropout=model_cfg['dropout'],
    )

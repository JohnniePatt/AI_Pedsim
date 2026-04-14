import torch
from torch import nn


class TimeEstimatorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=3, dropout=0.1):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_model(input_dim, config):
    model_cfg = config.get("model", {})
    return TimeEstimatorMLP(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_dim=len(config["features"]["target"]),
    )


def choose_device(config):
    requested = str(config.get("train", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)

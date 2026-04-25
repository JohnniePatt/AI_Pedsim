import torch
from torch import nn

# ---------------------------------------------------------------------------
# GCN Layers
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Simple GCN layer: D^-0.5 * A * D^-0.5 * X * W"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [Batch, Nodes, Features]
        # adj: [Batch, Nodes, Nodes]
        support = torch.matmul(adj, x)
        output = self.linear(support)
        return output

# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class TimeEstimatorGNN(nn.Module):
    def __init__(self, in_features, hidden_dims=[64, 32], output_dim=3, dropout=0.1):
        super().__init__()
        
        layers = []
        curr_in = in_features
        for h_dim in hidden_dims:
            layers.append(GCNLayer(curr_in, h_dim))
            curr_in = h_dim
        
        self.gcn_layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(curr_in, output_dim)

    def forward(self, x, adj):
        for layer in self.gcn_layers:
            x = layer(x, adj)
            x = self.relu(x)
            x = self.dropout(x)
            
        # Global Mean Pooling
        x = torch.mean(x, dim=1)
        return self.predictor(x)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(input_dim, config):
    model_cfg = config.get("model", {})
    return TimeEstimatorGNN(
        in_features=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [64, 32]),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_dim=len(config["features"]["target"])
    )


def is_mps_available():
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def choose_device(config):
    requested = str(config.get("train", {}).get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested == "mps" and not is_mps_available():
        return torch.device("cpu")
    return torch.device(requested)

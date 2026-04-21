import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class GNNDataset(Dataset):
    def __init__(self, dataframe, geo_root, feature_columns, target_columns, scaler=None):
        self.df = dataframe.reset_index(drop=True)
        self.geo_root = Path(geo_root)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        # Pre-load graphs
        self.plans_data = {}
        unique_plans = self.df["plan"].unique()
        for plan in unique_plans:
            self.plans_data[plan] = self._load_plan_graph(plan)
            
        # Targets
        targets = self.df[target_columns].astype(float).to_numpy(dtype=np.float32)
        self.y = np.log1p(targets)
        
        if scaler:
            self.y = (self.y - scaler["target_mean"]) / scaler["target_std"]

    def _load_plan_graph(self, plan_name):
        graph_path = self.geo_root / plan_name / "topological_graph.json"
        if not graph_path.exists():
            return None
        
        data = read_json(graph_path)
        nodes = data["nodes"]
        edges = data["edges"]
        num_nodes = len(nodes)
        
        # Adjacency Matrix (Normalized)
        adj = np.eye(num_nodes, dtype=np.float32)
        for edge in edges:
            u, v = edge["from"], edge["to"]
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_norm = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).astype(np.float32)
        
        # Node Features
        node_features = []
        for node in nodes:
            area = float(node.get("area_m2", 0.0))
            is_corridor = 1.0 if node.get("type") == "corridor" else 0.0
            node_features.append([area, is_corridor])
        
        return {
            "adj": adj_norm,
            "node_features": np.array(node_features, dtype=np.float32),
            "node_map": {node["name"]: i for i, node in enumerate(nodes)}
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        plan_name = row["plan"]
        graph = self.plans_data[plan_name]
        
        start_node = row["start_node"]
        end_node = row["end_node"]
        agents = float(row.get("computed_agents", 1.0))
        dist_straight = float(row.get("straight_distance_m", 0.0))
        dist_topo = float(row.get("topology_centerline_distance_m", 0.0))
        
        # Bottleneck features
        min_door_w = float(row.get("min_door_width_between_A_B", 1.0))
        door_count = float(row.get("door_count_between_A_B", 0.0))
        bottleneck = float(row.get("bottleneck_score", 0.0))
        
        base_features = graph["node_features"]
        num_nodes = base_features.shape[0]
        
        # Dynamic features: [IsStart, IsEnd, Agents, DistStraight, DistTopo, DoorWidth, DoorCount, Bottleneck]
        route_features = np.zeros((num_nodes, 8), dtype=np.float32)
        node_map = graph["node_map"]
        
        if start_node in node_map:
            route_features[node_map[start_node], 0] = 1.0
        if end_node in node_map:
            route_features[node_map[end_node], 1] = 1.0
            
        route_features[:, 2] = agents / 100.0
        route_features[:, 3] = dist_straight / 100.0
        route_features[:, 4] = dist_topo / 100.0
        route_features[:, 5] = min_door_w        # Physical unit ~1.0
        route_features[:, 6] = door_count / 10.0 # Scaling
        route_features[:, 7] = bottleneck        # Already roughly a score
        
        full_x = np.concatenate([base_features, route_features], axis=1)
        
        return {
            "x": torch.from_numpy(full_x),
            "adj": torch.from_numpy(graph["adj"]),
            "y": torch.from_numpy(self.y[idx])
        }

# ---------------------------------------------------------------------------
# Bundle Factory
# ---------------------------------------------------------------------------

def build_gnn_data_bundle(config, config_path):
    import sys
    sys.path.append(str(Path(config_path).parent.parent / "Method_MLP_PyTorch"))
    from dataset import load_formatted_dataframes
    
    frames, _ = load_formatted_dataframes(config, config_path)
    target_columns = config["features"]["target"]
    
    train_targets = np.log1p(frames["train"][target_columns].values)
    scaler = {
        "target_mean": train_targets.mean(axis=0),
        "target_std": np.maximum(train_targets.std(axis=0), 1e-6)
    }
    
    geo_root = Path(config_path).parent / config["data"]["geo_root"]
    
    train_ds = GNNDataset(frames["train"], geo_root, None, target_columns, scaler)
    val_ds = GNNDataset(frames["val"], geo_root, None, target_columns, scaler)
    test_ds = GNNDataset(frames["test"], geo_root, None, target_columns, scaler)
    
    return train_ds, val_ds, test_ds, scaler

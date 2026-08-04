"""Generate honest, normalized gallery previews from each method's own inference.

These artifacts are for UI framing only.  A manifest beside every image records
checkpoint/knowledge provenance and whether retraining is required before the
artifact can be used as a research result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point, Polygon


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "AI_GenerateTimeseries/AI_Train"
RESULT_ROOT = ROOT / "AI_GenerateTimeseries/AI_Result"
DATASET_ROOT = ROOT / "Dataset/Data_Traj_Table/Topo_HouseGAN"
sys.path.insert(0, str(TRAIN_ROOT))
from normalized_preview import load_walkable, plot_normalized_rollout  # noqa: E402


CASES = {
    "plan_110_fbd0": DATASET_ROOT / "train/case_plan_110_fbd0_42_00_full",
    "plan_102_8e0f": DATASET_ROOT / "train/case_plan_102_8e0f_42_00_full",
    "plan_110_fbd0_half": DATASET_ROOT / "train/case_plan_110_fbd0_100044_02_half",
}


OUTPUTS = {
    "GNN-CVAE": RESULT_ROOT / "Method_GNN_CVAE/outputs/run_6_evaluate/framing_previews",
    "Social GAN": RESULT_ROOT / "Method_SGAN/outputs/run_6_evaluate/framing_previews",
    "LSTM": RESULT_ROOT / "Method_LSTM_01/outputs/run_LSTM_20260327_184506_evaluate/framing_previews",
    "GPT+RAG": RESULT_ROOT / "Method_GPT_Knowledge/outputs/run_gpt_knowledge_evaluate/framing_previews",
}


@contextmanager
def method_imports(method_dir: pathlib.Path):
    """Temporarily expose a method directory with its legacy flat imports."""
    old_path = list(sys.path)
    conflicting = {name: sys.modules.pop(name) for name in ("model", "dataset") if name in sys.modules}
    sys.path.insert(0, str(method_dir))
    try:
        yield
    finally:
        for name in ("model", "dataset"):
            sys.modules.pop(name, None)
        sys.modules.update(conflicting)
        sys.path[:] = old_path


def output_path(method: str, key: str, case_dir: pathlib.Path) -> pathlib.Path:
    case_id = case_dir.name.removeprefix("case_")
    return OUTPUTS[method] / key / "test_results/predictions" / f"{case_id}_rollout_preview.png"


def write_manifest(method: str, details: dict, generated: list[pathlib.Path]):
    root = OUTPUTS[method]
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "purpose": "UI framing preview only",
        "research_valid": False,
        "contains_ground_truth_trajectory_lines": False,
        "renderer": "AI_GenerateTimeseries/AI_Train/normalized_preview.py",
        "cases": [str(path) for path in generated],
        **details,
    }
    (root / "preview_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_gnn(device: torch.device):
    method_dir = TRAIN_ROOT / "Method_GNN_CVAE"
    cfg_path = RESULT_ROOT / "Method_GNN_CVAE/outputs/run_6/config_train.json"
    checkpoint = RESULT_ROOT / "Method_GNN_CVAE/outputs/run_6/weights/best_model.pth"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    generated = []
    with method_imports(method_dir):
        dataset_mod = importlib.import_module("dataset")
        model_mod = importlib.import_module("model")
        prepare_mod = importlib.import_module("prepare_geometry_gnn_cvae")
        model = model_mod.GNNCVAE(
            hidden_dim=cfg.get("hidden_dim", 128),
            latent_dim=cfg.get("latent_dim", 32),
            geo_dim=cfg.get("geo_dim", 64),
            gnn_layers=cfg.get("gnn_layers", 2),
            neighbor_radius=cfg.get("neighbor_radius", 0.08),
            dropout=cfg.get("dropout", 0.0),
            use_social=cfg.get("use_social", True),
            max_residual=cfg.get("max_residual", 0.15),
            segment_samples=cfg.get("segment_samples", 5),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()

        # Avoid loading the entire split when only three explicit framing cases are needed.
        ds = dataset_mod.SimulationTrajectoryDataset.__new__(dataset_mod.SimulationTrajectoryDataset)
        for name in ("obs_len", "frame_stride", "max_seq_len", "grid_size", "geo_padding", "max_agents"):
            setattr(ds, name, cfg.get(name, {"obs_len": 1, "frame_stride": 8, "max_seq_len": 160, "grid_size": 64, "geo_padding": 1.0, "max_agents": 64}[name]))
        ds._geo_cache = {}

        for key, case_dir in CASES.items():
            sample = ds._load_case(str(case_dir))
            with torch.no_grad():
                result = model(
                    positions=sample["positions"].unsqueeze(0).to(device),
                    agent_mask=sample["agent_mask"].unsqueeze(0).to(device),
                    start_pt=sample["start_pt"].unsqueeze(0).to(device),
                    goal_pt=sample["goal_pt"].unsqueeze(0).to(device),
                    geo_mask=sample["geo_mask"].unsqueeze(0).to(device),
                    teacher_forcing=False,
                    sample_latent=False,
                    goal_weight=0.0,
                    kl_weight=0.0,
                )
            pred = result["positions"][0].cpu().numpy()
            mask = sample["agent_mask"].numpy()
            trajectories = []
            for agent_idx in range(pred.shape[0]):
                valid = mask[agent_idx]
                world = np.asarray([prepare_mod.grid_to_world(x, y, sample["meta"]) for x, y in pred[agent_idx, valid]])
                trajectories.append(world)
            target = output_path("GNN-CVAE", key, case_dir)
            plot_normalized_rollout(case_dir, trajectories, target, "GNN-CVAE")
            generated.append(target)
            print(f"[Framing] GNN-CVAE -> {target.relative_to(ROOT)}")

    write_manifest(
        "GNN-CVAE",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_dataset": "Topo_bottleneck",
            "preview_dataset": "Topo_HouseGAN/train",
            "dataset_match": False,
            "retrain_required": True,
        },
        generated,
    )


def load_spawn_points(case_dir: pathlib.Path, limit: int = 12) -> np.ndarray:
    spawn_path = next(case_dir.glob("Spawn_location_*.csv"))
    df = pd.read_csv(spawn_path).sort_values("id")
    return df[["pos_x", "pos_y"]].to_numpy(dtype=np.float32)[:limit]


def generate_sgan(device: torch.device):
    method_dir = TRAIN_ROOT / "Method_SGAN"
    checkpoint = RESULT_ROOT / "Method_SGAN/outputs/run_6/weights/sgan_ep10.pth"
    cfg = json.loads((method_dir / "config_train.json").read_text(encoding="utf-8"))
    generated = []
    with method_imports(method_dir):
        model_mod = importlib.import_module("model")
        model = model_mod.TrajectoryGenerator(
            emb_dim=cfg.get("emb_size", 64),
            h_dim=cfg.get("hidden_size", 128),
            pool_dim=cfg.get("social_pooling_size", 16),
            obs_len=cfg.get("obs_len", 8),
            pred_len=cfg.get("pred_len", 12),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        obs_len, chunks = int(cfg.get("obs_len", 8)), 14

        for key, case_dir in CASES.items():
            starts = load_spawn_points(case_dir)
            n_agents = len(starts)
            current = torch.as_tensor(starts, device=device)
            rel_history = torch.zeros(obs_len, n_agents, 2, device=device)
            trajectories = [[point.copy()] for point in starts]
            with torch.no_grad():
                for _ in range(chunks):
                    pred_rel = model(rel_history, [(0, n_agents)])
                    pred_abs = current.unsqueeze(0) + torch.cumsum(pred_rel, dim=0)
                    for agent_idx in range(n_agents):
                        trajectories[agent_idx].extend(pred_abs[:, agent_idx].cpu().numpy())
                    current = pred_abs[-1]
                    rel_history = pred_rel[-obs_len:]
            target = output_path("Social GAN", key, case_dir)
            plot_normalized_rollout(case_dir, [np.asarray(points) for points in trajectories], target, "Social GAN")
            generated.append(target)
            print(f"[Framing] Social GAN -> {target.relative_to(ROOT)}")

    write_manifest(
        "Social GAN",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_dataset": "unverified (legacy run has no config snapshot)",
            "preview_dataset": "Topo_HouseGAN/train",
            "dataset_match": None,
            "retrain_required": True,
            "note": "Current SGAN dataset loader expects flat split/*.parquet but HouseGAN is nested split/case_*/*.parquet.",
        },
        generated,
    )


class LSTMBaseline(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(9, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def exit_centroid(case_dir: pathlib.Path):
    from shapely import wkt as shapely_wkt

    df = pd.read_csv(next(case_dir.glob("Spawn_exit_*.csv")))
    return shapely_wkt.loads(df[df["type"] == "exit_area"].iloc[0]["area"]).centroid


def generate_lstm(device: torch.device):
    run_dir = RESULT_ROOT / "Method_LSTM_01/run_LSTM_20260327_184506"
    checkpoint = run_dir / "checkpoints/generator_best.pth"
    cfg = json.loads((run_dir / "config_active.json").read_text(encoding="utf-8"))
    seq_len = int(cfg.get("seq_len", 200))
    model = LSTMBaseline(int(cfg.get("hidden_size", 256)), int(cfg.get("num_layers", 3)), 2 * int(cfg.get("predict_len", 5))).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    generated = []

    for key, case_dir in CASES.items():
        starts = load_spawn_points(case_dir)
        walkable = load_walkable(case_dir)
        goal = exit_centroid(case_dir)
        min_x, min_y, max_x, max_y = walkable.bounds
        width, height = max_x - min_x, max_y - min_y
        diag = float(np.hypot(width, height))
        history = np.repeat(starts[:, None, :], seq_len, axis=1)
        trajectories = [[point.copy()] for point in starts]

        with torch.no_grad():
            for _ in range(320):
                cx, cy = history[:, :, 0], history[:, :, 1]
                vx = np.diff(cx, axis=1, prepend=cx[:, :1])
                vy = np.diff(cy, axis=1, prepend=cy[:, :1])
                in_walkable = np.asarray([[float(walkable.covers(Point(x, y))) for x, y in row] for row in history])
                feats = np.stack(
                    [
                        (cx - min_x) / width,
                        (cy - min_y) / height,
                        vx,
                        vy,
                        (goal.x - cx) / width,
                        (goal.y - cy) / height,
                        np.hypot(goal.x - cx, goal.y - cy) / diag,
                        in_walkable,
                        np.zeros_like(in_walkable),
                    ],
                    axis=-1,
                ).astype(np.float32)
                delta = model(torch.from_numpy(feats).to(device)).cpu().numpy()[:, :2]
                next_pos = history[:, -1, :] + delta
                history = np.concatenate([history[:, 1:, :], next_pos[:, None, :]], axis=1)
                for agent_idx, point in enumerate(next_pos):
                    trajectories[agent_idx].append(point.copy())
        target = output_path("LSTM", key, case_dir)
        plot_normalized_rollout(case_dir, [np.asarray(points) for points in trajectories], target, "LSTM")
        generated.append(target)
        print(f"[Framing] LSTM -> {target.relative_to(ROOT)}")

    write_manifest(
        "LSTM",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_dataset": "Topo_bottleneck (hard-coded by legacy training script)",
            "preview_dataset": "Topo_HouseGAN/train",
            "dataset_match": False,
            "retrain_required": True,
        },
        generated,
    )


def generate_gpt_rag():
    method_dir = TRAIN_ROOT / "Method_GPT_Knowledge"
    knowledge_dir = RESULT_ROOT / "Method_GPT_Knowledge/knowledge/topo_bottleneck_v3"
    cfg = json.loads((method_dir / "config_test.json").read_text(encoding="utf-8"))
    generated = []
    with method_imports(method_dir):
        generate_mod = importlib.import_module("generate_gpt_knowledge")
        for key, case_dir in CASES.items():
            result = generate_mod.generate_case_prediction(case_dir, knowledge_dir, cfg)
            trajectories = [group[["pos_x", "pos_y"]].to_numpy() for _, group in result["prediction_df"].groupby("id")]
            target = output_path("GPT+RAG", key, case_dir)
            plot_normalized_rollout(case_dir, trajectories, target, "GPT+RAG")
            generated.append(target)
            print(f"[Framing] GPT+RAG -> {target.relative_to(ROOT)}")

    write_manifest(
        "GPT+RAG",
        {
            "knowledge_index": str(knowledge_dir),
            "knowledge_datasets": {"Topo_bottleneck": 613, "Topo_HouseGAN": 116},
            "preview_dataset": "Topo_HouseGAN/train",
            "retrain_required": False,
            "research_action_required": "Rebuild/freeze a leakage-safe knowledge index, then evaluate held-out HouseGAN test cases.",
        },
        generated,
    )


def main(methods: list[str]):
    missing = [str(path) for path in CASES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing target case directories: {missing}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Framing] Device: {device}")
    runners = {
        "gnn": lambda: generate_gnn(device),
        "sgan": lambda: generate_sgan(device),
        "lstm": lambda: generate_lstm(device),
        "gpt": generate_gpt_rag,
    }
    for method in methods:
        runners[method]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=("gnn", "sgan", "lstm", "gpt"), default=["gnn", "sgan", "lstm", "gpt"])
    args = parser.parse_args()
    main(args.methods)


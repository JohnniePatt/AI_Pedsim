import argparse
import json
import os
import pathlib
import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset, seq_collate
from model import TrajectoryGenerator

AI_TRAIN_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(AI_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(AI_TRAIN_DIR))
from baseline_output import (  # noqa: E402
    create_evaluation_layout,
    finalize_evaluation,
    resolve_checkpoint,
    write_case_prediction,
)

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_test.json")
    parser.add_argument("--run_path", type=str, required=True, help="Path to the training run directory")
    args = parser.parse_args()

    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Resolve dataset_path relative to config file if it's relative
    dataset_path = config["dataset_path"]
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(config_dir, dataset_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Testing SGAN on device: {device}")

    # Load dataset
    test_dataset = TrajectoryDataset(data_dir=dataset_path, config=config, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=seq_collate
    )

    # Initialize model
    model = TrajectoryGenerator(
        emb_dim=config.get("emb_size", 64),
        h_dim=config.get("hidden_size", 128),
        pool_dim=config.get("social_pooling_size", 16),
        obs_len=config["obs_len"],
        pred_len=config["pred_len"]
    ).to(device)

    # Load strongest/latest weights
    run_path = pathlib.Path(args.run_path).resolve()
    latest_weight = resolve_checkpoint(run_path)
    if latest_weight is None:
        legacy_weights = sorted((run_path / "weights").glob("*.pth"))
        latest_weight = legacy_weights[-1] if legacy_weights else None
    if latest_weight is None:
        raise FileNotFoundError("No checkpoint found in checkpoints/ or legacy weights/")
    print(f"📦 Loading weights from {latest_weight}...")
    saved = torch.load(latest_weight, map_location=device)
    checkpoint_data = saved.get("data_config", {}) if isinstance(saved, dict) else {}
    state_dict = saved.get("model_state_dict", saved) if isinstance(saved, dict) else saved
    model.load_state_dict(state_dict)
    model.eval()

    compatibility_ok = (
        str(checkpoint_data.get("dataset_name", "")).casefold()
        == pathlib.Path(dataset_path).name.casefold()
    )
    dataset_id = config.get("dataset_id", "housegan_canonical_imagebase_split_v1")
    dataset_manifest = pathlib.Path(dataset_path) / "manifest_housegan_cases.csv"
    eval_layout = create_evaluation_layout(
        run_path,
        method_id="Method_SGAN_SF_01",
        dataset_id=dataset_id,
        split="test",
        protocol_version=config.get("protocol_version", "v1"),
        checkpoint_path=latest_weight,
        evaluation_config=config,
        dataset_manifest=dataset_manifest if dataset_manifest.exists() else None,
        compatibility_ok=compatibility_ok,
        invalid_reason=(
            "legacy SGAN loader does not preserve canonical case_id/agent_id identity"
        ),
    )
    
    total_ade_running_sum, total_fde_running_sum = 0, 0
    total_peds_evaluated = 0
    samples_count = 0
    
    print("SGAN Generating test trajectories...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            obs_traj = batch[0].to(device) # (obs_len, batch_size, 2)
            pred_traj_gt = batch[1].to(device) # (pred_len, batch_size, 2)
            obs_rel_traj = batch[2].to(device)
            pred_rel_traj_gt = batch[3].to(device)
            seq_start_end = batch[4]
            
            # Predict
            pred_rel_traj = model(obs_rel_traj, seq_start_end)
            
            # Convert relative back to absolute
            # pred_traj = obs_traj[-1] + cumsum(pred_rel_traj)
            # (pred_len, batch_size, 2)
            last_obs = obs_traj[-1] # shape (batch_size, 2)
            pred_traj = torch.zeros_like(pred_traj_gt)
            
            curr = last_obs.clone()
            for t in range(pred_traj.shape[0]):
                curr = curr + pred_rel_traj[t]
                pred_traj[t] = curr

            # Calculate metrics (ADE & FDE)
            # pred_traj: (pred_len, total_batch_peds, 2)
            # pred_traj_gt: (pred_len, total_batch_peds, 2)
            
            # 1. ADE (Average Displacement Error) over all steps
            diff = pred_traj - pred_traj_gt
            dist_per_step = torch.norm(diff, dim=-1) # (pred_len, total_batch_peds)
            batch_ade = dist_per_step.mean(dim=0).sum().item() # Sum of ADEs for all peds in batch
            
            # 2. FDE (Final Displacement Error) at the last step
            final_diff = pred_traj[-1] - pred_traj_gt[-1]
            dist_final = torch.norm(final_diff, dim=-1) # (total_batch_peds)
            batch_fde = dist_final.sum().item()

            total_ade_running_sum += batch_ade
            total_fde_running_sum += batch_fde
            total_peds_evaluated += dist_final.shape[0]
            
            # Visualize Scene
            obs_traj_cpu = obs_traj.cpu().numpy()
            pred_gt_cpu = pred_traj_gt.cpu().numpy()
            pred_cpu = pred_traj.cpu().numpy()
            
            for (start, end) in seq_start_end:
                plt.figure(figsize=(8, 8))
                plt.title(f"SGAN Scene {samples_count} Trajectories")
                plt.xlabel("X")
                plt.ylabel("Y")
                
                # Plot every pedestrian in the scene
                for p_idx in range(start, end):
                    o_x = obs_traj_cpu[:, p_idx, 0]
                    o_y = obs_traj_cpu[:, p_idx, 1]
                    
                    p_x = pred_cpu[:, p_idx, 0]
                    p_y = pred_cpu[:, p_idx, 1]
                    
                    plt.plot(o_x, o_y, 'b-', label='Observed' if p_idx == start else "")
                    plt.plot(p_x, p_y, 'r--', label='SGAN Predicted' if p_idx == start else "")
                    
                    # Mark current pos
                    plt.scatter(o_x[-1], o_y[-1], c='blue', s=20)
                    
                plt.legend()
                plt.grid(True)
                scene_id = f"legacy_scene_{samples_count:06d}"
                preview_dir = eval_layout.previews / scene_id
                preview_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(preview_dir / "raw_rollout.png", dpi=150)
                plt.close()
                prediction_rows = []
                for local_agent, p_idx in enumerate(range(start, end)):
                    for step_idx in range(pred_cpu.shape[0]):
                        prediction_rows.append({
                            "case_id": scene_id,
                            "split": "test",
                            "frame": int(config["obs_len"] + step_idx),
                            "agent_id": local_agent,
                            "pos_x": float(pred_cpu[step_idx, p_idx, 0]),
                            "pos_y": float(pred_cpu[step_idx, p_idx, 1]),
                            "is_active": True,
                        })
                write_case_prediction(
                    eval_layout, scene_id, pd.DataFrame(prediction_rows), variant="raw"
                )
                samples_count += 1
                
    # Finish Evaluation Log
    avg_ade = total_ade_running_sum / total_peds_evaluated if total_peds_evaluated > 0 else 0
    avg_fde = total_fde_running_sum / total_peds_evaluated if total_peds_evaluated > 0 else 0
    
    eval_file = eval_layout.reports / "evaluation_summary.md"
    with open(eval_file, "w") as f:
        f.write("Evaluation Results (SGAN)\n")
        f.write("-------------------------\n")
        f.write(f"Evaluated Scenes: {samples_count}\n")
        f.write(f"Total Pedestrians: {total_peds_evaluated}\n")
        f.write(f"Average Displacement Error (ADE): {avg_ade:.4f}\n")
        f.write(f"Final Displacement Error (FDE):   {avg_fde:.4f}\n")
        f.write("Plots generated in samples/ directory.\n")
        
    # Save CSV for UI Dashboard
    import pandas as pd
    summary_df = pd.DataFrame([{
        "method_id": "Method_SGAN_SF_01", "variant": "raw", "seed": config.get("seed", 42),
        "ADE": avg_ade, "FDE": avg_fde, "constraint_intervention_rate": 0.0,
    }])
    summary_df.to_csv(eval_layout.metrics / "summary_metrics.csv", index=False)
    research_valid = finalize_evaluation(
        eval_layout,
        case_ids=[f"legacy_scene_{index:06d}" for index in range(samples_count)],
        floorplan_ids=[],
        compatibility_ok=compatibility_ok,
        canonical_test_required=True,
        additional_failures=[
            "legacy SGAN loader does not preserve canonical case_id/agent_id identity"
        ],
    )
        
    print(f"✅ Testing complete! Generated {samples_count} scene visualisations.")
    print(f"📊 ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")
    print(f"Research valid: {research_valid}")

if __name__ == "__main__":
    test()

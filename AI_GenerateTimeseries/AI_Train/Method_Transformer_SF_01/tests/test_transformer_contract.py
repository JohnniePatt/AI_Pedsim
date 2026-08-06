import pathlib
import sys
import tempfile
import unittest

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

METHOD_DIR = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(METHOD_DIR))

from model import GoalConditionedGPT2
from dataset import PedestrianDataset
from prepare_geometry_transformer import grid_to_world, world_to_grid
from test_transformer import assert_checkpoint_compatible, plot_full_case_rollout


class TransformerContractTests(unittest.TestCase):
    def test_coordinate_round_trip(self):
        meta = {"min_x": -6.0, "min_y": -5.0, "scale": 23.0, "grid_size": 64}
        world = (4.25, 14.75)
        norm = world_to_grid(*world, meta)
        restored = grid_to_world(*norm, meta)
        np.testing.assert_allclose(restored, world, atol=1e-7)

    def test_checkpoint_dataset_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "checkpoint/dataset mismatch"):
            assert_checkpoint_compatible(
                {"dataset_name": "Topo_bottleneck"},
                "/datasets/Topo_HouseGAN",
                allow_dataset_mismatch=False,
            )

    def test_delta_rollout_is_bounded_and_loss_is_finite(self):
        model = GoalConditionedGPT2(
            d_model=32,
            nhead=4,
            num_layers=1,
            max_seq_len=32,
            max_neighbors=2,
            obs_len=3,
            geo_encoder_type="spatial",
            prediction_mode="delta",
        )
        obs = torch.rand(2, 3, 2)
        labels = torch.rand(2, 5, 2)
        starts = obs[:, 0]
        ends = torch.rand(2, 2)
        geo = torch.ones(2, 1, 64, 64)
        neigh = torch.zeros(2, 2, 3, 2)
        neigh_mask = torch.zeros(2, 2, dtype=torch.bool)
        lengths = torch.tensor([5, 3])

        trained = model(
            obs, starts, ends, geo,
            neighbor_trajs=neigh,
            neighbor_mask=neigh_mask,
            labels=labels,
            lengths=lengths,
        )
        self.assertTrue(torch.isfinite(trained["loss"]))

        rolled = model(
            obs, starts, ends, geo,
            neighbor_trajs=neigh,
            neighbor_mask=neigh_mask,
            pred_len=5,
        )["logits"]
        self.assertTrue(bool(((rolled >= 0.0) & (rolled <= 1.0)).all()))

    def test_reported_housegan_case_normalises_inside_shared_frame(self):
        dataset_root = PROJECT_ROOT / "Dataset/Data_Traj_Table/Topo_HouseGAN"
        if not dataset_root.exists():
            self.skipTest("integration fixture is not available")
        dataset = PedestrianDataset(
            str(dataset_root),
            split="test",
            case_id="plan_44_fd18_100042_00_half",
            obs_len=5,
            frame_stride=8,
        )
        self.assertEqual(len(dataset), 6)
        for sample in dataset.samples:
            points = torch.cat([sample["obs_traj"], sample["pred_traj"]])
            self.assertTrue(bool(((points >= 0.0) & (points <= 1.0)).all()))

    def test_plot_limits_use_complete_floorplan(self):
        case_dir = (
            PROJECT_ROOT
            / "Dataset/Data_Traj_Table/Topo_HouseGAN/test"
            / "case_plan_44_fd18_100042_00_half"
        )
        if not case_dir.exists():
            self.skipTest("integration fixture is not available")

        # Mimic an early rollout that never reaches the lower exit room.
        obs = np.array([[5.0, 14.5], [5.0, 14.0]], dtype=np.float32)
        pred = np.array([[4.8, 12.0], [4.5, 8.0], [4.2, 5.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "preview.png"
            _, ylim = plot_full_case_rollout(case_dir, [(obs, pred)], output)
            self.assertTrue(output.exists())
            self.assertLess(ylim[0], -4.4)
            self.assertGreater(ylim[1], 16.1)


if __name__ == "__main__":
    unittest.main()

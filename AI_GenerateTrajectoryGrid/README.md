# AI_GenerateTrajectoryGrid

This workspace is for grid-based pedestrian trajectory generation.

The first phase focuses on preparing Topo_HouseGAN layouts as metric-aware
walkable grids. Model experiments will be added later under `AI_Train/`.

Structure:

- `AI_Train/` contains future grid trajectory training methods.
- `AI_Result/` contains future outputs, checkpoints, and evaluations.
- `Streamlit_ui/` contains utility tools for preparing grid data.

Current utility:

```bash
python Tool_utility/prepare_layout_to_grid_walkablearea.py --overwrite
```

This writes `walkablearea_grid.json` into each plan folder under:

```text
Geo_scenario/Topo_HouseGAN/geo/<plan_name>/
```

Trajectory grid dataset utility:

```bash
python Tool_utility/prepare_dataset_trajectory_grid.py --overwrite
```

Dataset spec/defaults:

```text
Tool_utility/prepare_dataset_trajectory_grid.json
```

This reads raw Topo_HouseGAN files from:

```text
Geo_scenario/Topo_HouseGAN/dataswarm/<plan_name>/*.sqlite
Geo_scenario/Topo_HouseGAN/geo/<plan_name>/walkablearea_grid.json
Geo_scenario/Topo_HouseGAN/metadata/<plan_name>/route_<route_index>.json
```

And writes A/B training data to:

```text
Dataset/Data_TrajectoryGrid/Topo_HouseGAN/
  A/<split>/<plan_name>/<sqlite_stem>/
    walkablearea_grid.json
    spawn_agent.parquet
    exit_room.json
  B/<split>/<plan_name>/<sqlite_stem>/
    trajectory.parquet
  metadata/<split>/<plan_name>/<sqlite_stem>.json
  manifest_trajectory_grid.csv
```

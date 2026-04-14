# GeneratePlan_HouseGAN

Standalone HouseGAN plan generation and JuPedSim simulation pipeline.

This module keeps the old HouseGAN/JuPedSim idea, but separates it from the older mixed directories so experiments do not overwrite each other.

## Directory Layout

- `Prepare_data/` generates HouseGAN-style layout geometry.
- `Simulation/` runs density-aware JuPedSim simulations.
- `Streamlit_ui/` provides a dedicated UI for generating, simulating, and viewing results.

## Outputs

New outputs are written to:

```text
Geo_scenario/Topo_HouseGAN/
  geo/
  dataswarm/
  trajectory_line/
  heatmap_density/
  heatmap_speed/
  spawn_exit/
  metadata/
  previews/
```

## Key Policies

- Walkable area is computed from rooms and corridors, then wall thickness is carved out.
- Door cutouts from `geo_door.json` are subtracted from walls, so agents can pass through doors.
- Spawn area is offset inward from walls before agents are distributed.
- Agent count is computed from usable spawn area, for example `safe_area / 2.0` means one person per 2 square meters.
- Each route writes metadata explaining the exact parameters and geometry decisions used.

## CLI

Generate plans:

```bash
python GeneratePlan_HouseGAN/Prepare_data/generate_layout.py --config GeneratePlan_HouseGAN/Prepare_data/config_housegan.json
```

Run density simulation for one plan:

```bash
python GeneratePlan_HouseGAN/Simulation/density_housegan_sim.py --plan plan_42_abcd --config GeneratePlan_HouseGAN/Simulation/config_density_sim.json
```

Run all unsimulated plans:

```bash
python GeneratePlan_HouseGAN/Simulation/density_housegan_sim.py --batch --config GeneratePlan_HouseGAN/Simulation/config_density_sim.json
```

Open UI:

```bash
streamlit run GeneratePlan_HouseGAN/Streamlit_ui/app.py
```

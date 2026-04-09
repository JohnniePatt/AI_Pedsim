# Method_GPT_Knowledge

This method does not fine-tune a trajectory network.

Instead it treats the train split as a structured knowledge base:
- geometry from `Geo_room.json` + `Geo_corridor.json`
- spawn positions per agent
- spawn area and exit area
- full frame-by-frame trajectories

The first baseline has three stages:

1. `build_knowledge.py`
- scans the train split
- extracts scene-level features
- saves a retrieval index that points back to the original cases

2. `generate_gpt_knowledge.py`
- receives a new scene
- retrieves similar train cases
- builds a planner prompt for a future GPT call
- generates a baseline trajectory set by transferring paths from the best-matching case

3. `validate_gpt_knowledge.py`
- runs the generator on a validation or test split
- reports path-shape ADE/FDE, duration error, collision rate, and out-of-bounds rate

The key idea is:
- GPT is better used as a high-level planner or reasoning layer
- deterministic geometry-aware transfer is better used for the exact `(x, y, t)` rollout

Future upgrades:
- add a true GPT planner that chooses among retrieved flow templates
- add rule-based social collision resolution after transfer
- add visual outputs for retrieved-vs-generated-vs-ground-truth comparison

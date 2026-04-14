# AI_Pedsim

Project-level data and tools live at the repository root:

- `Geo_scenario/` stores generated geometry and simulation outputs.
- `Dataset/Data_Traj_Table/` stores normalized trajectory training/test/validation datasets.
- `Dataset/Data_Estimate/` stores formatted Train/Val/Test data for travel-time estimation.
- `GeneratePlan_HouseGAN/` generates floor plans and simulation scenarios.
- `AI_GenerateTrajectory/` contains the older trajectory-generation AI workspace.

Trajectory AI training, testing, result browsing, and model outputs now live under
`AI_GenerateTrajectory/AI_Train`, `AI_GenerateTrajectory/AI_Result`, and
`AI_GenerateTrajectory/Streamlit_ui`.

Launch the trajectory dashboard with:

```bash
streamlit run AI_GenerateTrajectory/Streamlit_ui/app.py
```

# AI_Pedsim

**AI_Pedsim** is an AI surrogate modeling and pedestrian simulation research framework. The project evaluates whether deep learning models can learn pedestrian dynamics and predict time-series trajectory/density outputs as a fast AI surrogate, replacing traditional physics-based simulations (e.g., Social Force Model / JuPedSim).

---

## 🔬 AI Surrogate Models & Experiment Taxonomy

The repository categorizes AI surrogate models across **3 main output paradigms**:

### 1. Time-Series / Microscopic Sequential Trajectory Output
*Predicts continuous agent position time-series $(x_t, y_t)$ over sequence steps $t = 1 \dots T$, replacing step-by-step physical simulation.*

- **Transformer (Goal-Conditioned GPT-2)** ([AI_GenerateTrajectory/AI_Train/Method_Transformer](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_Transformer)): Primary method using causal GPT-2 backbone, CNN floorplan GeoEncoder, and neighbor context tokens for autoregressive trajectory generation.
- **GNN + CVAE** ([AI_GenerateTrajectory/AI_Train/Method_GNN_CVAE](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_GNN_CVAE) & [Method_GNN_CVAE2](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_GNN_CVAE2)): Graph Neural Network modeling social interactions between agents combined with CVAE latent space for multi-modal trajectory forecasting.
- **CVAE (Conditional VAE)** ([AI_GenerateTrajectory/AI_Train/Method_CVAE](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_CVAE)): Probabilistic trajectory generator conditioned on spawn/goal coordinates and environment density.
- **SGAN (Social GAN)** ([AI_GenerateTrajectory/AI_Train/Method_SGAN](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_SGAN)): Generative Adversarial Network for multi-agent interaction prediction over short-to-mid horizon time-series.
- **LSTM Baseline** ([AI_GenerateTrajectory/AI_Train/Method_LSTM_01](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_LSTM_01)): Recurrent baseline model predicting sequential agent movements step-by-step.
- **Deep Reinforcement Learning (Actor-Critic / PPO)** ([AI_GenerateTrajectory/AI_Train/Method_RL](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_RL)): Reinforcement learning agent interacting with a virtual environment (`vir_pedsim.py`) with shared LSTM policy.
- **Grid Social Policy** ([AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectoryGrid/AI_Train/Method_GridSocialPolicy)): Cell/Grid-based trajectory policy model.

### 2. Spatial-Temporal / Macroscopic Image-Based Heatmap Output
*Predicts spatial pedestrian density distribution / occupancy heatmaps as image outputs over time.*

- **Pix2PixHD / Pix2PixHD No-D** ([AI_GenerateTrajectory/AI_Train/Method_pix2pixHD](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_pix2pixHD) & [Method_pix2pixhd_No_D](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_pix2pixhd_No_D)): High-resolution Image-to-Image GAN translating floor plan and spawn/exit points directly to predicted density heatmaps.
- **UNet / Plain UNet / Unet-pix2pix** ([AI_GenerateTrajectory/AI_Train/Method_UNet](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_UNet), [Method_PlainUnet](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_PlainUnet), [Method_Unet-pix2pix](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_GenerateTrajectory/AI_Train/Method_Unet-pix2pix)): Image-to-image encoder-decoder networks mapping environment conditions to spatial trajectory density maps.

### 3. Aggregate Summary Output (Travel Time Estimation)
*Predicts overall scalar simulation summary metrics ($A \rightarrow B$ travel time) directly without step-by-step simulation.*

- **MLP (PyTorch & Keras)** ([AI_Estimate/AI_Train/Method_MLP_PyTorch](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_Estimate/AI_Train/Method_MLP_PyTorch), [Method_MLP_Keras](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_Estimate/AI_Train/Method_MLP_Keras)): Tabular regression predicting `min_agent_time_s`, `mean_agent_time_s`, and `max_agent_time_s`.
- **GNN Time Estimator** ([AI_Estimate/AI_Train/Method_GNN](file:///Ubuntu/home/johnnie/programming/AI_Pedsim/AI_Pedsim/AI_Estimate/AI_Train/Method_GNN)): Graph Convolutional Network predicting travel time based on floorplan topological graph (`topological_graph.json`).

---

## 📝 Research & Paper Framework

For research paper formulation comparing AI Surrogate Models vs. Physical Simulation:

- **Evaluation Metrics**:
  - Microscopic Trajectory: **ADE** (Average Displacement Error, in meters) and **FDE** (Final Displacement Error, in meters).
  - Travel Time Scalar: **MAE**, **RMSE**, and **$R^2$ Score**.
  - Computational Efficiency: **Inference Speedup Factor** (AI execution time vs. JuPedSim simulation time).
- **Paper Positioning Strategy**:
  1. Evaluate Transformer (GPT-2) against GNN-CVAE, SGAN, and LSTM baselines for continuous trajectory generation.
  2. Contrast Agent-based Trajectory Series (Microscopic) against Grid Density Heatmaps (Macroscopic) and Scalar Travel Time estimation (Aggregate).

---

## 📁 Repository Structure

```text
AI_Pedsim/
├── AI_GenerateTrajectory/   # Trajectory generation models, training, results & UI
│   ├── AI_Train/            # Training methods (Transformer, GNN_CVAE, SGAN, CVAE, UNet, etc.)
│   ├── AI_Result/           # Checkpoints, logs, and evaluation samples
│   └── Streamlit_ui/        # Dashboard for training and visualizing trajectories
├── AI_Estimate/            # Travel time estimation models (MLP, GNN)
├── AI_GenerateTrajectoryGrid/ # Grid-based trajectory policy models
├── GeneratePlan_HouseGAN/  # Procedural floor plan generator (HouseGAN workflow)
├── Geo_scenario/           # Raw geometry and SQLite simulation data
├── Dataset/                # Formatted Parquet & CSV training/testing datasets
└── Document_Research/      # Research notes, paper references, and framework summaries
```

---

## 🚀 Quick Start

Launch the Trajectory Dashboard:
```bash
streamlit run Streamlit_ui/app.py
```

Launch the Travel Time Estimation Dashboard:
```bash
streamlit run AI_Estimate/Streamlit_ui/app.py
```

# Concept: GNN-CVAE Pedestrian Full-Trajectory Generation

## Goal
Train a graph-based conditional variational autoencoder that generates whole pedestrian trajectories for all agents in one simulation while respecting:
- walkable geometry
- start and goal information
- time progression
- social interaction between nearby agents

## Core Idea
Each simulation is treated as a time-varying multi-agent graph.
At each frame:
- each agent is a node
- nearby agents define edges
- message passing builds a social context vector

A CVAE latent variable captures per-agent motion style. The decoder then rolls trajectories forward in time using:
- latent z
- spawn point
- exit goal
- geometry embedding
- GNN social context
- current time index

## Training Signals
The baseline implementation optimizes:
- trajectory reconstruction loss
- KL divergence loss
- goal consistency loss

Later we can extend it with:
- boundary / walkable-area penalties
- collision penalties
- flow-distribution matching

## Output Structure
Training writes to:
AI_Result/Method_GNN_CVAE/outputs/run_N/
- logs/training_history.csv
- weights/latest_model.pth
- weights/best_model.pth
- weights/epoch_XXX.pth
- samples/epoch_XXX/ validation exports for GT vs AI

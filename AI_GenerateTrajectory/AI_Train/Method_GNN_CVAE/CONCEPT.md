# Concept: GNN-CVAE Pedestrian Curriculum

## Goal

Train a graph-based conditional variational autoencoder that can generate pedestrian trajectories while respecting:
- start and goal
- walkable geometry
- nearby-agent interaction
- full-scene multi-agent structure

This method is now treated as a staged curriculum instead of one flat training recipe.

## Curriculum Overview

### Step 1: Goal Only

Teach the simplest skill first:
- one pedestrian at a time
- no social interaction
- focus on learning start -> goal motion

Purpose:
- verify that the goal-conditioned decoder works
- verify that the model can produce plausible single-agent paths before adding harder constraints

### Step 1.5: Goal + Geometry

Add spatial discipline before social complexity:
- still no social interaction
- a few agents at most
- stronger out-of-bounds penalty
- lower learning rate for fine-tuning from step 1

Purpose:
- learn which regions are walkable
- avoid walls, holes, and map exits
- stabilize trajectories inside valid corridors

This step exists because the jump from goal-only to goal+social was too large. The model should first understand the map itself.

### Step 2: Goal + Social

Add local interaction after geometry behavior is acceptable:
- multi-agent scenes
- GNN message passing between nearby agents
- still goal-conditioned

Purpose:
- preserve movement toward the goal
- react to neighboring agents
- reduce unrealistic overlap in dense scenes

### Step 3: Full Dataset

Scale the same idea to the intended full setting:
- more agents
- more cases
- longer stored sequences
- full training subset

Purpose:
- move from controlled curriculum stages to realistic data scale

## Core Model Idea

Each simulation case is treated as one multi-agent sample.

At each frame:
- each agent is a node
- nearby agents define graph edges
- message passing builds a social context vector

A CVAE latent variable is used per agent to capture trajectory style variation.

The decoder generates trajectories using:
- start point
- goal point
- geometry embedding
- latent code
- optional social context
- time progression

## Decoder Philosophy

The decoder is goal-anchored.

It does not only predict arbitrary free-form coordinates. Instead it uses:
- a linear anchor from start to goal
- a learned residual that bends the path away from the straight line

This gives the model a built-in tendency to move toward the goal while still allowing detours.

## Geometry Understanding

Geometry enters the model in two ways:
- a geometry encoder turns the walkable mask into a learned embedding
- a differentiable out-of-bounds loss penalizes predicted points that fall outside the walkable area

The geometry curriculum step increases the importance of this second signal on purpose.

## Social Understanding

When social mode is enabled:
- nearby agents inside `neighbor_radius` exchange messages through the custom GNN
- this social context is injected into the decoder at each step

This is introduced only after the model has a better grasp of valid walkable-space behavior.

## Training Signals

The shared implementation optimizes:
- reconstruction loss
- KL divergence loss
- goal consistency loss
- out-of-bounds loss

Interpretation:
- reconstruction keeps the path close to stored GT
- KL regularizes latent space
- goal loss keeps the final prediction oriented toward the exit goal
- OOB loss discourages walking through walls or outside the map

## Important Data Limitation

The stored training trajectory is not the raw full simulation:
- frames are sub-sampled by `frame_stride`
- then truncated by `max_seq_len`

So visualized GT exports may stop before the physical exit even when the raw simulation continues.

This matters when interpreting plots and FDE.

## FDE Interpretation

The plotted goal star corresponds to the centroid of the exit polygon.

Raw GT often ends near the exit boundary, not necessarily at the centroid.
Truncated GT may stop even earlier.

So FDE against `GT_final` can be misleading if the question is "did the model reach the exit area?"

For this reason, using exit-centroid distance as the "reach-the-goal" reference is often more meaningful for this method.

## Recommended Training Order

1. train `Step_01_GoalOnly`
2. fine-tune into `Step_01_5_GoalGeometry`
3. fine-tune into `Step_02_GoalSocial`
4. fine-tune into `Step_03_FullDataset`

This is the intended use of the method going forward.

# Method_Transformer_SF_01

Research name: **Social-Force-Informed Joint Multi-Agent Transformer**

Status: **trainable** through `run_pipeline.py`. The active pipeline uses the
shared synchronized HouseGAN scene loader, analytic desired/agent/wall force
prior, temporal Transformer encoder, inter-agent attention, bounded learned
residual, stop head, scheduled sampling, and full-path autoregressive rollout.

The copied legacy ego-agent trainer is intentionally disabled.

This method will predict all active agents synchronously. Its intended inputs
are trajectory history, velocity, exit/navigation direction, walkable geometry,
wall distance/normal, neighboring-agent state, and local density. The model
will combine temporal attention with inter-agent social attention and learn a
residual over social-force-inspired desired, agent-repulsion, and wall-repulsion
components.

Required outputs:

- `Transformer-SF-Raw`: direct joint model rollout.
- `Transformer-SF-Constrained`: rollout after the shared walkability,
  collision, kinematic, and exit safety executor.

The training and evaluation dataset must use the canonical Image-based
HouseGAN plan split.

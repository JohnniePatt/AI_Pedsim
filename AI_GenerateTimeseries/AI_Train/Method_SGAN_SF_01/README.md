# Method_SGAN_SF_01

Research name: **Social-Force-Informed Joint Multi-Agent Social GAN**

Status: **trainable** through `run_pipeline.py`. The active pipeline contains
a stochastic synchronized joint generator, scene discriminator, adversarial
objective, inter-agent attention, analytic desired/agent/wall force prior,
bounded learned residual, stop head, scheduled sampling, and mean@K full-path
evaluation. The copied non-adversarial legacy trainer is disabled.

This method must be implemented as a genuine conditional GAN, including a
stochastic joint generator, scene-level discriminator, adversarial objective,
social interaction module, geometry/goal conditioning, and multimodal sampling.
It will predict all active agents synchronously and learn a residual over
social-force-inspired desired, agent-repulsion, and wall-repulsion components.

Required outputs:

- `SGAN-SF-Raw`: direct stochastic generator rollout.
- `SGAN-SF-Constrained`: rollout after the shared walkability, collision,
  kinematic, and exit safety executor.

The training and evaluation dataset must use the canonical Image-based
HouseGAN plan split.

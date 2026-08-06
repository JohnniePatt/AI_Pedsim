# Method_LSTM_SF_01

Research name: **Social-Force-Informed Joint Multi-Agent LSTM**

Status: **trainable** through `run_pipeline.py`. The active pipeline uses the
canonical synchronized HouseGAN loader, temporal LSTM encoder, inter-agent
attention, analytic desired/agent/wall force prior, bounded learned residual,
stop head, scheduled sampling, and full-path autoregressive rollout. All copied
Topo_bottleneck trainers are intentionally disabled.

This method is separate from the existing `Method_LSTM_01` bottleneck
pipeline. It will use a shared temporal LSTM per agent plus inter-agent social
attention or graph pooling, followed by a synchronous joint decoder. Intended
conditioning includes navigation direction, walkable geometry, wall
distance/normal, relative position/velocity, time-to-collision, and local
density. The network will learn a residual over social-force-inspired desired,
agent-repulsion, and wall-repulsion components.

Required outputs:

- `LSTM-SF-Raw`: direct joint model rollout.
- `LSTM-SF-Constrained`: rollout after the shared walkability, collision,
  kinematic, and exit safety executor.

The training and evaluation dataset must use the canonical Image-based
HouseGAN plan split.

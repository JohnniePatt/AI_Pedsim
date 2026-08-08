# Method_LSTM_SF_01

Research name: **Social-Force-Informed Joint Multi-Agent LSTM**

Status: **trainable** through `run_pipeline.py`. Interactive runs show the
standard operation menu first, with `Check configuration` as the default. When
training is selected, the script asks whether to use the fast sanity-training
profile, rotating quarter-plan profile, rotating half-plan profile, or the full
research-scale profile; use `--stage train --profile fast`,
`--stage train --profile quarter`, `--stage train --profile half`, or
`--stage train --profile full` to skip the prompts. The active pipeline uses the
canonical synchronized HouseGAN loader, temporal LSTM encoder, inter-agent
attention, analytic desired/agent/wall force prior, bounded learned residual,
stop head, scheduled sampling, case-grouped batch loading, and full-path autoregressive rollout. All copied
Topo_bottleneck trainers are intentionally disabled.
The full profile keeps all training windows but uses larger batches plus AMP to
reduce batch count and per-step overhead.
The quarter profile samples 25% of training plans each epoch, resampling on
each new epoch, so epoch time is shorter while coverage accumulates over time.
The half profile samples 50% of training plans each epoch as a stronger
middle-ground protocol.

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

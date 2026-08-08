# Method_Transformer_SF_01

Research name: **Social-Force-Informed Joint Multi-Agent Transformer**

Status: **trainable** through `run_pipeline.py`. The active pipeline uses the
shared synchronized HouseGAN scene loader, analytic desired/agent/wall force
prior, temporal Transformer encoder, inter-agent attention, bounded learned
residual, stop head, scheduled sampling, and full-path autoregressive rollout.

The copied legacy ego-agent trainer is intentionally disabled.

## Train

Interactive pipeline:

```bash
cd /home/johnfaqpc/programming/AI_Pedsim/AI_GenerateTimeseries/AI_Train/Method_Transformer_SF_01
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py
```

When training is selected, the menu offers:

```text
1) fast    - quick debug/sanity training
2) quarter - rotate 25% of train plans each epoch
3) full    - full research-scale training
```

Direct profile commands:

```bash
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile fast
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile quarter
/home/johnfaqpc/programming/AI_Pedsim-env/bin/python3 run_pipeline.py --profile full
```

`quarter` uses the shared joint-SF case-grouped sampler with
`case_fraction_per_epoch=0.25` and `group_fraction_by_plan=true`, so each epoch
samples about 25% of train plans and rotates the sampled plans on the next epoch.
It is faster per epoch than `full`, but it is a stochastic training protocol and
should be reported separately from full-data-per-epoch training.

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

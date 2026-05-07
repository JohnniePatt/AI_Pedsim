# CVAE Concept

This method handles one-to-many mapping better than deterministic image-to-image models by using a latent variable `z`.

Current implementation:

- Framework: PyTorch
- Input: scenario image `A`
- Target: binary trajectory-line mask `B`
- Output: 1-channel trajectory probability map
- Inference default: `z=0` for stable repeatable output
- Optional stochastic inference: use `--num_samples N` during test to average N latent samples

Training objective:

- L1 reconstruction on probability map
- Weighted BCE for sparse foreground pixels
- Dice loss
- Sobel edge loss
- KL regularization with annealing

Note: this is still an image-level CVAE. For true per-agent trajectories, the next step should condition on each agent/start point and generate one trajectory per agent.

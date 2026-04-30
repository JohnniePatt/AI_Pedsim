# CVAE Concept

This method handles one-to-many mapping better than deterministic image-to-image generators.

- Input: map-condition image `A`
- Target: trajectory-line image `B`
- Latent variable `z` is sampled internally (no extra z file required)

Training objective:
- Weighted L1 reconstruction
- Mask BCE
- Mask Dice
- KL regularization with annealing

Inference default in this implementation uses `z=0` for stable, repeatable outputs.

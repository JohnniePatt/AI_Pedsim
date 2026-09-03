# Image-model 2x2 factorial protocol at 256

This implementation is isolated from the historical image baselines. It tests two
generator architectures (`unet`, `resnet9`) crossed with two training objectives
(`l1`, `wgangp_l1`). The two cells in each architecture pair instantiate the exact
same generator class and, for a given seed, have identical initial-weight hashes.

Controlled variables include the canonical HouseGAN split, 256 x 256 bilinear
preprocessing, one-channel target, input/target ranges, batch size, epoch budget,
generator optimizer, learning rate, data order, seeds, validation checkpoint rule,
test batch size, timing boundary, and metric implementation. Both adversarial cells
share the exact same conditional PatchGAN critic and WGAN-GP objective.

The `Pix2PixHD factorial variant` label preserves the comparison cell requested by
the project, but this cell intentionally does not claim to be the historical
Pix2PixHD implementation: feature matching, density weighting, multi-scale critics,
three-channel targets, and the different generator definition are removed because
they would confound the 2x2 design.

The active pilot config uses seed `42` only (four runs). A one-seed factorial can
report the four cell results and within-seed main-effect/interaction contrasts, but
it cannot estimate between-seed variance or a confidence interval. Additional
seeds must therefore be completed before treating the factorial inference as final.

Runtime protocol `image_test_runtime_v2` defines `test_pipeline_wall_time_s` from
checkpoint loading through metric-summary writing. `metrics_wall_time_s` includes
the density metrics, walkable metrics, LPIPS setup/inference, aggregation, and
metric-summary writes. `runtime_excluding_metrics_s` subtracts that metric time,
while `Time Generate` synchronizes CUDA around Generator forward only. The
prediction/output loop is also retained separately as
`prediction_output_loop_wall_time_s`.

Normal entry point:

```bash
python run_pipeline.py
```

Automation:

```bash
python run_pipeline.py --stage plan
python run_pipeline.py --stage all
python run_pipeline.py --stage all --experiment-dir <existing experiment directory>
```

Completed runs are renamed only after canonical test evaluation succeeds:

```text
run_<method>_<UTC timestamp>_seed<seed>__model_evaluate_256_factorial
```

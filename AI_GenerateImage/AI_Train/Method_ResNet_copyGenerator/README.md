# Method_ResNet_copyGenerator

This isolated experiment copies only the exact Generator architecture used by the
active original Pix2PixHD run: RGB input/output, three downsampling stages, nine
512-channel residual blocks, affine InstanceNorm, three transposed-convolution
upsampling stages, and Tanh output. It contains no discriminator and no
feature-matching or adversarial loss.

Training uses seed 42, 256 x 256 images, batch size 8, 50 epochs, Adam
(`lr=0.0002`, betas `0.0, 0.9`) and Pix2PixHD's density-aware L1 reconstruction
term. Model selection uses the canonical validation split and final evaluation
uses the canonical 862-case/117-floorplan test split.

```bash
python run_pipeline.py --stage plan
python run_pipeline.py --stage all
```

This new run can be compared descriptively with the retained original Pix2PixHD
run. It is not a paired-seed factorial contrast because that historical run does
not record its seed or initial Generator hash.

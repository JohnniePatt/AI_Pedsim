# Corrected representative 2x2 image comparison

This evaluator compares existing saved predictions without retraining:

- the corrected shared-generator U-Net pair (Plain U-Net and Pix2Pix WGAN-GP), and
- the original project implementations of ResNet-9 and Pix2PixHD.

It is a **method-family representative comparison**, not a strict
component-isolated factorial. All four saved PNG prediction sets are evaluated
against the same canonical 862-case HouseGAN test split after bilinear resizing
to 256 x 256. Pixel metrics and AlexNet LPIPS (full-image and walkable-mask) use
that common representation. Using the saved uint8 PNGs is intentional because
the legacy runs do not retain float predictions.

Run with the project environment:

```bash
/home/johnnie/programming/AI_Pedsim/AI_Pedsim-env/bin/python3 \
  AI_GenerateImage/AI_Train/Method_ImageRepresentative_2x2/evaluate_representative_2x2.py
```

The evaluator only creates a new timestamped directory below
`AI_GenerateImage/AI_Result/RepresentativeComparisons`; it does not alter the
dataset, checkpoints, predictions, or prior metric files.

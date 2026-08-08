# Concept: Method_ResNet (9-Block ResNet Generator)

## Overview
`Method_ResNet` implements a **9-Block ResNet Generator** architecture matching the exact generator depth of `Pix2PixHD`, but optimized directly with reconstruction L1 loss to serve as a pure ResNet architecture baseline.

## Key Architectural Specifications
- **Input:** $256 \times 256 \times 3$ RGB Floorplan Image
- **Downsampling:** 3 Convolutional Blocks ($256 \rightarrow 128 \rightarrow 64 \rightarrow 32$)
- **Residual Core:** **9 ResNet Residual Blocks** ($x + f(x)$ with Instance Normalization)
- **Upsampling:** 3 Transposed Convolutional Blocks ($32 \rightarrow 64 \rightarrow 128 \rightarrow 256$)
- **Total Stage Depth:** 15 Stage Blocks (Identical Generator Depth to `Pix2PixHD`)

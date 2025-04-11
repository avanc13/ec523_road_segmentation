# EC523 Road Segmentation

This repository contains our implementation of ResUNet and Unet for road segmentation using the Massachusetts Roads Dataset for our EC523 final project.
We hace used source code from https://github.com/edwinpalegre/EE8204-ResUNet/tree/master?tab=readme-ov-file for baseline resUnet and dataset downloading.
We aim to compare the performance of our custom ResUNet architecture against traditional UNet for binary segmentation of aerial images.

We use the Massachusetts Roads Dataset, which consists of:

1108 training images

49 test images

14 validation images

We implemented a custom ResUNet with:

Pre-activation residual blocks

Fewer filters (16/32/64) to reduce overfitting and stabilize training

Bilinear upsampling in the decoder

We trained using:

Weighted Binary Cross Entropy (BCE) loss to address class imbalance


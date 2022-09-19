
#
HAZE-Net: High-Frequency Attentive Super-Resolved Gaze Estimation in Low-Resolution Face Images.

This code is the PyTorch implementation of HAZE-Net.

To prove our code's reproducibility, we present validation of HAZE-Net on MPIIFaceDatsets (9,000 images) for scale factor 4x.

# Datasets
LR : './mpii_test/LR/x4/'
HR : './mpii_test/val'

## Weights

# HAZE_SR weights
https://drive.google.com/file/d/1BnjKFKPj2RjqJGsMzyFZM5Oc7OwMUEUp/view?usp=sharing

dir:
'./SR_weights/hazex4_mpii/model/HAZE_SR_weights.pt'

# HAZE_Gaze weights
https://drive.google.com/file/d/1XL9jJ4ZW924D_4qlf6ZlO4Kq6gB6XvMj/view?usp=sharing

dir:
'./Gaze_weights/haze_mpii/model/HAZE_gaze_weights.pt'

## Create enviroments
conda env create -f hazenet.yaml

## Quick Run
python main.py --test_only --scale 4
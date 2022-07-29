<h1 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #CCCCFF; color: black;"><center><br>HuBMAP + HPA 👀: Hacking the Human Body</center></h1>

![](https://drive.google.com/uc?id=1pbIvjTlhGywfhiMTqcsdOB5LSHlklM90)

---
This is a work in progress
---
## Tech Stack: 
[Pytorch](https://pytorch.org/)

[Albumentations](https://albumentations.ai/docs/) for augmentations

[Semi-Supervised ImageNet1K Models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py) for modeling

## Methodology:

Due to limited computation resources, we focus more on data augmentation, data collection, hyperparameter tuning, and post-processing.


### Data + Augmentation:

* External data: collect on many stains of PAS, H&E, DAB/H of the target organs
* Augmentation: robust augmentation pipeline (basic, morphology, color)

### Model architecture:
* UNeXt101 + UNeXt50

### Training setup:
* Loss functions: BCE, and symmetric Lovasz
* Iterations: 1000 with slicing learning rates

### Testing setup:
* Evaluation and Inference of the predictions from multiple models with Test Time Augmentation(TTA)

### Post-processing: 
* CascadePSP

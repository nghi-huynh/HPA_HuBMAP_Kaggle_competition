# HPA + HuBMAP: Hacking the Human Body

## Methodology:

Since we don’t have the computation resources, we will focus more on data augmentation, data collection, hyperparameter tuning, and post-processing.


### Data + Augmentation:

* External data: collect on many stains of PAS, H&E, DAB/H of the target organs
* Augmentation: basic, morphology, color

### Model architecture:
* UNeXt101

### Training setup:
* Loss functions: BCE, and symmetric Lovasz
* Iterations: 1000 with slicing learning rates

### Post-processing: 
* CascadePSP

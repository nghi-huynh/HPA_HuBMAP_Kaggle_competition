<h1 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #CCCCFF; color: black;"><center><br>HuBMAP + HPA 👀: Hacking the Human Body</center></h1>

![](https://drive.google.com/uc?id=1pbIvjTlhGywfhiMTqcsdOB5LSHlklM90)

---
This is a work in progress!
---
## Tech Stack: 
[Pytorch](https://pytorch.org/)

[Albumentations](https://albumentations.ai/docs/) for augmentations

[Semi-Supervised ImageNet1K Models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models/blob/master/hubconf.py) for modeling

## Task description:

---

In this [Kaggle competition](https://www.kaggle.com/competitions/hubmap-organ-segmentation), we'll **identify and segment functional tissue units (FTUs)** accross **five** human organs:

* Prostate
* Spleen
* Lung
* Kidney
* Large Intestine

The challenge in this competition is to build algorithms that **generalize**:
* across different **organs** and
* across different **dataset** differences

=> This is a **semantic segmentation** problem.

## Data description:

---

They only release public *Human Protein Atlas (HPA)* data for the training dataset. However, they will release private *HPA* data and *Human BioMolecular Atlas Program (HuBMAP)* for their public test set. For the private test set, they only use *HuBMAP* data.

## Methodology:

---

Due to limited computation resources, we focus more on **data augmentation**, **data collection**, **hyperparameter tuning**, and **post-processing**.


### Data + Augmentation:

* External data: collect on many stains of PAS, H&E, DAB/H of the target organs
* Augmentation: robust augmentation pipeline (basic, morphology, color)

### Model architecture:
* UNeXt101
* UNeXt50 (baseline)

### Training setup:
* Loss functions: BCE, and symmetric Lovasz
* Iterations: 1000 with slicing learning rates

### Testing setup:
* Evaluation and Inference of the predictions from multiple models with Test Time Augmentation(TTA)

### Post-processing: 
* CascadePSP

## Usage:

---

* Clone the repository

**1. Data Exploration**

  * Visualize data using `notebooks/EDA.ipynb`

**2. Stain Normalization**

  * Normalize training data based on StainNet using `notebooks/Stain_Normalization.ipynb`

**3. Train**

  * Train models using `notebooks/training-fastai-baseline.ipynb`

**4. Inference**

  * Validate models and generate submissions using `notebooks/inference-fastai-baseline.ipynb`

## Code structure

---

```
src
├── data
│   ├── dataset.py          # Torch datasets
│   └── transforms.py       # Augmentations
├── inference
│   └── test.py             # Test inference
├── models
│   └── models.py           # Model architectures
├── params.py               # Main parameters
├── training
│   ├── predict.py          # Prediction functions
│   └── train.py            # Model Fitting
└── utils
    ├── lovasz_loss.py      # Loss functions
    ├── metrics.py          # Metrics for the competition
    └── rle.py              # RLE encoding utils
```

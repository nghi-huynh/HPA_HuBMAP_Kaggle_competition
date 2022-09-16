<h1 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 3px; background-color: #CCCCFF; color: black;"><center><br>HuBMAP + HPA 👀: Hacking the Human Body</center></h1>

<img src='imgs/hubmap_header.png'>

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


## Usage:

---

Clone the repository

**1. Data Exploration**

  * Visualize data using `notebooks/EDA.ipynb`

**2. Stain Normalization**

  * Normalize training data based on StainNet using `notebooks/Stain_Normalization.ipynb`
  
**3. Data Augmentation: Pyramid Blending**

  * Blend images and masks based on organ to augment imbalanced classes (spleen, lung, and large intestine) using `notebooks/data-augmentation-laplacian-pyramid-blending.ipynb`
  
**4. WSI Preprocessing**
  * Tile and identify tissue/non-tissue based on thresholding technique using `notebooks/wsi-preprocessing-tiling-tissue-segmentation.ipynb`

**5. Train**

  * Train models using `notebooks/training-fastai-baseline.ipynb`

**6. Inference**

  * Validate models and generate submissions using `notebooks/inference-fastai-baseline.ipynb`

## Code structure

---

```
src
├── data
│   ├── dataset.py              # Torch dataset
│   └── transforms.py           # Augmentation
├── data_preparation
│   ├── data_preparation.py     # Generate rescaled + tilede dataset
│   ├── dataset.py              # Torch dataset
│   ├── get_config.py           # Configuration
│   └── utils.py                # Util functions
├── inference
│   └── test.py                 # Test inference
├── models
│   └── models.py               # Model architectures
├── params.py                   # Main parameters
├── training
│   ├── predict.py              # Prediction helper functions
│   └── train.py                # Model fitting
└── utils
    ├── lovasz_loss.py          # Loss functions
    ├── metrics.py              # Metrics used in the competition
    └── rle.py                  # RLE encoding utils
```

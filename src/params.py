# Hardcoded stuff, paths are to adapt to your setup

import torch
import numpy as np

MEAN = np.array([0.7720342, 0.74582646, 0.76392896])
STD = np.array([0.24745085, 0.26182273, 0.25782376])
TH = 0.225  # threshold for positive predictions

METADATA_PATH = '../input/hubmap-2022-256x256/train/'
TRAIN_PATH = '../input/hubmap-2022-256x256/masks/'
MASK_PATH = '../input/hubmap-organ-segmentation/train.csv'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 64

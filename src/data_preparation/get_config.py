import random
from bleach import VERSION

VESION = '01_01'
def get_config():
    config = {
        'VERSION': VERSION,
        'OUT_TRAIN': 'tiled_images.zip',
        'OUT_MASK': 'tiled_masks.zip',
        'OUT_RESCALED_TRAIN': 'rescaled_imgs.zip',
        'OUT_RESCALED_MASK' : 'rescaled_masks.zip',
        'MASKS_PATH' : '../input/hubmap-organ-segmentation/train.csv',
        'DATA_PATH' : '../input/hubmap-organ-segmentation/train_images',
        'tile_sz' : 256,
        'scaled_sz': (1024,1024),
    }
    return config
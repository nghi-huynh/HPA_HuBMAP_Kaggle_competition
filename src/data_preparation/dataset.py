import os
import tifffile as tiff
from torch.utils.data import Dataset
from data_preparation.utils import *
from data_preparation.get_config import get_config

config = get_config()

class HuBMAPDataset(Dataset):
    
    def __init__(self, idx, scale_sz, tile_sz, rle=None):
        # load img, mask
        self.img = tiff.imread(os.path.join(config['DATA_PATH'], str(idx)+'.tiff'))
        self.mask = rle2mask(rle, (self.img.shape[1], self.img.shape[0])) if rle is not None else None
        
        self.scale_sz = scale_sz
        self.tile_sz = tile_sz
        
        self.scaled_img, self.scaled_mask = rescale(self.img, self.mask, self.scale_sz)
        
        self.thres = (thresholding(self.img, method='otsu') + thresholding(self.img, method='triangle'))/2
        self.idx = idx
        
    def __len__(self):
        return (self.scale_sz[0]//self.tile_sz)*(self.scale_sz[1]//self.tile_sz) # num_tiles
    
    def __getitem__(self, idx):
        img_tiles, mask_tiles = make_tiles(self.scaled_img, self.scaled_mask, self.tile_sz)
        
        selected_imgs = []
        selected_masks = []
        
        # loop through all img_tiles
        # select tiles based on the given threshold
        selected_num_tiles = 0
        for i, img_crop in enumerate(img_tiles):
            img_c = 255-img_crop
            if img_c.mean() > self.thres:
                selected_num_tiles += 1
                selected_imgs.append(img_crop)
                selected_masks.append(mask_tiles[i])
               
        return selected_imgs[idx], selected_masks[idx], idx, selected_num_tiles
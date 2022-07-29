import os
import cv2
import rasterio
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from albumentations import *
from fastai.vision.all import *
from rasterio.windows import Window
from torch.utils.data import Dataset
import warnings; warnings.filterwarnings("ignore")

from params import (METADATA_PATH, TRAIN_PATH, MASK_PATH, MEAN, STD)
from data.transforms import img2tensor


class HuBMAPDatasetTrain(Dataset):
    def __init__(self, fold, nfolds, train=True, tfms=None):
        ids = pd.read_csv(METADATA_PATH).id.astype(str).values
        kf = KFold(n_splits=nfolds,shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(TRAIN_PATH) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms # transforms
        
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_PATH,fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASK_PATH,fname),cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None: # augment images and masks if augmentation pipeline is given
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        # normalize image pixel values relative to the dataset mean and standard deviation
        # since we want each pixel to have a similar range so the gradients don't go out of control
        
        # dont normalize the mask because we dont want the target (per pixel) to change based on
        # the global statistics of the mask
        return img2tensor((img/255.0 - MEAN)/STD),img2tensor(mask) 

class HuBMAPDatasetTest(Dataset):
    def __init__(self, idx, sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(TRAIN_PATH,idx+'.tiff'), transform = rasterio.Affine(1, 0, 0, 0, 1, 0),
                                 num_threads='all_cpus')
        # some images have issues with their format 
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for _, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce*sz
        self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
        self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz
        
    def __len__(self):
        return self.n0max*self.n1max
    
    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0,n1 = idx//self.n1max, idx%self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0,y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz
        # make sure that the region to read is within the image
        p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
        p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
        img = np.zeros((self.sz,self.sz,3),np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1,2,3],
                window=Window.from_slices((p00,p01),(p10,p11))), 0, -1)
        else:
            for i,layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i] =\
                  layer.read(1,window=Window.from_slices((p00,p01),(p10,p11)))
        
        if self.reduce != 1:
            img = cv2.resize(img,(self.sz//reduce,self.sz//reduce),
                             interpolation = cv2.INTER_AREA)
        #check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _,s,_ = cv2.split(hsv)
        s_th = 40 #saturation blancking threshold
        p_th = 1000*(self.sz//256)**2 #threshold for the minimum number of pixels
        if (s>s_th).sum() <= p_th or img.sum() <= p_th:
            #images with -1 will be skipped
            return img2tensor((img/255.0 - MEAN)/STD), -1
        else: return img2tensor((img/255.0 - MEAN)/STD), idx
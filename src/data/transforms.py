import numpy as np
import torch
import cv2
from albumentations import (Compose, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
                            ShiftScaleRotate, ElasticTransform,
                            GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
                            RandomBrightnessContrast, HueSaturationValue, IAASharpen,
                            RandomGamma, RandomBrightness, RandomBrightnessContrast,
                            GaussianBlur,CLAHE, IAAPiecewiseAffine,
                            Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
                            Normalize, OneOf, NoOp)

MEAN = np.array([0.7720342, 0.74582646, 0.76392896])
STD = np.array([0.24745085, 0.26182273, 0.25782376])

def get_transforms_train(p=1):
    transform_train = Compose([
        #Basic
        RandomRotate90(p=1),
        HorizontalFlip(p=0.5),
        
        #Morphology
        ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                         interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
        GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
        GaussianBlur(blur_limit=(3,7), p=0.5),
        
        #Color
        #RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                 #brightness_by_max=True,p=0.5),
        #HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
                           #val_shift_limit=0, p=0.5),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                 brightness_by_max=True),            
        ], p=0.3),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        
    ], p=p)
    return transform_train    

def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

import cv2
import zipfile
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
#from rasterio.windows import Window
from data_preparation.get_config import get_config
from data_preparation.dataset import *

config = get_config()

df_masks = pd.read_csv(config['MASKS_PATH'])[['id', 'rle']].set_index('id')

if __name__=='__main__':
    x_tot,x2_tot = [],[]
    with zipfile.ZipFile(config['OUT_TRAIN'], 'w') as img_out,\
        zipfile.ZipFile(config['OUT_MASKS'], 'w') as mask_out,\
        zipfile.ZipFile(config['OUT_RESCALED_TRAIN'], 'w') as rescaled_img_out,\
        zipfile.ZipFile(config['OUT_RESCALED_MASK'], 'w') as rescaled_mask_out:
        for index, encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
            #image+mask dataset
            ds = HuBMAPDataset(index,rle=encs[0])
            
            # write rescaled img, mask
            # cv2.imencode -> tuple (True/False, array[])
            rescaled_img = cv2.imencode('.png', cv2.cvtColor(ds.scaled_img, cv2.COLOR_RGB2BGR))[1]
            rescaled_img_out.writestr(f'{index}.png', rescaled_img)
            rescaled_mask = cv2.imencode('.png', ds.scaled_mask)[1]
            rescaled_mask_out.writestr(f'{index}.png', rescaled_mask)
            
            for i in range(len(ds)):
                try:
                    img,mask,idx, _ = ds[i]
                    #if idx < 0: continue

                    x_tot.append((img/255.0).reshape(-1,3).mean(0))
                    x2_tot.append(((img/255.0)**2).reshape(-1,3).mean(0))

                    #write data   
                    img = cv2.imencode('.png',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]
                    img_out.writestr(f'{index}_{idx:04d}.png', img)
                    mask = cv2.imencode('.png',mask)[1]
                    mask_out.writestr(f'{index}_{idx:04d}.png', mask)
                except IndexError:
                    break
            
    #image stats
    img_avr =  np.array(x_tot).mean(0)
    img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
    print('mean:',img_avr, ', std:', img_std)
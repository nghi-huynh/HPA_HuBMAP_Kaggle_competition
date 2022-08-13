import cv2
import numpy as np
import matplotlib.pyplot as plt

# functions to convert encoding to mask and mask to encoding
# https://www.kaggle.com/paulorzp/rle-functions-run-length-encode-decode
def mask2rle(img): # encoder
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)): # decoder
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# rescale to a desire img size
# keep in mide that the size needs to be large enough
# to keep all important features for model training
def rescale(img, mask, size=(1024,1024)):
    scaled_img = cv2.resize(img, size)
    scaled_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return scaled_img, scaled_mask

def thresholding(img, method='otsu'):
    # convert to grayscale complement image
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_c = 255 - grayscale_img
    thres, thres_img = 0, img_c.copy()
    if method == 'otsu':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'triangle':
        thres, thres_img = cv2.threshold(img_c, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    return thres #, thres_img, img_c

# adapted from: https://www.kaggle.com/code/analokamus/a-fast-tile-generation
def make_tiles(img, mask, tile_size=256):
    '''
    img: np.ndarray with dtype np.uint8 and shape (width, height, channel)
    mask: np.ndarray with dtype np.uint9 and shape (width, height)
    '''
    w_i, h_i, ch = img.shape
    w_m, h_m     = mask.shape
    
    pad0, pad1 = (tile_size - w_i%tile_size) % tile_size, (tile_size - h_i%tile_size) % tile_size
    
    padding_i = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2], [0, 0]]
    padding_m = [[pad0//2, pad0-pad0//2], [pad1//2, pad1-pad1//2]]
    
    img = np.pad(img, padding_i, mode='constant', constant_values=255)
    img = img.reshape(img.shape[0]//tile_size, tile_size, img.shape[1]//tile_size, tile_size, ch)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, ch)
    
    mask = np.pad(mask, padding_m, mode='constant', constant_values=255)
    mask = mask.reshape(mask.shape[0]//tile_size, tile_size, mask.shape[1]//tile_size, tile_size)
    mask = mask.transpose(0, 2, 1, 3).reshape(-1, tile_size, tile_size)
    
    #     if len(img) < num_tiles: # pad images so that the output shape be the same
    #         padding = [[0, num_tiles-len(img)], [0, 0], [0, 0], [0, 0]]
    #         img = np.pad(img, padding, mode='constant', constant_values=255)
    #idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_tiles] # pick up Top N dark tiles
    #img = img[idxs]
    return img, mask

def visualize_tiles(ds):
    plt.figure(figsize=(12,10))
    plt.subplot(1,2,1)
    plt.imshow(ds.img)
    plt.imshow(ds.mask, cmap='seismic', alpha=0.4)
    plt.title(f'Original Image + Mask\nImage Size: {ds.img.shape}')
    plt.subplot(1,2,2)
    plt.imshow(ds.scaled_img)
    plt.imshow(ds.scaled_mask, cmap='seismic', alpha=0.4)
    plt.title(f'Rescaled Image + Mask\nImage Size: {ds.scaled_img.shape}')
    #plt.xticks([])
    #plt.yticks([])
    plt.show()
    
    plt.figure(figsize=(10,5))
    _,_,_,num_tiles = ds[0]
    for i in range(num_tiles):
        #rows = cols = int(math.sqrt(num_tiles))
        rows = 2
        cols = num_tiles//2
        plt.subplot(rows,cols,i+1)
        img, mask, _, _ = ds[i]
        plt.imshow(img)
        plt.imshow(mask, cmap='seismic',alpha=0.4)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


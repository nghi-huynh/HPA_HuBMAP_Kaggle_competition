import gc
from utils.rle import *
from params import *
from training.predict import *
from data.dataset import *
from fastai.vision.all import *
from tqdm import tqdm


def inference(df_sample):
    names,preds = [],[]
    for idx,row in tqdm(df_sample.iterrows(),total=len(df_sample)):
        idx = str(row['id'])
        ds = HuBMAPDatasetTest(idx)
        #rasterio cannot be used with multiple workers
        dl = DataLoader(ds,BATCH_SIZE,num_workers=0,shuffle=False,pin_memory=True)
        mp = Model_pred(models,dl)
        #generate masks
        mask = torch.zeros(len(ds),ds.sz,ds.sz,dtype=torch.int8)
        for p,i in iter(mp): mask[i.item()] = p.squeeze(-1) > TH
        
        #reshape tiled masks into a single mask and crop padding
        mask = mask.view(ds.n0max,ds.n1max,ds.sz,ds.sz).\
            permute(0,2,1,3).reshape(ds.n0max*ds.sz,ds.n1max*ds.sz)
        mask = mask[ds.pad0//2:-(ds.pad0-ds.pad0//2) if ds.pad0 > 0 else ds.n0max*ds.sz,
            ds.pad1//2:-(ds.pad1-ds.pad1//2) if ds.pad1 > 0 else ds.n1max*ds.sz]
        
        #convert to rle
        #https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
        rle = rle_encode_less_memory(mask.numpy())
        names.append(idx)
        preds.append(rle)
        del mask, ds, dl
        gc.collect()
    
    return names, preds
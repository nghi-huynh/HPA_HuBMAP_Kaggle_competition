import cv2
import gc
from utils.metrics import *
from data.dataset import *
from params import *
from models.models import *
from training.predict import *
from data.transforms import *



dice = Dice_th_pred(np.arange(0.2,0.7,0.01))

def save_img(data,name,out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)

def train(nfolds):
    for fold in range(nfolds):
        ds_t = HuBMAPDatasetTrain(fold=fold, train=True, tfms=get_transforms_train())
        ds_v = HuBMAPDatasetTest(fold=fold, train=False)
        data = ImageDataLoaders.from_dsets(ds_t,ds_v,bs=BATCH_SIZE,
                    num_workers=NUM_WORKERS,pin_memory=True).cuda()
        model = UneXt50().cuda()
        learn = Learner(data, model, loss_func=symmetric_lovasz,
                    metrics=[Dice_soft(),Dice_th()], 
                    splitter=split_layers).to_fp16()
        
        #start with training the head
        learn.freeze_to(-1) #doesn't work
        for param in learn.opt.param_groups[0]['params']:
            param.requires_grad = False
        learn.fit_one_cycle(4, lr_max=0.5e-2)

        #continue training full model
        learn.unfreeze()
        learn.fit_one_cycle(16, lr_max=slice(2e-4,2e-3),
            cbs=SaveModelCallback(monitor='dice_th',comp=np.greater))
        torch.save(learn.model.state_dict(),f'model_{fold}.pth')
        
        #model evaluation on val and saving the masks
        mp = Model_pred(learn.model,learn.dls.loaders[1])
        with zipfile.ZipFile('val_masks_tta.zip', 'a') as out:
            for p in progress_bar(mp):
                dice.accumulate(p[0],p[1])
                save_img(p[0],p[2],out)
        gc.collect()

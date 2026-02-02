import os 
import torch
import tifffile

import argparse
import DenT
import data
from utils import *
import torch.nn as nn
import torch.nn.functional as F

# 
import sys
from pathlib import Path
from datetime import datetime

from misc_utils import print_nvidia_smi, \
    set_reproducibility, seed_worker, get_args, set_args_dirs, load_config
# -----------------------------------------------------------------------------/


def reconstruct(pred, seg):
    """ (docstring)
    """
    pred_patch, seg_patch, new_preds, new_segs = list(),list(),list(),list()
    for i in range(len(pred)):
        pred_patch.append(pred[i])
        seg_patch.append(seg[i])    
    for i in range(0,len(pred),2):
        new_preds.append(np.concatenate(pred_patch[i:i+2], axis=2))
        new_segs.append(np.concatenate(seg_patch[i:i+2], axis=2))
    recon_preds = np.expand_dims(np.concatenate(new_preds, axis=1), 0)
    recon_segs = np.expand_dims(np.concatenate(new_segs, axis=1), 0)
    
    return recon_preds, recon_segs
    # -------------------------------------------------------------------------/


def evaluate(args, model, data_loader, save_img=False):
    """ (docstring)
    """
    preds_dict = dict()
    gts_dict = dict()

    criterion = None
    criterion = BCEDiceLoss() #nn.BCEWithLogitsLoss() #BCEDiceLoss()
    loss = 0
    count = 0

    model.eval() # <-- 切換到驗證模式, 鎖定 BatchNorm + 停用 Dropout
    with torch.no_grad():
        for idx, (img_names, imgs, segs) in enumerate(data_loader):
            if args.patch == True:
                imgs = imgs.contiguous().view(-1, 1, 32, 256, 256)
                segs = segs.contiguous().view(-1, 1, 32, 256, 256)

            imgs = imgs.cuda()
            segs = segs.cuda()
            preds = model(imgs) #(batch_size, C, H, W)
            
            #evaluate validation loss
            loss += criterion(preds, segs)
            count += 1

            preds = torch.sigmoid(preds).cpu().numpy().squeeze(1)
            segs = segs.cpu().numpy().squeeze(1)
            
            # reconstruct images from patches
            if save_img:
                if args.patch == True:
                    preds, segs = reconstruct(preds, segs)
                for img_name, pred, seg in zip(img_names, preds, segs): 
                    preds_dict[img_name] = pred
                    gts_dict[img_name] = seg  
            else:
                for i in range(len(preds)):
                    preds_dict[str(i)] = preds[i]
                    gts_dict[str(i)] = segs[i]

    gts = np.concatenate(list(gts_dict.values()))
    preds = np.concatenate(list(preds_dict.values()))

    meanIoU = iou_score(preds, gts)
    meanloss = loss / count

    if args.seg_dir != '' and save_img:
        if not os.path.exists(args.seg_dir):
            os.makedirs(args.seg_dir)
        for img_name, pred in preds_dict.items():
            tifffile.imwrite(os.path.join(args.seg_dir, img_name), (pred*255).astype('uint8'))

    return meanIoU, meanloss
    # -------------------------------------------------------------------------/


if __name__ == '__main__':
    
    args = get_args("test")  # 

    ''' Setup GPU '''
    # torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache() # 

    ''' Setup Random Seed '''
    set_reproducibility(args.random_seed) # 
    seed = np.random.randint(100000) # 
    assert seed == 15725 # 
    
    ''' Set dirs '''
    set_args_dirs(args, seed, "test") # 
    time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    print(f"datetime: {time_stamp}\n")
    
    ''' Retrieve params of `CusDenT` from `args.toml` '''
    if args.model == "CusDenT":
        found_list = list(Path(args.log_dir).glob("*_args.toml"))
        assert len(found_list) == 1, "`args.toml` is not unique !!!"
        args_config = load_config(found_list[0])
        args.use_multiheads = args_config["use_multiheads"]
        args.add_pos_emb = args_config["add_pos_emb"]
    
    ''' Load Dataset and Prepare Dataloader '''
    print('===> Preparing dataloader ... ')
    test_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='test'),
                                         batch_size=args.test_batch,
                                         num_workers=args.workers,
                                         pin_memory=True, # 
                                         shuffle=False)
    
    ''' prepare best model for visualization and evaluation '''
    print('===> Preparing model ...')
    model = None
    if args.model == 'DenT':
        model = DenT.DenseTransformer(args)
    elif args.model == 'CusDenT':
        model = DenT.CustomizableDenT(add_pos_emb=args.add_pos_emb,
                                      use_multiheads=[bool(m) for m in args.use_multiheads])
    else:
        raise NotImplementedError

    '''Data Parallel'''
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    
    best_model_path = Path(args.checkpoints).joinpath(f"model_{args.model}_best_pth.tar") # 
    best_checkpoint = torch.load(best_model_path); print(f"load model : {best_model_path.resolve()}") # 
    model.load_state_dict(best_checkpoint)
    mIoU, test_loss = evaluate(args, model, test_loader, save_img=True)
    print('Testing mIoU: {}'.format(mIoU))
    print('Testing loss: {}'.format(test_loss.data.cpu().numpy()))
    print()
    print_nvidia_smi() # 
    # -------------------------------------------------------------------------/
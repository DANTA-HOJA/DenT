import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", '.*output shape of zoom.*')

""" self-defined modules """
from test import evaluate

import data
import DenT
from misc_utils import (dump_config, get_args, print_nvidia_smi, seed_worker,
                        set_args_dirs, set_reproducibility)
from utils import BCEDiceLoss
# -----------------------------------------------------------------------------/


def save_model(model, save_path):
    """ (docstring)
    """
    torch.save(model.state_dict(), save_path)
    # -------------------------------------------------------------------------/


def main():
    """ (docstring)
    """
    args = get_args("train")  # 

    ''' Setup GPU '''
    # torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache() # 

    ''' Setup Random Seed '''
    set_reproducibility(args.random_seed) # 
    g = torch.Generator() # 
    g.manual_seed(0) # 
    seed = np.random.randint(100000)

    ''' Set dirs '''
    set_args_dirs(args, seed, "train") # 
    time_stamp: str = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    print(f"datetime: {time_stamp}\n")
    dump_config(Path(args.log_dir).joinpath(f"{time_stamp}_args.toml"), args.__dict__) # 

    ''' Load Dataset and Prepare Dataloader '''
    print('===> Preparing dataloader ... ')
    train_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               worker_init_fn=seed_worker, # 
                                               generator=g, # 
                                               pin_memory=True, # 
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.SegDataset(args, mode='val'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             pin_memory=True, # 
                                             shuffle=False)

    ''' Load Model '''
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

    ''' Define Loss '''
    criterion = None
    criterion = BCEDiceLoss() #nn.BCEWithLogitsLoss() #BCEDiceLoss() #nn.CrossEntropyLoss()

    ''' Setup Optimizer '''
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    ''' Setup Tensorboard '''
    writer = SummaryWriter(args.log_dir) # 
    
    ''' Train Model '''
    print('===> Start training ...\n') # 
    iters = 0
    best_mIoU = 0
    best_val_loss = np.inf # 
    best_epoch = -1
    for epoch in range(1, args.epoch + 1):

        # /--- Training the Model ---/
        model.train() # <-- 切回訓練模式
        for idx, (img_ids, imgs, segs) in enumerate(train_loader): # 
            train_info = 'Epoch: [{0}][{1}/{2}], [{3}]'.format(epoch, idx+1, len(train_loader), img_ids) # 
            iters += 1

            if args.patch == True:
                imgs = imgs.contiguous().view(-1, 1, 32, 256, 256)
                segs = segs.contiguous().view(-1, 1, 32, 256, 256)
            
            imgs, segs = imgs.cuda(), segs.cuda()
            print(f"imgs_shape: {imgs.shape}, segs_shape: {segs.shape}")
            
            if args.deep_supervision:
                outputs = model(imgs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, segs)
                loss /= len(outputs)

            else:
                output = model(imgs)
                loss = None
                loss = criterion(output, segs)
            
            optimizer.zero_grad()   # set grad of all parameters to zero
            loss.backward()         # compute gradient for each parameters
            optimizer.step()        # update parameters
            
            ''' Write Out Information to Tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:4f}'.format(loss.data.cpu().numpy())
            print(train_info)

        # /--- Evaluate the Model ---/
        if epoch % args.val_epoch == 0:
            mIoU, val_loss = evaluate(args, model, val_loader)
            writer.add_scalar('val_mIoU', mIoU, epoch)
            print('Epoch [{}]: mean IoU: {}'.format(epoch, mIoU))
            writer.add_scalar('val_loss', val_loss.data.cpu().numpy(), epoch)
            print('Epoch [{}]: validation loss: {}'.format(epoch, val_loss.data.cpu().numpy()))

            # Save Best Model 
            if val_loss < best_val_loss - 1e-4:
                save_model(model, Path(args.checkpoints).joinpath(f"model_{args.model}_best_pth.tar")) # 
                #best_mIoU = mIoU
                best_val_loss = val_loss
                best_epoch = epoch

            # CLI divider
            print(); print("="*80, "\n") # 
        
        # /--- Save Model (routine) ---/
        if epoch % 50 == 0:
            save_model(model, Path(args.checkpoints).joinpath(f"model_{args.model}_{epoch}_pth.tar")) # 
    
    print('The best model occurred in Epoch [{}] with validation loss {}'.format(best_epoch, best_val_loss))
    print()
    print_nvidia_smi() # 
    # writer.add_graph(model, imgs) #  ( 改成 evaluate 再存 )
    # -------------------------------------------------------------------------/


if __name__ == '__main__':
	main()
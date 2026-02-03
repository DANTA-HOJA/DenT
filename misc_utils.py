import argparse
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import toml
import tomlkit
import torch
from tomlkit.toml_document import TOMLDocument
# -----------------------------------------------------------------------------/


def print_nvidia_smi():
    """ show `nvidia-smi`
    """ 
    # 要執行的命令
    command = "nvidia-smi"
    # 使用Popen執行命令，將stdout捕獲到PIPE中
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 從stdout讀取輸出
    output, error = process.communicate()
    # 將bytes轉換為字符串
    output_str = output.decode('utf-8')
    # 輸出結果
    print(f"{output_str}")
    # -------------------------------------------------------------------------/



def set_reproducibility(seed):
    """ Pytorch reproducibility
        - ref: https://clay-atlas.com/us/blog/2021/08/24/pytorch-en-set-seed-reproduce/?amp=1
        - ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # current GPU
    torch.cuda.manual_seed_all(seed) # all GPUs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # -------------------------------------------------------------------------/



def seed_worker(worker_id):
    """ DataLoader reproducibility
        ref: 'https://pytorch.org/docs/stable/notes/randomness.html'
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # -------------------------------------------------------------------------/



def get_args(mode):
    """ DenT & MitoPrediction input args
    """
    parser = argparse.ArgumentParser(description='DenT-PaperRevision')
    
    """ Datasets Parameters """
    parser.add_argument('--data_path', type=str, required=True, help="root path to data directory")
    parser.add_argument('--workers', type=int, default=4, help="number of data loading workers")
    parser.add_argument('--target_image', type=str, default='target', help='which target to be predicted')
    parser.add_argument('--source_image', type=str, default='source', help='which source to be trained')

    """ Training parameters """
    # parser.add_argument('--gpu', default=0, type=int)
    
    if mode == "train":
        parser.add_argument('--epoch', default=200, type=int, help="number of training iterations")
        parser.add_argument('--val_epoch', default=1, type=int, help="number of validation iterations")
        parser.add_argument('--train_batch', default=2, type=int)
        parser.add_argument('--lr', default=2e-4, type=float)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
    elif mode == "test":
        parser.add_argument('--test_batch', default=1, type=int)

    parser.add_argument('--model', type=str, default='CusDenT')
    parser.add_argument('--image_type', type=str, default='3D')
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=123)

    # DenT-PaperRevision ( reviewer's comment )
    parser.add_argument('--use_multiheads', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--add_pos_emb', action='store_true')
    
    # Unet++ parameters
    parser.add_argument('--deep_supervision', type=bool, default=False)
    # Patches
    parser.add_argument('--patch', type=bool, default=False)
    # KiUnet parameters
    parser.add_argument('--crop', type=bool, default=False)
    
    args = parser.parse_args()
    
    return args
    # -------------------------------------------------------------------------/



def set_args_dirs(args, seed:int, mode:str):
    """
    """
    new_dir_name = f"{args.model}_{args.target_image}_{args.image_type}_{seed}"
    
    if mode == "test":
        if new_dir_name in args.result_dir:
            save_dir = Path(args.result_dir)
        else: raise ValueError("`result_dir` not match with other arguments !!!")
    elif mode == "train":
        found_list = sorted(Path(args.result_dir).joinpath(args.target_image).glob(f"{new_dir_name}_*"),
                            key=lambda x: int(str(x).split(os.sep)[-1].split("_")[-1]))
        
        if len(found_list) > 0:
            temp_list: list = []
            pattern = f"{new_dir_name}_\d+"
            for path in found_list:
                match = re.search(pattern, str(path))
                if match:
                    temp_list.append(match.group(0))
            new_dir_name = f"{new_dir_name}_{int(str(temp_list[-1]).split('_')[-1]) + 1}"
        else:
            new_dir_name = f"{new_dir_name}_1"
        
        save_dir = Path(args.result_dir).joinpath(args.target_image, new_dir_name)
    else:
        raise NotImplementedError("mode accept 'train' or 'test' only")
    
    # args.log_dir
    setattr(args, "log_dir", str(save_dir))
    if not save_dir.exists(): os.makedirs(save_dir)
    
    # args.checkpoints
    temp_path = save_dir.joinpath("checkpoints")
    setattr(args, "checkpoints", str(temp_path))
    if not temp_path.exists(): os.makedirs(temp_path)

    # args.seg_dir
    temp_path = save_dir.joinpath("seg_results")
    setattr(args, "seg_dir", str(temp_path))
    if not temp_path.exists(): os.makedirs(temp_path)
    # -------------------------------------------------------------------------/



def dump_config(path:Path, config:Union[dict, TOMLDocument]):
    """
    """
    with open(path, mode="w") as f_writer:
        tomlkit.dump(config, f_writer)
    # -------------------------------------------------------------------------/



def load_config(path:Path, reserve_comment:bool=False) -> Union[dict, TOMLDocument]:
    """
    """
    if reserve_comment:
        load_fn = tomlkit.load
    else:
        load_fn = toml.load
    
    print(f"Config Path: '{path.resolve()}'\n")
    with open(path, mode="r") as f_reader:
        config = load_fn(f_reader)
    
    return config
    # -------------------------------------------------------------------------/



def get_score_args():
    """ DenT & MitoPrediction input args
    """
    parser = argparse.ArgumentParser(description="Calculate 'IoU' and 'Dice' score")
    
    parser.add_argument('--dent_seg_results', type=str, required=True, help="path to 'seg_results' under each result folder")
    parser.add_argument('--dent_seg_gt', type=str, required=True, help="path to ground-truth of 'DenT' segmentation")
    
    args = parser.parse_args()
    
    return args
    # -------------------------------------------------------------------------/
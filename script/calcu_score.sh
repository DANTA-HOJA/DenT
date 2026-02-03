#!/bin/bash

# module purge
# module load miniconda3
conda activate "DenT-tensorboard-py310"

# variables
target="target" # options: target / target_dna
model="CusDenT" # options: (DenT) DenT, CusDenT(revision)
data_root="../data"
result_root="../results"
record_dir="CusDenT_target_3D_15725_1" # change result dir

python "../calcu_score.py" \
    --dent_seg_results "$result_root/$target/$record_dir/seg_results" \
    --dent_seg_gt "$data_root/DenT/test/$target"
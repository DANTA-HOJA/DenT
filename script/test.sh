#!/bin/bash

# module purge
# module load miniconda3
conda activate "DenT-tensorboard-py310"

# variables
target="target" # options: target / target_dna
model="CusDenT" # options: (DenT-Repo) DenT, CusDenT(revision)
data_root="../data"
result_root="../results"
record_dir="CusDenT_target_3D_15725_1" # change result dir

python "../test.py" \
    --data_path "$data_root/DenT" \
    --target_image "$target" \
    --source_image "source" \
    --test_batch 1 \
    --model "$model" \
    --image_type "3D" \
    --result_dir "$result_root/$target/$record_dir" \
    --random_seed 123 \

    # --deep_supervision \ # Unet++ parameters
    # --crop \ # KiUnet parameters
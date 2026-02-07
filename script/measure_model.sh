#!/bin/bash

# module purge
# module load miniconda3
conda activate "DenT-tensorboard-py310"

# variables
target="target" # options: target / target_dna
model="CusDenT" # options: (DenT-Repo) DenT, CusDenT(revision)
data_root="../data"
result_root="../results"

python "../measure_model.py" \
    --data_path "$data_root/DenT" \
    --target_image "$target" \
    --source_image "source" \
    --epoch 200 \
    --val_epoch 1 \
    --train_batch 2 \
    --lr 2e-4 \
    --weight_decay 5e-4 \
    --model "$model" \
    --image_type "3D" \
    --result_dir "$result_root" \
    --random_seed 123 \
    --use_multiheads 1 1 1 1 \
    --add_pos_emb

    # --deep_supervision \ # Unet++ parameters
    # --crop \ # KiUnet parameters
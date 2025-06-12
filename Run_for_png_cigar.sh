#!/bin/bash

set -e

# <<< 加入这一段激活 conda 环境 >>>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate word
# <<< 以上是关键！ >>>

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"

EXPERIMENT=conformal_0.5_dist_pixel_100_kernel201


MODE="png"
CONCEPT="skull" # Change this to the desired concept
INPUT_PATH="/root/autodl-tmp/DLproject/code/data/png/cigarette" # Change this to the desired SVG file path
SEED=0
COLOR=boolean{True} 

echo "Deforming PNG ${INPUT_PATH} with concept ${CONCEPT} with seed ${SEED}"
ARGS="--mode $MODE --experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --color ${COLOR}"
CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --png_path "${INPUT_PATH}"
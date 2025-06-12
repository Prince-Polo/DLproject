#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"

EXPERIMENT=conformal_0.5_dist_pixel_100_kernel201


MODE="svg"
CONCEPT="dog" # Change this to the desired concept
INPUT_PATH="/root/autodl-tmp/DLproject/code/data/svg/cat" # Change this to the desired SVG file path
SEED=0
COLOR=boolean{True} 
COLOR_PROMPT="paint the dog in a realistic style with natural colors, focusing on the fur texture and details."

# echo "Deforming SVG ${INPUT_PATH} with concept ${CONCEPT} with seed ${SEED}"
# ARGS="--mode $MODE --experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --color ${COLOR} --color_prompt \"$Color_prompt\""
# CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --svg_path "${INPUT_PATH}"

python code/main.py \
  --mode svg \
  --experiment conformal_0.5_dist_pixel_100_kernel201 \
  --seed 0 \
  --use_wandb 0 \
  --wandb_user none \
  --color $COLOR \
  --color_prompt "$COLOR_PROMPT" \
  --semantic_concept dog \
  --svg_path "/root/autodl-tmp/DLproject/code/data/svg/cat"
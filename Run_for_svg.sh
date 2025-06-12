#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"

EXPERIMENT=conformal_0.5_dist_pixel_100_kernel201


MODE="svg"
CONCEPT="DOG" # Change this to the desired concept
INPUT_PATH="/root/autodl-tmp/git_resp/DLproject/code/data/svg/svgviewer-output" # Change this to the desired SVG file path
SEED=0

echo "Deforming SVG ${INPUT_PATH} with concept ${CONCEPT} with seed ${SEED}"
ARGS="--mode $MODE --experiment $EXPERIMENT --seed $SEED --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER}"
CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${CONCEPT}" --svg_path "${INPUT_PATH}"
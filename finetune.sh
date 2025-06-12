export MODEL_NAME="/root/autodl-tmp/model/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14" # 选用的主模型
export OUTPUT_DIR="/root/autodl-tmp/ft_model/vector" # LoRA模型保存地址
export DATASET_NAME="/root/autodl-tmp/ft/output_pngs" # 训练数据路径

accelerate launch --mixed_precision="no"  /root/autodl-tmp/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=5000 \
  --seed=2048
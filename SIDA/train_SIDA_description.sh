
deepspeed --include localhost:1 --master_port=24999 train_SIDA_description.py \
  --version="./ck/SIDA-7B" \
  --dataset_dir='/path_to/text_label_images/' \
  --vision_pretrained="./ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/path_to/text_label_images/"\
  --batch_size=2 \
  --exp_name="SIDA-7B-description" \
  --epochs=5 \
  --steps_per_epoch=100 \

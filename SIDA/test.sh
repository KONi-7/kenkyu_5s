python test.py \
  --version="./ck/SIDA-7B" \
  --dataset_dir='./' \
  --vision_pretrained="./ck/sam_vit_h_4b8939.pth" \
  --test_dataset="./" \
  --precision fp16 \
  --test_batch_size 1 \
  --test_only 

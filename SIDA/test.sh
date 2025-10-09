deepspeed --include localhost: --master_port=24999 test.py \
  --version="./ck/SIDA-7B" \
  --dataset_dir='/path/to/benchmark/' \
  --vision_pretrained="./ck/sam_vit_h_4b8939.pth" \
  --test_dataset="/path/to/benchmark/"\
  --test_only 

import argparse
import os
import sys
import time     #追加
import torch.nn as nn  # 追加

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from thop import profile #追加
import torch.profiler  # 追加

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


#追加
KEEP_TOKEN_RATIO = 0.3  # 残す割合（必要に応じて調整）

def aggregate_token_scores(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return attention scores from the newest token back to the prompt tokens."""
    # attentions: tuple[layer] -> (batch, num_heads, seq_len, seq_len)
    last_layer = attentions[-1].mean(dim=1)  # (batch, seq_len, seq_len)
    # Focus on the attention distribution of the newly generated token (last row) toward previous tokens.
    return last_layer[:, -1, :-1].contiguous()  # drop self token

def select_salient_tokens(input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          attentions: tuple[torch.Tensor, ...],
                          keep_ratio: float,
                          tokenizer) -> torch.Tensor:
    scores = aggregate_token_scores(attentions)[0]  # (seq_len - 1,)
    scores = scores / (scores.sum() + 1e-6)

    full_seq_len = input_ids.size(1)
    device = input_ids.device

    if scores.size(0) != full_seq_len - 1:
        target = full_seq_len - 1
        if scores.size(0) < target:
            pad = target - scores.size(0)
            scores = F.pad(scores, (0, pad), value=0.0)
        else:
            scores = scores[:target]

    keep_quota = max(int(scores.size(0) * keep_ratio), 0)

    raw_special_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
    flat_special_ids: list[int] = []
    for tok in raw_special_tokens:
        if tok is None:
            continue
        if isinstance(tok, (list, tuple)):
            flat_special_ids.extend(tok)
        else:
            flat_special_ids.append(tok)
    special_ids = (torch.tensor(flat_special_ids, device=device, dtype=input_ids.dtype)
                   if flat_special_ids else torch.empty(0, device=device, dtype=input_ids.dtype))

    prompt_tokens = input_ids[0, :full_seq_len - 1]
    mandatory_mask = torch.zeros_like(scores, dtype=torch.bool)
    mandatory_mask[0] = True  # keep BOS
    if special_ids.numel() > 0:
        mandatory_mask |= torch.isin(prompt_tokens, special_ids)
    mandatory_mask |= prompt_tokens == IMAGE_TOKEN_INDEX

    keep_indices = torch.nonzero(mandatory_mask, as_tuple=False).squeeze(1)

    remaining_quota = max(keep_quota - keep_indices.numel(), 0)
    available_slots = scores.size(0) - keep_indices.numel()
    if remaining_quota > 0 and available_slots > 0:
        candidate_scores = scores.clone()
        candidate_scores[mandatory_mask] = float('-inf')
        k = min(remaining_quota, available_slots)
        if k > 0:
            top_indices = torch.topk(candidate_scores, k=k).indices
            keep_indices = torch.cat([keep_indices, top_indices])

    keep_indices = torch.cat([torch.tensor([0], device=device), keep_indices])  # ensure BOS
    keep_indices = keep_indices.unique()
    keep_indices, _ = torch.sort(keep_indices)

    keep_indices_full = torch.cat([keep_indices, torch.tensor([full_seq_len - 1], device=device)])
    keep_indices_full = keep_indices_full.unique()
    keep_indices_full, _ = torch.sort(keep_indices_full)

    return keep_indices_full
#追加)




def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA chat")
    parser.add_argument("--version", default="SIDA-7B-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument("--measure_flops", action="store_true", default=False)#追加

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = SIDAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, cls_token_idx=args.cls_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Skip DeepSpeed initialization for now
    if torch.cuda.is_available():
        model = model.cuda()

    print("Before vision tower initialization")
    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
    except AttributeError:
        print("Vision tower initialization skipped as SIDA-7B-v1 may not have this module.")

    print("Before precision setting")
    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    print("Before clip_image_processor")
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    print("Before model.eval()")
    model.eval()
    print("Model loaded successfully")

    #追加
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    else:
        model = model.cuda()
    #追加)

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        #追加
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).cuda()
        sida_model = model.module if isinstance(model, nn.DataParallel) else model

        with torch.inference_mode():
            warm_outputs = sida_model.generate(
                images=image_clip,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        attentions = warm_outputs.attentions
        if attentions is None or len(attentions) == 0:
            keep_indices = torch.arange(input_ids.size(1), device=input_ids.device)
        else:
            attn_layers = attentions[-1] if isinstance(attentions[0], tuple) else attentions
            attn_layers = tuple(attn_layers)
            keep_indices = select_salient_tokens(
                input_ids,
                attention_mask,
                attn_layers,
                KEEP_TOKEN_RATIO,
                tokenizer,
            )

        input_ids = input_ids[:, keep_indices]
        attention_mask = attention_mask[:, keep_indices]

        # 計算コスト測定開始
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        
                # 推論（プロファイルなし／ありを切り替え）
        if args.measure_flops:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                profile_memory=False,
                record_shapes=False,
            ) as prof:
                output_ids, pred_masks = model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
            total_flops = sum(ev.flops for ev in prof.key_averages() if ev.flops)
            print(f"Total FLOPs: {total_flops / 1e12:.2f} TFLOPs")
        else:
            output_ids, pred_masks = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )

                # 計算コスト測定終了
        end_time = time.time()
        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)


        # FLOPs計算をここに移動（推論後）
        try:
            llama = model.model  # LlamaModel
            llama.eval()
            dummy_input_ids = torch.randint(
                0, tokenizer.vocab_size, (1, 10), device=llama.device
            )

            flops, params = profile(
                llama,
                inputs=(dummy_input_ids,),
                verbose=False,
            )
            print(f"Language FLOPs: {flops / 1e12:.2f} TFLOPs")
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")

        # コスト表示
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Peak GPU Memory: {peak_memory:.2f} GB")
        #追加)


        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])

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
                         DEFAULT_IMAGE_TOKEN)
from utils.pruning import prune_batch_inputs




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
    parser.add_argument(
        "--vision_pretrained",
        default="PATH_TO_SAM_ViT-H",
        type=str,
        help="Path to pretrained SAM ViT-H checkpoint",
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

    parser.add_argument(
        "--prune_keep_ratio",
        type=float,
        default=0.3,
        help="Fraction of image tokens to keep after observe-layer pruning",
    )
    parser.add_argument(
        "--prune_observe_layer",
        type=int,
        default=-24,
        help="Layer index (supports negative indexing) to stop observe phase",
    )
    parser.add_argument(
        "--disable_token_pruning",
        action="store_true",
        help="Skip token pruning and run full prompt through all layers",
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


def _pad_token_mask(mask: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    left = torch.zeros((mask.shape[0], pad_left), dtype=torch.bool, device=mask.device)
    right = torch.zeros((mask.shape[0], pad_right), dtype=torch.bool, device=mask.device)
    return torch.cat([left, mask, right], dim=1)


def run_pruned_inference(
    sida_model,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    image: torch.Tensor,
    resize_list,
    original_size_list,
):
    device = input_ids.device
    batch = input_ids.shape[0]

    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.detach().clone().contiguous()

    def _align_mask_length(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        current_len = mask.shape[1]
        if current_len == target_len:
            return mask
        if current_len > target_len:
            return mask[:, current_len - target_len :]
        pad = torch.zeros(
            (mask.shape[0], target_len - current_len),
            dtype=mask.dtype,
            device=mask.device,
        )
        return torch.cat([pad, mask], dim=1)

    def _match_mask_to_logits(mask: torch.Tensor, logits_flat: torch.Tensor) -> torch.Tensor:
        target_len = logits_flat.shape[0]
        mask_flat = mask.reshape(-1)
        current_len = mask_flat.shape[0]
        if current_len == target_len:
            return mask_flat
        if current_len > target_len:
            return mask_flat[current_len - target_len :]
        pad = torch.zeros(
            target_len - current_len,
            dtype=mask_flat.dtype,
            device=mask_flat.device,
        )
        return torch.cat([pad, mask_flat], dim=0)

    cls_token_mask = (input_ids[:, 1:] == sida_model.cls_token_idx)
    cls_token_mask = torch.cat(
        [
            cls_token_mask,
            torch.zeros((batch, 1), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    cls_token_mask = _pad_token_mask(cls_token_mask, pad_left=255, pad_right=0)
    cls_token_mask = _align_mask_length(cls_token_mask, hidden_states.shape[1])

    last_hidden_state_cls = sida_model.model.cls_head[0](hidden_states)
    cls_logits_flat = last_hidden_state_cls.view(-1, last_hidden_state_cls.shape[-1])
    cls_mask_flat = _match_mask_to_logits(cls_token_mask, cls_logits_flat)
    cls_result = cls_logits_flat[cls_mask_flat]

    predicted_class = 0
    if cls_result.shape[0] > 0:
        predicted_class = torch.argmax(cls_result[-1], dim=-1).item()

    pred_masks = []
    if predicted_class == 2:
        seg_token_mask = (input_ids[:, 1:] == sida_model.seg_token_idx)
        seg_token_mask = _pad_token_mask(seg_token_mask, pad_left=255, pad_right=1)
        seg_token_mask = _align_mask_length(seg_token_mask, hidden_states.shape[1])

        if seg_token_mask.any():
            hidden_proj = sida_model.model.text_hidden_fcs[0](hidden_states)
            last_hidden_state = hidden_proj

            last_hidden_state_flat = last_hidden_state.reshape(
                -1, last_hidden_state.shape[-1]
            )
            seg_mask_flat = _match_mask_to_logits(seg_token_mask, last_hidden_state_flat)
            pred_embeddings = last_hidden_state_flat[seg_mask_flat]

            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    seg_token_offset,
                ],
                dim=0,
            )
            offset = torch.arange(0, batch + 1, dtype=torch.long, device=device)
            seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])

            pred_embeddings = pred_embeddings_

            cls_projected = sida_model.model.sida_fc1(cls_result)
            enhanced_pred_embeddings = []
            for i in range(len(pred_embeddings)):
                if pred_embeddings[i].shape[0] == 0:
                    enhanced_pred_embeddings.append(pred_embeddings[i])
                    continue
                query = cls_projected[i].unsqueeze(0)
                key = pred_embeddings[i]
                value = pred_embeddings[i]
                attn_output, _ = sida_model.model.attention_layer(
                    query=query, key=key, value=value
                )
                enhanced_embeddings = pred_embeddings[i] + attn_output
                enhanced_pred_embeddings.append(enhanced_embeddings)

            image_embeddings = sida_model.get_visual_embs(image)
            multimask_output = False

            for i in range(len(enhanced_pred_embeddings)):
                if enhanced_pred_embeddings[i].shape[0] == 0:
                    continue
                sparse_embeddings, dense_embeddings = sida_model.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=enhanced_pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(
                    enhanced_pred_embeddings[i].dtype
                )
                low_res_masks, _ = sida_model.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sida_model.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                original_size = (
                    original_size_list[i]
                    if original_size_list is not None
                    else resize_list[i]
                )
                pred_mask = sida_model.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size,
                )
                pred_masks.append(pred_mask[:, 0])

    return {
        "predicted_class": predicted_class,
        "logits": cls_result,
        "pred_masks": pred_masks,
    }
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
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        cls_token_idx=args.cls_token_idx,
        vision_pretrained=args.vision_pretrained,
        **kwargs,
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

        sida_model = model.module if isinstance(model, nn.DataParallel) else model

        special_token_ids = [
            tok
            for tok in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
                args.cls_token_idx,
                args.seg_token_idx,
            ]
            if tok is not None
        ]

        def full_forward_pass():
            local_input_ids = tokenizer_image_token(
                prompt, tokenizer, return_tensors="pt"
            ).unsqueeze(0).cuda()
            local_attention_mask = torch.ones_like(
                local_input_ids, dtype=torch.long
            ).cuda()

            input_dict = {
                "input_ids": local_input_ids,
                "attention_masks": local_attention_mask,
                "images_clip": image_clip,
            }

            if not args.disable_token_pruning:
                input_dict = prune_batch_inputs(
                    model,
                    input_dict,
                    tokenizer,
                    keep_ratio=args.prune_keep_ratio,
                    observe_layer=args.prune_observe_layer,
                    special_token_ids=special_token_ids,
                )
            else:
                input_dict.setdefault(
                    "keep_indices",
                    [torch.arange(local_input_ids.shape[1], device=local_input_ids.device)],
                )

            pruned_input_ids = input_dict["input_ids"].to(local_input_ids.device)
            pruned_attention = input_dict["attention_masks"].to(local_attention_mask.device)
            cached_hidden_states = input_dict.get("cached_hidden_states")

            if cached_hidden_states is None:
                with torch.inference_mode():
                    fallback = sida_model(
                        input_ids=pruned_input_ids,
                        attention_mask=pruned_attention,
                        images=image_clip,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                cached_hidden_states = fallback.hidden_states[-1].detach()

            return run_pruned_inference(
                sida_model=sida_model,
                hidden_states=cached_hidden_states,
                input_ids=pruned_input_ids,
                image=image,
                resize_list=resize_list,
                original_size_list=original_size_list,
            )

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        prof_total_flops = None
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
                inference_outputs = full_forward_pass()
            prof_total_flops = sum(ev.flops for ev in prof.key_averages() if ev.flops)
        else:
            inference_outputs = full_forward_pass()

        end_time = time.time()
        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        predicted_class = inference_outputs["predicted_class"]
        pred_masks = inference_outputs["pred_masks"]


        # FLOPs計算をここに移動（推論後）
        language_flops = None
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
            language_flops = flops
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")

        if prof_total_flops is not None:
            print(f"Total FLOPs: {prof_total_flops / 1e12:.2f} TFLOPs")
        else:
            print("Total FLOPs: unavailable (run with --measure_flops to profile end-to-end)")

        if language_flops is not None:
            print(f"Language FLOPs: {language_flops / 1e12:.2f} TFLOPs")

        # コスト表示
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Peak GPU Memory: {peak_memory:.2f} GB")
        #追加)

        class_responses = {
            0: "[CLS] This image is classified as real. It shows no signs of tampering or synthesis.",
            1: "[CLS] This image is classified as full synthetic. It appears entirely artificially generated.",
            2: "[CLS] This image is classified as tampered. It has been altered. [SEG] A mask highlighting the tampered region is provided.",
        }
        text_output = class_responses.get(
            predicted_class, "[CLS] Unable to determine the manipulation status."
        )
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

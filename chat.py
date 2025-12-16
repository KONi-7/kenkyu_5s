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
import torch.profiler  # 追加

from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from utils.pruning import prune_batch_inputs


def _extract_last_layer_hidden_states(fallback_output) -> torch.Tensor:
    """Extract last-layer hidden states from various HF/custom output shapes."""
    hidden_states = getattr(fallback_output, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Fallback forward did not return hidden_states")

    # Some implementations return a tuple/list of per-layer tensors.
    if isinstance(hidden_states, (tuple, list)):
        if len(hidden_states) == 0:
            raise RuntimeError("Fallback forward returned empty hidden_states")
        last = hidden_states[-1]
        if not isinstance(last, torch.Tensor):
            raise RuntimeError(
                f"Unexpected hidden_states[-1] type: {type(last)}"
            )
        return last

    # Other implementations may return a single tensor (already last hidden).
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states

    raise RuntimeError(f"Unexpected hidden_states type: {type(hidden_states)}")


def _sum_profiled_flops(prof: torch.profiler.profile) -> int:
    """Sum FLOPs reported by torch.profiler (may be partial depending on kernels)."""
    total_flops = 0
    for ev in prof.key_averages():
        f = getattr(ev, "flops", None)
        if f:
            total_flops += int(f)
    return total_flops


def _format_flops(flops: int) -> str:
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    x = float(flops)
    for u in units:
        if x < 1000.0:
            return f"{x:.2f} {u}"
        x /= 1000.0
    return f"{x:.2f} EFLOPs"


def _get_cuda_device_ids_for_peak(model: nn.Module, device: torch.device) -> list[int]:
    """Return CUDA device ids to consider when reporting peak GPU memory."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    if isinstance(model, nn.DataParallel):
        # DataParallel keeps a replica on each device_id.
        return list(getattr(model, "device_ids", []) or [torch.cuda.current_device()])
    return [torch.cuda.current_device()]


def _reset_peak_cuda_memory_stats(device_ids: list[int]) -> None:
    for dev in device_ids:
        try:
            torch.cuda.reset_peak_memory_stats(dev)
        except Exception:
            # Best-effort: if CUDA isn't fully initialized for a device, skip.
            pass


def _peak_cuda_memory_gb(device_ids: list[int]) -> float:
    if not device_ids:
        return 0.0
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    peaks = []
    for dev in device_ids:
        try:
            peaks.append(float(torch.cuda.max_memory_allocated(dev)))
        except Exception:
            pass
    if not peaks:
        return 0.0
    # Report the max across devices (single headline number).
    return max(peaks) / (1024 ** 3)




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
        default=1.0,
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
    last_layer_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    image: torch.Tensor,
    resize_list,
    original_size_list,
):
    device = input_ids.device
    bs = last_layer_hidden.shape[0]
    seq_len_out = last_layer_hidden.shape[1]
    seq_len_in = input_ids.shape[1]
    delta = seq_len_out - seq_len_in

    # Find [CLS] position in input_ids, then shift by delta if it occurs after the image token.
    cls_token_mask_in = (input_ids == sida_model.cls_token_idx)
    has_cls = cls_token_mask_in.any(dim=1)
    cls_pos = torch.zeros(bs, dtype=torch.long, device=device)
    if has_cls.any():
        cls_pos[has_cls] = cls_token_mask_in[has_cls].int().argmax(dim=1)

    if delta != 0:
        img_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        has_img = img_token_mask.any(dim=1)
        img_pos = torch.zeros(bs, dtype=torch.long, device=device)
        if has_img.any():
            img_pos[has_img] = img_token_mask[has_img].int().argmax(dim=1)
        shift = (cls_pos > img_pos) & has_img
        cls_pos = cls_pos + shift.long() * delta

    cls_pos = cls_pos.clamp_(0, seq_len_out - 1)
    last_hidden_state_cls = sida_model.model.cls_head[0](last_layer_hidden)
    cls_result = last_hidden_state_cls[
        torch.arange(bs, device=device), cls_pos, :
    ]

    # Predicted class from the last [CLS] (batch-size assumed 1 for interactive chat).
    predicted_class = int(torch.argmax(cls_result[0], dim=-1).item())

    pred_masks = []
    if predicted_class == 2:
        seg_token_mask_in = (input_ids == sida_model.seg_token_idx)
        if seg_token_mask_in.any():
            # Map [SEG] token positions from input token space -> expanded hidden-state space.
            img_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
            has_img = img_token_mask.any(dim=1)
            img_pos = torch.zeros(bs, dtype=torch.long, device=device)
            if delta != 0 and has_img.any():
                img_pos[has_img] = img_token_mask[has_img].int().argmax(dim=1)

            hidden_proj = sida_model.model.text_hidden_fcs[0](last_layer_hidden)
            per_sample_seg_embeds = []
            for b in range(bs):
                seg_pos_in = torch.nonzero(seg_token_mask_in[b], as_tuple=False).squeeze(1)
                if seg_pos_in.numel() == 0:
                    per_sample_seg_embeds.append(
                        torch.zeros((0, hidden_proj.shape[-1]), device=device, dtype=hidden_proj.dtype)
                    )
                    continue
                seg_pos_out = seg_pos_in
                if delta != 0 and has_img[b]:
                    seg_pos_out = seg_pos_in + (seg_pos_in > img_pos[b]).long() * delta
                seg_pos_out = seg_pos_out.clamp(0, seq_len_out - 1)
                per_sample_seg_embeds.append(hidden_proj[b, seg_pos_out, :])

            # Interactive chat uses batch size 1; keep list-of-tensors structure.
            pred_embeddings = per_sample_seg_embeds

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
                sparse_embeddings = sparse_embeddings.to(enhanced_pred_embeddings[i].dtype)
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

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
    )
    gpu_fallback_reason = None

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

    # Precision policy:
    # - On CUDA: honor args.precision.
    # - On CPU: force float32 (many ops, e.g. CLIP conv2d, don't support fp16/bf16 on CPU).
    torch_dtype = torch.float32
    if device.type == "cuda":
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

    # Move model to device.
    # NOTE: If this fails (e.g., OOM), we fall back to CPU but keep a clear reason so
    # Peak GPU Memory doesn't misleadingly show 0.00 GB as if it were measured.
    try:
        model = model.to(device)
    except Exception as exc:
        print(f"Failed to move model to {device}: {exc}")
        gpu_fallback_reason = str(exc)
        device = torch.device("cpu")
        model = model.to(device)

    cpu_forced_fp32 = device.type != "cuda"

    print("Before vision tower initialization")
    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=(torch.float32 if cpu_forced_fp32 else torch_dtype), device=device)
    except AttributeError:
        print("Vision tower initialization skipped as SIDA-7B-v1 may not have this module.")

    print("Before precision setting")
    # Avoid dtype casting for quantized models.
    if not (args.load_in_4bit or args.load_in_8bit):
        if cpu_forced_fp32:
            model = model.float()
        else:
            if args.precision == "bf16":
                model = model.bfloat16()
            elif args.precision == "fp16":
                model = model.half()
            else:
                model = model.float()
    model = model.to(device)

    print("Before clip_image_processor")
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    print("Before model.eval()")
    model.eval()
    print("Model loaded successfully")

    #追加
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    #追加)

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        try:
            prompt = input("Please input your prompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        try:
            image_path = input("Please input the image path: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
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
            .to(device)
        )
        if cpu_forced_fp32:
            image_clip = image_clip.float()
        else:
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
            .to(device)
        )
        if cpu_forced_fp32:
            image = image.float()
        else:
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(device)
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

        base_input_dict = {
            "input_ids": input_ids,
            "attention_masks": attention_mask,
            "images_clip": image_clip,
        }

        do_prune = (not args.disable_token_pruning) and (float(args.prune_keep_ratio) < 1.0)

        def execute_inference():
            """Run the *full* inference compute (optionally including pruning observe pass).

            FLOPs scope policy:
            - Includes: token pruning observe+selection (if enabled), LLaVA forward to obtain
              last-layer hidden states (if not provided by pruning), and SIDA heads + SAM mask decoder.
            - Excludes: image loading/decoding, prompt text I/O, and CPU-side preprocessing.
            """
            with torch.no_grad():
                working = dict(base_input_dict)
                if do_prune:
                    working = prune_batch_inputs(
                        model,
                        working,
                        tokenizer,
                        keep_ratio=args.prune_keep_ratio,
                        observe_layer=args.prune_observe_layer,
                        special_token_ids=special_token_ids,
                    )

                run_input_ids = working["input_ids"].to(device)
                run_attention_mask = working["attention_masks"].to(device)
                cached_hidden_states = working.get("cached_hidden_states")

                if cached_hidden_states is None:
                    # No pruning (or pruning path couldn't provide cached hidden states):
                    # run the underlying LLaVA/LLaMA forward to get last-layer hidden states.
                    fallback = LlavaLlamaForCausalLM.forward(
                        sida_model,
                        input_ids=run_input_ids,
                        attention_mask=run_attention_mask,
                        images=image_clip,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    cached_hidden_states = _extract_last_layer_hidden_states(fallback).detach()

                # Normalize hidden states shape to batch-first (B, seq, hidden).
                if not isinstance(cached_hidden_states, torch.Tensor):
                    raise RuntimeError(
                        f"cached_hidden_states must be a Tensor, got {type(cached_hidden_states)}"
                    )
                if cached_hidden_states.dim() == 2:
                    cached_hidden_states = cached_hidden_states.unsqueeze(0)
                elif cached_hidden_states.dim() == 4:
                    # Some models might return (layers, B, seq, hidden); take the last layer.
                    cached_hidden_states = cached_hidden_states[-1]
                if cached_hidden_states.dim() != 3:
                    raise RuntimeError(
                        f"Unexpected cached_hidden_states shape: {tuple(cached_hidden_states.shape)}"
                    )

                return run_pruned_inference(
                    sida_model=sida_model,
                    last_layer_hidden=cached_hidden_states,
                    input_ids=run_input_ids,
                    image=image,
                    resize_list=resize_list,
                    original_size_list=original_size_list,
                )

        cuda_device_ids = _get_cuda_device_ids_for_peak(model, device)
        if device.type == "cuda" and cuda_device_ids:
            _reset_peak_cuda_memory_stats(cuda_device_ids)
        start_time = time.time()

        if args.measure_flops:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda" and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                profile_memory=False,
                record_shapes=False,
            ) as prof:
                inference_outputs = execute_inference()
            total_flops = _sum_profiled_flops(prof)
            print(f"Total FLOPs: {_format_flops(total_flops)}")
        else:
            inference_outputs = execute_inference()

        end_time = time.time()
        inference_time = end_time - start_time
        if device.type == "cuda" and cuda_device_ids:
            peak_memory = _peak_cuda_memory_gb(cuda_device_ids)
            peak_memory_text = f"{peak_memory:.2f} GB"
        else:
            peak_memory_text = "N/A (ran on CPU)" if gpu_fallback_reason else "N/A"

        predicted_class = inference_outputs["predicted_class"]
        pred_masks = inference_outputs["pred_masks"]
        # コスト表示
        print(f"Inference Time: {inference_time:.4f} seconds")
        print(f"Peak GPU Memory: {peak_memory_text}")
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

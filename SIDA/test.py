import argparse
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# import deepspeed
import numpy as np
import torch
import tqdm
import transformers
# from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from utils.SID_Set import collate_fn, CustomDataset
from utils.batch_sampler import BatchSampler
import torch.distributed as dist
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
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
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--test_dataset", default="test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="sida", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--sample_ratio", default=None, type=float,
                        help="Fraction of test batches to sample for quick evaluation (0-1)")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of classes for classification in stage 1")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True,
                   help="Whether to use Stage 1 CLS token in Stage 2")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--no_test", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    return parser.parse_args(args)
def main(args):
    args = parse_args(args)
    if not args.test_only:
        # deepspeed.init_distributed()
        pass
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_token = tokenizer.add_tokens("[CLS]")
    num_added_token = tokenizer.add_tokens("[SEG]")
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = SIDAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print("\nChecking specific components:")
    for component in [ "cls_head", "sida_fc1", "attention_layer", "text_hidden_fcs"]:
        matching_params = [n for n, _ in model.named_parameters() if component in n]
        if matching_params:
            print(f"Found {component} in parameters: {matching_params}")
        else:
            print(f"Component not found: {component}")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.test_only:
        model.get_model().initialize_sida_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False


    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                                "cls_head",
                                "sida_fc1",
                                "attention_layer",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        # lora_alpha = args.lora_alpha
        # lora_dropout = args.lora_dropout
        # lora_target_modules = find_linear_layers(
        #         model, args.lora_target_modules.split(",")
        # )
        # lora_config = LoraConfig(
        #     r=lora_r,
        #     lora_alpha=lora_alpha,
        #     target_modules=lora_target_modules,
        #     lora_dropout=lora_dropout,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model, lora_config)
        # model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["embed_tokens", "mask_decoder", "text_hidden_fcs","cls_head", "sida_fc1","attention_layer"]
            ]
        ):
            p.requires_grad = True

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    if not args.test_only:
        train_dataset = CustomDataset(
            base_image_dir=args.dataset_dir,  
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,  
            split="train", 
            precision=args.precision,  
            image_size=args.image_size, 

        )
        print(f"\nInitializing datasets:")
        print(f"Training split size: {len(train_dataset)}")

    if args.no_test == False:
        test_dataset = CustomDataset(
            base_image_dir=args.dataset_dir, 
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,  
            split="test", 
            precision=args.precision, 
            image_size=args.image_size, 
    )
        if not args.test_only:
            print(
                f"Training with {len(train_dataset)} examples and testing with {len(test_dataset)} examples."
            )
        else:
            print(f"Testing with {len(test_dataset)} examples.")
    else:
        test_dataset = None
        if not args.test_only:
            print(f"Training with {len(train_dataset)} examples.")
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,  
            "initial_scale_power": 12,  
            "loss_scale_window": 1000,
            "min_loss_scale": 1,
            "hysteresis": 2
        },
        "gradient_clipping": 1.0,
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    if not args.test_only:
        batch_sampler = BatchSampler(
            dataset=train_dataset,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
                cls_token_idx=args.cls_token_idx,
            ),
        )
        # model_engine, optimizer, _, scheduler = deepspeed.initialize(
        #     model=model,
        #     model_parameters=model.parameters(),
        #     config=ds_config,
        #     training_data=None, 
        # )
        model_engine = model.cuda()
    else:
        model_engine = model.cuda()

    # if args.auto_resume and len(args.resume) == 0:
    #     resume = os.path.join(args.log_dir,  "ckpt_model")
    #     if os.path.exists(resume):
    #         args.resume = resume

    # if args.resume:
    #     load_path, client_state = model_engine.load_checkpoint(args.resume)
    #     with open(os.path.join(args.resume, "latest"), "r") as f:
    #         ckpt_dir = f.readlines()[0].strip()
    #     args.start_epoch = (
    #         int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
    #     )
    #     print(
    #         "resume training from {}, start from epoch {}".format(
    #         args.resume, args.start_epoch
    #     )
    # )

    if test_dataset is not None:
        test_sampler = BatchSampler(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                 collate_fn,
                 tokenizer=tokenizer,
                 conv_type=args.conv_type,
                 use_mm_start_end=args.use_mm_start_end,
                 local_rank=args.local_rank,
             ),
        )

    if not args.test_only:
        train_iter = iter(train_loader)

    best_acc, best_score, cur_ciou = 0.0, 0.0, 0.0

    if args.test_only:
        acc, giou, ciou, _ = test(test_loader, model_engine, 0, writer, args, sample_ratio=args.sample_ratio) 
        exit()

    test_epochs = [1,3,5,7,10]
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"test will be performed after epochs: {test_epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        if (epoch + 1) in test_epochs: 
            if args.local_rank == 0:
                print(f"\nPerforming test after epoch {epoch + 1}")

            if args.no_test == False:
                acc, giou, ciou, _ = test(test_loader, model_engine, epoch, writer, args, sample_ratio=args.sample_ratio)
                best_score = max(giou, best_score)
                is_best_iou = giou > best_score
                cur_ciou = ciou if is_best_iou else cur_ciou
                is_best_acc = acc > best_acc
                best_acc = max(acc, best_acc)
                cur_acc = acc if is_best_acc else cur_acc
                is_best = is_best_iou or is_best_acc

            if args.local_rank == 0:
                print(f"Current accuracy: {acc:.2f}%, Best accuracy: {best_acc:.2f}%")
                print(f"Current iou: {cur_ciou:.2f}%, Best score: {best_score:.2f}%")
            if args.no_test or is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                                {"epoch": epoch},
                                os.path.join(
                                    args.log_dir,
                                    f"meta_log_acc{best_acc:.3f}_iou{best_score:.3f}.pth"
                                ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping test.")

        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, cls_losses, mask_bce_losses, mask_dice_losses, mask_losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        model.zero_grad()
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()
            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            cls_loss = output_dict["cls_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            losses.update(loss.item(), input_dict["images"].size(0))
            cls_losses.update(cls_loss.item(), input_dict["images"].size(0))
            if input_dict['cls_labels'][0] == 2:
                mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
                mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
                mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                cls_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/cls_loss", cls_losses.avg, global_step)
                writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, global_step)
                writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, global_step)
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            cls_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter
import random

def test(test_loader, model_engine, epoch, writer, args, sample_ratio=None):
    model_engine.eval()
    correct = 0
    total = 0
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes, device='cuda')
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    target_meter = AverageMeter("Target", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    # Calculate total number of batches and samples to use
    total_batches = len(test_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        # Generate random indices for sampling
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\ntest on {num_batches}/{total_batches} randomly sampled batches...")

    # Lists for AUC and PR computation (pixel-wise) for tampered masks
    all_mask_scores = []
    all_mask_labels = []

    for batch_idx, input_dict in enumerate(tqdm.tqdm(test_loader)):
        # Skip batches not in our sample if sampling is enabled
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue
        if batch_idx == 0:
            print("\nFirst test batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        # Debug first processed batch
        if total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", input_dict.keys())

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # Get predictions
        logits = output_dict["logits"]
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        cls_labels = input_dict["cls_labels"]
        correct += (preds == cls_labels).sum().item()
        total += cls_labels.size(0)

        for t, p in zip(cls_labels, preds):
            confusion_matrix[t.long(), p.long()] += 1
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            area_target_total = None
            for mask_idx, (mask_i, output_i) in enumerate(zip(masks_list, output_list)):
                # compute intersection/union
                intersection_i, union_i, area_target_i = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target

                # collect pixel-wise probabilities and labels for AUC/PR computation
                # pred_masks[0] shape: [num_masks, H, W]
                try:
                    pred_score_map = pred_masks[0][mask_idx]
                except Exception:
                    # fallback: use the binary output if indexing fails
                    pred_score_map = output_i.float()
                # Convert logits to probabilities via sigmoid for ranking metrics
                pred_prob_map = torch.sigmoid(pred_score_map).detach().cpu().numpy().reshape(-1)
                gt_map = mask_i.detach().cpu().numpy().reshape(-1)
                all_mask_scores.append(pred_prob_map)
                all_mask_labels.append(gt_map)
                # accumulate area_target per class
                try:
                    area_np = area_target_i.detach().cpu().numpy()
                except Exception:
                    try:
                        area_np = np.array(area_target_i)
                    except Exception:
                        area_np = None
                if area_np is not None:
                    if area_target_total is None:
                        area_target_total = area_np
                    else:
                        area_target_total = area_target_total + area_np
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            # area_target_i was computed inside the loop; but we only need per-mask values aggregated
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection)
            union_meter.update(union)
            # area_target has been accumulated and is a tensor/array
            if area_target_total is not None:
                target_meter.update(area_target_total)
            else:
                # no gt masks present in this sample
                target_meter.update(np.array([0.0, 0.0]))
            acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    if not args.test_only:
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1] if len(iou_class) > 1 else 0.0
    giou = acc_iou_meter.avg[1] if len(acc_iou_meter.avg) > 1 else 0.0

    accuracy = correct / total * 100.0
    confusion_matrix = confusion_matrix.cpu()
    class_names = ['Real', 'Full Synthetic', 'Tampered']
    per_class_metrics = {}
    for i in range(num_classes):
        tp = confusion_matrix[i, i]  
        fp = confusion_matrix[:, i].sum() - tp  
        fn = confusion_matrix[i, :].sum() - tp 
        tn = confusion_matrix.sum() - (tp + fp + fn)  

        total_class_samples = confusion_matrix[i, :].sum()

        class_accuracy = float(tp / total_class_samples) if total_class_samples > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[class_names[i]] = {
            'accuracy': class_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    pixel_correct = intersection_meter.sum[1]  
    # total positive pixels in GT
    pixel_gt_total = target_meter.sum[1] if len(target_meter.sum) > 1 else target_meter.sum
    # total predicted positive (can be derived)
    pixel_pred_total = union_meter.sum[1] + intersection_meter.sum[1] - target_meter.sum[1]
    # compute pixel accuracy (TP + TN) / total
    # total pixels = sum of target counts for all classes
    total_pixels = target_meter.sum.sum() if isinstance(target_meter.sum, np.ndarray) else target_meter.sum
    if total_pixels == 0:
        pixel_accuracy = 0.0
    else:
        tp = intersection_meter.sum[1]
        fp = union_meter.sum[1] - target_meter.sum[1]
        fn = target_meter.sum[1] - intersection_meter.sum[1]
        tn = total_pixels - (tp + fp + fn)
        pixel_accuracy = (tp + tn) / (total_pixels + 1e-10) * 100.0

    iou = ciou  
    # compute segmentation F1 from aggregated pixel counts
    if target_meter.sum[1] + union_meter.sum[1] + intersection_meter.sum[1] == 0:
        seg_f1 = 0.0
    else:
        tp = intersection_meter.sum[1]
        fp = union_meter.sum[1] - target_meter.sum[1]
        fn = target_meter.sum[1] - intersection_meter.sum[1]
        seg_f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-10)

    avg_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])

    # Compute AUC / Average Precision for mask localization using collected scores
    auc_roc = None
    auc_pr = None
    max_pr_f1 = None
    try:
        import sklearn.metrics as skl_metrics
        if len(all_mask_labels) > 0:
            y_scores = np.concatenate(all_mask_scores)
            y_true = np.concatenate(all_mask_labels)
            # remove ignore labels if any (255)
            valid_mask = (y_true != 255)
            if valid_mask.sum() > 0 and y_true.sum() > 0 and (y_true.size - y_true.sum()) > 0:
                auc_roc = float(skl_metrics.roc_auc_score(y_true[valid_mask], y_scores[valid_mask]))
                auc_pr = float(skl_metrics.average_precision_score(y_true[valid_mask], y_scores[valid_mask]))
                precision_pr, recall_pr, _ = skl_metrics.precision_recall_curve(y_true[valid_mask], y_scores[valid_mask])
                f1s = (2 * precision_pr * recall_pr) / (precision_pr + recall_pr + 1e-10)
                max_pr_f1 = float(np.nanmax(f1s))
            else:
                auc_roc = None
                auc_pr = None
                max_pr_f1 = None
    except Exception:
        # fallback: coarse thresholding if sklearn not available
        if len(all_mask_labels) > 0:
            y_scores = np.concatenate(all_mask_scores)
            y_true = np.concatenate(all_mask_labels)
            valid_mask = (y_true != 255)
            if valid_mask.sum() > 0 and y_true.sum() > 0 and (y_true.size - y_true.sum()) > 0:
                thresholds = np.linspace(0.0, 1.0, 101)
                tprs = []
                fprs = []
                precs = []
                recs = []
                for t in thresholds:
                    preds = (y_scores >= t).astype(np.int32)
                    tp = np.logical_and(preds == 1, y_true == 1).sum()
                    fp = np.logical_and(preds == 1, y_true == 0).sum()
                    fn = np.logical_and(preds == 0, y_true == 1).sum()
                    tn = np.logical_and(preds == 0, y_true == 0).sum()
                    if (tp + fn) > 0:
                        tpr = tp / (tp + fn)
                    else:
                        tpr = 0.0
                    if (fp + tn) > 0:
                        fpr = fp / (fp + tn)
                    else:
                        fpr = 0.0
                    if (tp + fp) > 0:
                        prec = tp / (tp + fp)
                    else:
                        prec = 0.0
                    if (tp + fn) > 0:
                        rec = tp / (tp + fn)
                    else:
                        rec = 0.0
                    tprs.append(tpr)
                    fprs.append(fpr)
                    precs.append(prec)
                    recs.append(rec)
                auc_roc = float(np.trapz(tprs, fprs))
                auc_pr = float(np.trapz(precs[::-1], recs[::-1]))
                f1s = [(2 * p * r) / (p + r + 1e-10) for p, r in zip(precs, recs)]
                max_pr_f1 = float(np.max(f1s))

    if args.local_rank == 0:
        writer.add_scalar("test/accuracy", accuracy, epoch)
        writer.add_scalar("test/giou", giou, epoch)
        writer.add_scalar("test/ciou", ciou, epoch)
        writer.add_scalar("test/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("test/iou", iou, epoch)
        writer.add_scalar("test/seg_f1", seg_f1, epoch)
        if auc_roc is not None:
            writer.add_scalar("test/auc_roc", auc_roc, epoch)
        if auc_pr is not None:
            writer.add_scalar("test/auc_pr", auc_pr, epoch)
        if max_pr_f1 is not None:
            writer.add_scalar("test/max_pr_f1", max_pr_f1, epoch)
        for class_name, metrics in per_class_metrics.items():
            for metric_name, value in metrics.items():
                writer.add_scalar(f"test/{class_name.lower().replace('/', '_')}_{metric_name}", value, epoch)

        test_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{test_type} test Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU (Tampered): {iou:.4f}")
        print(f"Segmentation F1 (pixel-level): {seg_f1:.4f}")
        if auc_roc is not None:
            print(f"AUC (ROC) for localization: {auc_roc:.4f}")
        if auc_pr is not None:
            print(f"AUC (PR / AP) for localization: {auc_pr:.4f}")
        if max_pr_f1 is not None:
            print(f"Max F1 on PR curve: {max_pr_f1:.4f}")
        print(f"Total correct classifications: {correct}")
        print(f"Total classification samples: {total}")
        print("\nPer-Class Metrics:")
        for class_name, metrics in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")

        # report detection metric specifically for tampered class
        tampered_metrics = per_class_metrics.get('Tampered', None)
        if tampered_metrics is not None:
            print(f"\nDetection (Tampered) - Precision: {tampered_metrics['precision']:.4f}, F1: {tampered_metrics['f1']:.4f}")

        print("\nConfusion Matrix:")
        print("Predicted ")
        print("Actual ")
        print(f"{'':20}", end="")  
        for name in class_names:
            print(f"{name:>12}", end="") 
        print()  

        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="") 
            for j in range(num_classes):
                print(f"{confusion_matrix[i, j]:12.0f}", end="")
            print()  

    return accuracy, giou, ciou, per_class_metrics

if __name__ == "__main__":
    main(sys.argv[1:])

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from model.llava.model.token_pruning import prune_sequence_tensor
from utils.utils import IGNORE_INDEX


def prune_batch_inputs(
    model,
    input_dict: Dict[str, torch.Tensor],
    tokenizer,
    keep_ratio: float,
    observe_layer: int,
    special_token_ids: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Run observe-layer pruning on a batch of inputs.

    Mutates and returns ``input_dict`` with pruned tensors plus cached hidden states
    for downstream reuse. ``keep_ratio`` in (0, 1). When ``keep_ratio`` is outside
    that range or required tensors are missing, ``input_dict`` is returned unchanged.
    """

    if keep_ratio <= 0 or keep_ratio >= 1:
        return input_dict

    input_ids = input_dict.get("input_ids")
    attention_masks = input_dict.get("attention_masks")
    images_clip = input_dict.get("images_clip")

    if input_ids is None or attention_masks is None or images_clip is None:
        return input_dict

    if input_ids.dim() != 2 or input_ids.size(1) <= 2:
        return input_dict

    if special_token_ids is None:
        special_token_ids = [
            tok
            for tok in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ]
            if tok is not None
        ]

    sidamodel = model.module if hasattr(model, "module") else model

    pruning = sidamodel.observe_prune_prompt(
        input_ids=input_ids,
        attention_mask=attention_masks,
        images=images_clip,
        keep_ratio=keep_ratio,
        observe_layer=observe_layer,
        special_token_ids=special_token_ids,
    )

    input_dict["input_ids"] = pruning["input_ids"]
    input_dict["attention_masks"] = pruning["attention_mask"]

    labels = input_dict.get("labels")
    if labels is not None:
        pruned_labels, _ = prune_sequence_tensor(
            labels, pruning["keep_indices"], pad_value=IGNORE_INDEX
        )
        input_dict["labels"] = pruned_labels

    input_dict["keep_indices"] = pruning["keep_indices"]
    input_dict["sequence_lengths"] = pruning["lengths"]

    resume_outputs = pruning.get("resume_outputs")
    if resume_outputs is not None and hasattr(resume_outputs, "last_hidden_state"):
        cached_hidden_states = resume_outputs.last_hidden_state.detach()
        # Some transformers/LLaVA combinations return (seq, hidden) for last_hidden_state
        # when batch_size == 1. Downstream expects batch-first (B, seq, hidden).
        if cached_hidden_states.dim() == 2:
            cached_hidden_states = cached_hidden_states.unsqueeze(0)
        input_dict["cached_hidden_states"] = cached_hidden_states

    return input_dict

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from utils.utils import IMAGE_TOKEN_INDEX


logger = logging.getLogger(__name__)


@dataclass
class PrunedSequence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    keep_indices: List[torch.Tensor]
    lengths: torch.Tensor


def _ensure_list_of_indices(
    indices: Sequence[torch.Tensor], device: torch.device
) -> List[torch.Tensor]:
    normalized: List[torch.Tensor] = []
    for tensor in indices:
        if tensor.numel() == 0:
            normalized.append(torch.zeros(0, dtype=torch.long, device=device))
        else:
            normalized.append(tensor.to(device=device, dtype=torch.long))
    return normalized


def compute_image_token_keep_indices(
    attentions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    keep_ratio: float,
    image_token_id: int = IMAGE_TOKEN_INDEX,
    special_token_ids: Optional[Iterable[int]] = None,
) -> List[torch.Tensor]:
    """Select image token indices to keep, preserving all text tokens.

    Args:
        attentions: Tensor of shape (batch, num_heads, seq_len, seq_len)
            taken from observe_layers output. We'll average heads and slice the
            final query token for saliency estimates.
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) with 1 for valid tokens
        keep_ratio: Fraction of image tokens to retain (0-1).
        image_token_id: Token id that denotes image patch placeholders.
        special_token_ids: Optional iterable of token ids that should always be kept.

    Returns:
        List of 1-D tensors, each containing sorted indices to keep per sample.
    """

    if attentions.dim() != 4:
        raise ValueError(
            f"Expected attentions to be 4-D (batch, heads, seq, seq); got {attentions.shape}"
        )

    if not 0 <= keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be within [0, 1], got {keep_ratio}")

    batch_size, _, _, seq_len = attentions.shape
    device = input_ids.device
    special_ids_tensor: Optional[torch.Tensor] = None
    if special_token_ids:
        special_ids_tensor = torch.tensor(
            list(special_token_ids), device=device, dtype=input_ids.dtype
        )

    # Average heads and focus on the final query position ("CLS" equivalent).
    attn_mean = attentions.mean(dim=1)  # (batch, seq, seq)
    query_scores = attn_mean[:, -1, :]  # (batch, seq)

    keep_indices: List[torch.Tensor] = []
    for b in range(batch_size):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len <= 1:
            keep_indices.append(torch.arange(valid_len, device=device))
            continue

        sample_scores = query_scores[b, :valid_len].contiguous()
        prompt_tokens = input_ids[b, :valid_len]
        image_mask = prompt_tokens == image_token_id

        mandatory_mask = torch.ones(valid_len, dtype=torch.bool, device=device)
        mandatory_mask &= ~image_mask
        mandatory_mask[0] = True
        mandatory_mask[-1] = True

        if special_ids_tensor is not None and special_ids_tensor.numel() > 0:
            mandatory_mask |= torch.isin(prompt_tokens, special_ids_tensor)

        keep_idx = torch.nonzero(mandatory_mask, as_tuple=False).squeeze(1)

        image_positions = torch.nonzero(image_mask, as_tuple=False).squeeze(1)
        if image_positions.numel() == 0 or keep_ratio >= 1.0:
            keep_indices.append(torch.unique(keep_idx))
            continue

        image_scores = sample_scores[image_positions]
        image_scores = image_scores / (image_scores.sum() + 1e-6)
        keep_quota = max(int(math.ceil(image_positions.numel() * keep_ratio)), 0)
        keep_quota = min(keep_quota, image_positions.numel())

        if keep_quota == 0:
            selected = torch.empty(0, dtype=torch.long, device=device)
        elif keep_quota == image_positions.numel():
            selected = image_positions
        else:
            topk = torch.topk(image_scores, k=keep_quota, largest=True).indices
            selected = image_positions[topk]

        merged = torch.cat([keep_idx, selected]) if selected.numel() else keep_idx
        merged = torch.unique(merged)
        merged, _ = torch.sort(merged)
        keep_indices.append(merged)

    return keep_indices


def prune_sequence_tensor(
    tensor: torch.Tensor,
    keep_indices: Sequence[torch.Tensor],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice a batch-first sequence tensor along dim=1 and pad to equal length.

    Args:
        tensor: Shape (batch, seq_len, ...).
        keep_indices: List of 1-D tensors with indices for each sample.
        pad_value: Value used to pad shorter sequences to the max kept length.

    Returns:
        Tuple of (padded_tensor, lengths_tensor).
    """

    if tensor.dim() < 2:
        raise ValueError("Expected tensor with at least 2 dims (batch, seq_len, ...)")

    batch_size = tensor.size(0)
    device = tensor.device

    if len(keep_indices) != batch_size:
        logger.warning(
            "prune_sequence_tensor expected %d keep index lists but received %d; keeping full sequences.",
            batch_size,
            len(keep_indices),
        )
        keep_indices = [torch.arange(tensor.size(1), device=device) for _ in range(batch_size)]
    else:
        keep_indices = _ensure_list_of_indices(keep_indices, device)

    kept_tensors: List[torch.Tensor] = []
    lengths = []
    for b in range(batch_size):
        idx = keep_indices[b]
        if idx.numel() == 0:
            kept = tensor.new_zeros((0,) + tensor.shape[2:])
        else:
            kept = tensor[b].index_select(dim=0, index=idx)
        kept_tensors.append(kept)
        lengths.append(kept.size(0))

    max_len = max(lengths) if lengths else 0
    output_shape = (batch_size, max_len) + tensor.shape[2:]
    padded = tensor.new_full(output_shape, pad_value)

    for b, kept in enumerate(kept_tensors):
        seq_len = kept.size(0)
        if seq_len == 0:
            continue
        padded[b, :seq_len] = kept

    length_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
    return padded, length_tensor


def prune_attention_mask(
    attention_mask: torch.Tensor, keep_indices: Sequence[torch.Tensor]
) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be rank-2")

    device = attention_mask.device
    if len(keep_indices) != attention_mask.size(0):
        logger.warning(
            "prune_attention_mask expected %d keep index lists but received %d; keeping full sequences.",
            attention_mask.size(0),
            len(keep_indices),
        )
        keep_indices = [
            torch.arange(attention_mask.size(1), device=device) for _ in range(attention_mask.size(0))
        ]
    else:
        keep_indices = _ensure_list_of_indices(keep_indices, device)
    batch_size = attention_mask.size(0)
    lengths = [idx.numel() for idx in keep_indices]
    max_len = max(lengths) if lengths else 0

    pruned = attention_mask.new_zeros((batch_size, max_len))
    for b, idx in enumerate(keep_indices):
        seq_len = idx.numel()
        if seq_len == 0:
            continue
        pruned[b, :seq_len] = 1
    return pruned


def prune_past_key_values(
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
    keep_indices: Sequence[torch.Tensor],
    pad_value: float = 0.0,
) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
    if past_key_values is None:
        return None

    device = past_key_values[0][0].device
    keep_indices = _ensure_list_of_indices(keep_indices, device)
    max_len = max(idx.numel() for idx in keep_indices) if keep_indices else 0

    pruned_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for key_states, value_states in past_key_values:
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise ValueError("Expected key/value states with shape (batch, heads, seq, dim)")

        batch_size, num_heads, _, head_dim = key_states.shape
        pruned_keys = key_states.new_full(
            (batch_size, num_heads, max_len, head_dim), pad_value
        )
        pruned_values = value_states.new_full(
            (batch_size, num_heads, max_len, head_dim), pad_value
        )

        for b, idx in enumerate(keep_indices):
            seq_len = idx.numel()
            if seq_len == 0:
                continue
            pruned_keys[b, :, :seq_len] = key_states[b, :, idx, :]
            pruned_values[b, :, :seq_len] = value_states[b, :, idx, :]

        pruned_layers.append((pruned_keys, pruned_values))

    return tuple(pruned_layers)

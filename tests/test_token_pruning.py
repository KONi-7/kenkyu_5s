import torch

from model.llava.model.token_pruning import (
    compute_image_token_keep_indices,
    prune_attention_mask,
    prune_past_key_values,
    prune_sequence_tensor,
)


def _dummy_attn(batch: int, seq: int, heads: int = 2):
    attn = torch.rand(batch, heads, seq, seq)
    attn = attn / attn.sum(dim=-1, keepdim=True)
    return attn


def test_compute_image_token_keep_indices_preserves_text_tokens():
    batch, seq = 1, 6
    input_ids = torch.tensor([[101, -200, -200, 42, -200, 102]])
    attn = _dummy_attn(batch, seq)
    mask = torch.ones(batch, seq)
    keep = compute_image_token_keep_indices(attn, input_ids, mask, keep_ratio=0.5)
    keep_idx = keep[0].tolist()
    assert 0 in keep_idx and seq - 1 in keep_idx
    # Ensure text token id 42 remains
    assert 3 in keep_idx
    # At least one image token should remain because ratio=0.5 of 3 -> 2
    assert any(idx in keep_idx for idx in [1, 2, 4])


def test_prune_sequence_tensor_shapes():
    hidden = torch.randn(2, 5, 8)
    keep = [torch.tensor([0, 2, 4]), torch.tensor([1, 3])]
    pruned, lengths = prune_sequence_tensor(hidden, keep)
    assert pruned.shape == (2, 3, 8)
    assert lengths.tolist() == [3, 2]
    # Second sample padded at the end should be zeros
    assert torch.allclose(pruned[1, 2], torch.zeros(8))


def test_prune_attention_mask_matches_lengths():
    mask = torch.ones(2, 5)
    keep = [torch.tensor([0, 1, 2]), torch.tensor([0])]
    pruned = prune_attention_mask(mask, keep)
    assert torch.equal(pruned[0], torch.tensor([1, 1, 1]))
    assert torch.equal(pruned[1], torch.tensor([1, 0, 0]))


def test_prune_past_key_values_applies_indices():
    batch, heads, seq, dim = 2, 2, 5, 4
    keys = torch.randn(batch, heads, seq, dim)
    values = torch.randn(batch, heads, seq, dim)
    keep = [torch.tensor([0, 2, 4]), torch.tensor([1, 3])]
    pruned_cache = prune_past_key_values(((keys, values),), keep)
    assert pruned_cache is not None
    pruned_keys, pruned_values = pruned_cache[0]
    assert pruned_keys.shape == (batch, heads, 3, dim)
    assert pruned_values.shape == (batch, heads, 3, dim)
    assert torch.allclose(pruned_keys[0, :, 1], keys[0, :, 2])
    assert torch.allclose(pruned_values[1, :, 0], values[1, :, 1])

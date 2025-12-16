import torch


def _build_masks(input_ids: torch.Tensor, attention_masks: torch.Tensor, cls_id: int, seg_id: int):
    cls_mask = (input_ids == cls_id)
    seg_mask = (input_ids == seg_id)
    attn_bool = attention_masks.to(torch.bool)
    if attn_bool.shape == cls_mask.shape:
        cls_mask = cls_mask & attn_bool
        seg_mask = seg_mask & attn_bool
    return cls_mask, seg_mask


def test_cls_seg_masks_are_length_safe_and_pad_aware():
    # Sequence length intentionally not 255+something.
    batch = 2
    seq_len = 37
    cls_id = 100
    seg_id = 101

    input_ids = torch.zeros((batch, seq_len), dtype=torch.long)
    attention_masks = torch.ones((batch, seq_len), dtype=torch.long)

    # Put CLS/SEG at a few positions.
    input_ids[0, 5] = cls_id
    input_ids[0, 10] = seg_id
    input_ids[1, 7] = cls_id

    # Add padding at end of sample 1 and ensure tokens there won't be selected.
    attention_masks[1, 30:] = 0
    input_ids[1, 31] = cls_id

    cls_mask, seg_mask = _build_masks(input_ids, attention_masks, cls_id, seg_id)

    assert cls_mask.shape == (batch, seq_len)
    assert seg_mask.shape == (batch, seq_len)

    # True where we placed them (and not padded)
    assert cls_mask[0, 5].item() is True
    assert seg_mask[0, 10].item() is True
    assert cls_mask[1, 7].item() is True

    # Padded region must be False even if token id matches
    assert cls_mask[1, 31].item() is False

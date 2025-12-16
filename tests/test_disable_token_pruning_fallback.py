import torch


def _apply_no_pruning_fallback(input_dict: dict) -> dict:
    """Minimal reproduction of `test.py` no-pruning fallback contract.

    Contract we rely on in `model_forward()`:
    - keep_indices: list[Tensor] of shape [seq_len]
    - sequence_lengths: list[int] length = batch
    - cached_hidden_states: Tensor [batch, seq_len, hidden]

    This is a unit-level check that doesn't load the full model.
    """
    seq_len = input_dict["input_ids"].shape[1]
    device = input_dict["input_ids"].device

    if "keep_indices" not in input_dict:
        input_dict["keep_indices"] = [torch.arange(seq_len, device=device) for _ in range(input_dict["input_ids"].shape[0])]

    if "sequence_lengths" not in input_dict:
        input_dict["sequence_lengths"] = input_dict["attention_masks"].sum(dim=1).detach().cpu().tolist()

    if "cached_hidden_states" not in input_dict:
        # Dummy hidden states (we only assert shape contract here)
        hidden = 8
        input_dict["cached_hidden_states"] = torch.zeros(
            (input_dict["input_ids"].shape[0], seq_len, hidden), device=device
        )

    return input_dict


def test_no_pruning_fallback_contract_shapes():
    batch = 2
    seq_len = 16
    device = torch.device("cpu")

    input_dict = {
        "input_ids": torch.ones((batch, seq_len), dtype=torch.long, device=device),
        "attention_masks": torch.ones((batch, seq_len), dtype=torch.long, device=device),
    }

    out = _apply_no_pruning_fallback(input_dict)

    assert "keep_indices" in out
    assert isinstance(out["keep_indices"], list)
    assert len(out["keep_indices"]) == batch
    assert all(x.shape == (seq_len,) for x in out["keep_indices"])

    assert "sequence_lengths" in out
    assert isinstance(out["sequence_lengths"], list)
    assert out["sequence_lengths"] == [seq_len, seq_len]

    assert "cached_hidden_states" in out
    assert out["cached_hidden_states"].shape[0] == batch
    assert out["cached_hidden_states"].shape[1] == seq_len

import torch

from model.llava.model.language_model.llava_llama import (
    LlavaCausalLMOutputWithPast,
    LlavaConfig,
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
    LlavaModelOutputWithPastAndObservations,
)


def _tiny_config():
    return LlavaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_llava_model_collects_requested_layers():
    config = _tiny_config()
    model = LlavaLlamaModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    outputs = model(
        input_ids=input_ids,
        observe_layers=[0, 1],
        collect_attentions=True,
        stop_at_last_observe_layer=True,
        use_cache=True,
        return_dict=True,
    )

    assert isinstance(outputs, LlavaModelOutputWithPastAndObservations)
    assert outputs.observed_layer_indices == [0, 1]
    assert len(outputs.observed_hidden_states) == 2
    assert outputs.observed_hidden_states[0].shape == (2, 5, config.hidden_size)
    assert outputs.observed_attentions is not None
    assert outputs.observed_attentions[0] is not None
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == 2


def test_causal_lm_forward_includes_observations_without_early_stop():
    config = _tiny_config()
    model = LlavaLlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))

    outputs = model(
        input_ids=input_ids,
        observe_layers=[1],
        return_dict=True,
    )

    assert isinstance(outputs, LlavaCausalLMOutputWithPast)
    assert outputs.logits.shape == (1, 4, config.vocab_size)
    assert outputs.observed_layer_indices == [1]
    assert outputs.observed_hidden_states is not None
    assert len(outputs.observed_hidden_states) == 1
    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == config.num_hidden_layers


def test_forward_from_layer_matches_full_forward():
    config = _tiny_config()
    model = LlavaLlamaModel(config)
    seq_len = 6
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    attention_mask = torch.ones_like(input_ids)

    full_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    observe_layer = 1
    partial_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        observe_layers=[observe_layer],
        collect_attentions=True,
        stop_at_last_observe_layer=True,
    )

    boundary_hidden = partial_outputs.observed_hidden_states[-1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    resumed_outputs = model.forward_from_layer(
        hidden_states=boundary_hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        layer_start=observe_layer + 1,
    )

    assert torch.allclose(
        full_outputs.last_hidden_state, resumed_outputs.last_hidden_state, atol=1e-5
    )

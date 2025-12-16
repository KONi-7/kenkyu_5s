#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import (AutoConfig, AutoModelForCausalLM, LlamaConfig,
                          LlamaForCausalLM, LlamaModel)
from transformers.models.llama import modeling_llama as hf_llama
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.utils import logging

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


os.environ.setdefault("LLAMA_ATTENTION_IMPL", "eager")


logger = logging.get_logger(__name__)

if not hasattr(hf_llama, "_llava_rope_patched"):
    _original_apply_rotary = hf_llama.apply_rotary_pos_emb

    def _reshape_rotary_input(tensor: torch.Tensor, cos: torch.Tensor):
        if tensor.dim() == 4:
            return tensor, False

        if tensor.dim() != 3 or cos.dim() < 2:
            return tensor, False

        seq_len = cos.shape[-2]
        head_dim = cos.shape[-1]
        dims = list(tensor.shape)

        if seq_len not in dims or head_dim not in dims:
            return tensor, False

        seq_idx = next((i for i, size in enumerate(dims) if size == seq_len), None)
        head_idx = next(
            (i for i, size in enumerate(dims) if size == head_dim and i != seq_idx),
            None,
        )

        if seq_idx is None or head_idx is None:
            return tensor, False

        remaining_idx = [i for i in range(len(dims)) if i not in (seq_idx, head_idx)]
        if not remaining_idx:
            return tensor, False

        permute_order = remaining_idx + [seq_idx, head_idx]
        permuted = tensor.permute(permute_order).contiguous()

        batch = cos.shape[0] if cos.shape[0] > 0 else 1
        lead_dim = permuted.shape[0]
        if lead_dim % batch != 0:
            batch = 1
        num_heads = lead_dim // batch if lead_dim >= batch else 1

        reshaped = permuted.reshape(batch, num_heads, seq_len, head_dim)
        return reshaped, True

    def _apply_rotary_with_logging(q, k, cos, sin, *args, **kwargs):
        q_fixed, q_changed = _reshape_rotary_input(q, cos)
        k_fixed, k_changed = _reshape_rotary_input(k, cos)
        if (q_changed or k_changed) and not hasattr(hf_llama, "_warned_rotary_input_reshape"):
            logger.warning(
                "Detected rotary inputs lacking batch/head dims. Automatically reshaping to q=%s k=%s.",
                tuple(q_fixed.shape),
                tuple(k_fixed.shape),
            )
            hf_llama._warned_rotary_input_reshape = True

        try:
            return _original_apply_rotary(q_fixed, k_fixed, cos, sin, *args, **kwargs)
        except RuntimeError:
            logger.error(
                "Rotary application failed: q=%s k=%s cos=%s sin=%s",
                tuple(q_fixed.shape),
                tuple(k_fixed.shape),
                tuple(cos.shape),
                tuple(sin.shape),
            )
            raise

    hf_llama.apply_rotary_pos_emb = _apply_rotary_with_logging
    hf_llama._llava_rope_patched = True

if not hasattr(hf_llama, "_llava_attention_forward_patched"):

    # Newer `transformers` versions provide `eager_attention_forward` and an
    # attention-implementation registry. If they're missing, don't patch
    # attention at import-time; keep the stock implementation.
    if not hasattr(hf_llama, "eager_attention_forward") or not hasattr(hf_llama, "ALL_ATTENTION_FUNCTIONS"):
        logger.warning(
            "Transformers version lacks `eager_attention_forward`/`ALL_ATTENTION_FUNCTIONS`; skipping LLaVA attention patch."
        )
        hf_llama._llava_attention_forward_patched = True
    else:

        def _llava_attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            cos, sin = position_embeddings
            expected_seq_len = cos.shape[-2] if cos is not None else None

            if hidden_states.dim() == 1:
                hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)

            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
                if attention_mask is not None and attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(0)

            if (
                hidden_states.dim() == 3
                and expected_seq_len is not None
                and hidden_states.shape[1] != expected_seq_len
                and hidden_states.shape[0] == expected_seq_len
            ):
                hidden_states = hidden_states.transpose(0, 1).contiguous()

            if hidden_states.dim() != 3:
                raise ValueError(
                    f"Expected hidden_states with 3 dims (batch, seq, hidden), got shape={tuple(hidden_states.shape)}"
                )

            bsz, q_len, _ = hidden_states.size()
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads
            head_dim = self.head_dim

            def _shape(states: torch.Tensor, num_heads_local: int):
                return (
                    states.view(bsz, q_len, num_heads_local, head_dim)
                    .transpose(1, 2)
                    .contiguous()
                )

            query_states = _shape(self.q_proj(hidden_states), num_heads)
            key_states = _shape(self.k_proj(hidden_states), num_kv_heads)
            value_states = _shape(self.v_proj(hidden_states), num_kv_heads)

            position_ids = kwargs.pop("position_ids", None)

            if key_states.shape[2] != value_states.shape[2] and not hasattr(self, "_warned_value_seq_len"):
                logger.warning(
                    "Layer %s detected mismatch between key/value seq dim: key=%s value=%s hidden=%s",
                    getattr(self, "layer_idx", -1),
                    tuple(key_states.shape),
                    tuple(value_states.shape),
                    tuple(hidden_states.shape),
                )
                self._warned_value_seq_len = True

            query_states, key_states = hf_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids=position_ids
            )

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            output_attentions = kwargs.pop("output_attentions", False)
            kwargs.pop("use_cache", None)

            attention_interface: Callable = hf_llama.eager_attention_forward
            attn_impl = getattr(self.config, "_attn_implementation", "eager")
            if attn_impl != "eager":
                attention_interface = hf_llama.ALL_ATTENTION_FUNCTIONS[attn_impl]

            attention_kwargs = {
                "dropout": 0.0 if not self.training else self.attention_dropout,
                "scaling": self.scaling,
                "output_attentions": output_attentions,
            }
            attention_kwargs.update(kwargs)

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                **attention_kwargs,
            )

            if output_attentions and attn_weights is None and attn_impl != "eager":
                logger.warning(
                    "Attention implementation '%s' returned no attention weights; rerunning layer %s with eager attention for pruning support.",
                    attn_impl,
                    getattr(self, "layer_idx", -1),
                )
                attention_interface = hf_llama.eager_attention_forward
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    **attention_kwargs,
                )
                if getattr(self.config, "_attn_implementation", None) != "eager":
                    setattr(self.config, "_attn_implementation", "eager")
                if getattr(self.config, "attn_implementation", None) != "eager":
                    setattr(self.config, "attn_implementation", "eager")

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, num_heads * head_dim)
            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        hf_llama.LlamaAttention.forward = _llava_attention_forward
        hf_llama._llava_attention_forward_patched = True


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """Create the causal mask used for decoder self-attention."""

    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len,
                    past_key_values_length,
                    dtype=dtype,
                    device=device,
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
):
    """Expand a 2D attention mask to a 4D causal mask."""

    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LlavaConfig(LlamaConfig):
    model_type = "llava"


@dataclass
class LlavaModelOutputWithPastAndObservations(BaseModelOutputWithPast):
    observed_hidden_states: Optional[List[torch.Tensor]] = None
    observed_attentions: Optional[List[torch.Tensor]] = None
    observed_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    observed_layer_indices: Optional[List[int]] = None


@dataclass
class LlavaCausalLMOutputWithPast(CausalLMOutputWithPast):
    observed_hidden_states: Optional[List[torch.Tensor]] = None
    observed_attentions: Optional[List[torch.Tensor]] = None
    observed_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    observed_layer_indices: Optional[List[int]] = None


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        desired_attn_impl = "eager"
        current_attn_impl = getattr(config, "attn_implementation", None)
        if current_attn_impl != desired_attn_impl:
            logger.info(
                "Switching attention implementation from %s to '%s' to enable attention map outputs.",
                current_attn_impl,
                desired_attn_impl,
            )
            setattr(config, "attn_implementation", desired_attn_impl)

        if getattr(config, "_attn_implementation", None) != desired_attn_impl:
            setattr(config, "_attn_implementation", desired_attn_impl)

        attn_config = getattr(config, "attn_config", None)
        if isinstance(attn_config, dict):
            if attn_config.get("name") != desired_attn_impl:
                attn_config["name"] = desired_attn_impl
            if attn_config.get("attn_impl") != desired_attn_impl:
                attn_config["attn_impl"] = desired_attn_impl

        super(LlavaLlamaModel, self).__init__(config)

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        """Aligns with Hugging Face's decoder attention mask creation for causal decoders."""

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _decoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        observe_layers: Optional[List[int]],
        collect_attentions: bool,
        stop_at_last_observe_layer: bool,
        layer_start: int = 0,
    ) -> Union[Tuple, LlavaModelOutputWithPastAndObservations]:
        num_layers = len(self.layers)
        if layer_start < 0 or layer_start >= num_layers:
            raise ValueError(
                f"layer_start must be within [0, {num_layers - 1}], got {layer_start}"
            )

        normalized_observe_layers: List[int] = []
        if observe_layers:
            seen = set()
            for layer_idx in observe_layers:
                if layer_idx < 0:
                    layer_idx = num_layers + layer_idx
                if layer_idx < layer_start:
                    continue
                if layer_idx < 0 or layer_idx >= num_layers:
                    raise ValueError(
                        f"observe_layers contains invalid index {layer_idx} for model with {num_layers} layers."
                    )
                if layer_idx not in seen:
                    seen.add(layer_idx)
                    normalized_observe_layers.append(layer_idx)
            normalized_observe_layers.sort()

        observe_layer_set = set(normalized_observe_layers)
        observed_hidden_states: Dict[int, torch.Tensor] = {}
        observed_attentions: Dict[int, torch.Tensor] = {}
        observed_past_key_values: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        last_observe_layer = (
            normalized_observe_layers[-1] if normalized_observe_layers else None
        )
        should_stop_early = (
            stop_at_last_observe_layer and last_observe_layer is not None
        )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        last_executed_layer = layer_start - 1
        early_stop_triggered = False

        position_embeddings = None
        if position_ids is not None:
            rotary_source = getattr(self, "rotary_emb", None)
            if rotary_source is None and len(self.layers) > 0:
                rotary_source = getattr(self.layers[0].self_attn, "rotary_emb", None)

            if rotary_source is not None:
                seq_len = int(position_ids.detach().max().item()) + 1
                # transformers versions differ in LlamaRotaryEmbedding.forward signature:
                # - forward(x, seq_len=...)
                # - forward(x, position_ids)
                try:
                    param_names = set(inspect.signature(rotary_source.forward).parameters.keys())
                except (TypeError, ValueError):
                    param_names = set()

                if "seq_len" in param_names:
                    position_embeddings = rotary_source(hidden_states, seq_len=seq_len)
                elif "position_ids" in param_names:
                    position_embeddings = rotary_source(hidden_states, position_ids)
                else:
                    # Fallback for unknown signatures.
                    try:
                        position_embeddings = rotary_source(hidden_states, seq_len=seq_len)
                    except TypeError:
                        try:
                            position_embeddings = rotary_source(hidden_states, seq_len)
                        except TypeError:
                            position_embeddings = rotary_source(hidden_states, position_ids)

        for idx in range(layer_start, num_layers):
            decoder_layer = self.layers[idx]
            last_executed_layer = idx

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is not None:
                if len(past_key_values) == num_layers:
                    past_key_value = past_key_values[idx]
                elif len(past_key_values) == num_layers - layer_start:
                    past_key_value = past_key_values[idx - layer_start]
                else:
                    raise ValueError(
                        "past_key_values length must be either `num_layers` or `num_layers - layer_start`."
                    )
            else:
                past_key_value = None

            layer_requires_attn = output_attentions or collect_attentions

            layer_position_embeddings = position_embeddings
            if layer_position_embeddings is not None:
                cos, sin = layer_position_embeddings
                target_dim = decoder_layer.self_attn.head_dim
                if not hasattr(self, "_logged_rotary_shape"):
                    logger.warning(
                        "Rotary embedding base shape=%s, target head_dim=%s",
                        tuple(cos.shape),
                        target_dim,
                    )
                    logger.warning(
                        "LlamaAttention head_dim property=%s", decoder_layer.self_attn.head_dim
                    )
                    self._logged_rotary_shape = True
                if cos.shape[-1] != target_dim:
                    if cos.shape[-1] < target_dim:
                        raise ValueError(
                            "Rotary position embedding dimension is smaller than the attention head dimension."
                        )
                    if not hasattr(self, "_warned_rotary_dim_mismatch"):
                        logger.warning(
                            "Adjusting rotary embeddings from dim=%s to match layer head_dim=%s",
                            cos.shape[-1],
                            target_dim,
                        )
                        self._warned_rotary_dim_mismatch = True
                    cos = cos[..., :target_dim]
                    sin = sin[..., :target_dim]
                    layer_position_embeddings = (cos, sin)

            layer_input_states = hidden_states

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs,
                            output_attentions=layer_requires_attn,
                            use_cache=False,
                            position_embeddings=layer_position_embeddings,
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=layer_requires_attn,
                    use_cache=use_cache,
                    position_embeddings=layer_position_embeddings,
                )

            hidden_states = layer_outputs[0]

            attn_weights = None
            if layer_requires_attn:
                if len(layer_outputs) > 1:
                    attn_weights = layer_outputs[1]
                else:
                    attn_weights = None

                if attn_weights is None:
                    logger.warning(
                        "Decoder layer %s did not return attention weights; computing them manually via self-attention.",
                        getattr(decoder_layer.self_attn, "layer_idx", -1),
                    )
                    self_attn_outputs = decoder_layer.self_attn(
                        layer_input_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=True,
                        use_cache=use_cache,
                        cache_position=None,
                        position_embeddings=layer_position_embeddings,
                    )
                    if isinstance(self_attn_outputs, tuple) and len(self_attn_outputs) >= 2:
                        attn_weights = self_attn_outputs[1]
                    else:
                        raise RuntimeError(
                            "Attention weights were requested but could not be retrieved even after manual self-attention pass."
                        )

            if use_cache:
                cache_index = 2 if layer_requires_attn else 1
                if len(layer_outputs) <= cache_index:
                    raise RuntimeError(
                        "Decoder layer did not return past key values although use_cache=True."
                    )
                present_key_value = layer_outputs[cache_index]
                next_decoder_cache += (present_key_value,)
            else:
                present_key_value = None

            if output_attentions and attn_weights is not None:
                all_self_attns += (attn_weights,)

            if idx in observe_layer_set:
                observed_hidden_states[idx] = hidden_states
                if collect_attentions and attn_weights is not None:
                    observed_attentions[idx] = attn_weights
                if use_cache and present_key_value is not None:
                    observed_past_key_values[idx] = present_key_value

            if should_stop_early and idx == last_observe_layer:
                early_stop_triggered = True
                break

        completed_all_layers = last_executed_layer == num_layers - 1 and not early_stop_triggered
        if completed_all_layers:
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            outputs = [hidden_states, next_cache, all_hidden_states, all_self_attns]
            return tuple(v for v in outputs if v is not None)

        observed_layer_indices = (
            normalized_observe_layers if normalized_observe_layers else None
        )
        observed_hidden_list = (
            [observed_hidden_states[i] for i in normalized_observe_layers]
            if normalized_observe_layers
            else None
        )
        observed_attn_list = (
            [observed_attentions.get(i) for i in normalized_observe_layers]
            if normalized_observe_layers and collect_attentions
            else None
        )
        observed_cache_list = (
            [observed_past_key_values.get(i) for i in normalized_observe_layers]
            if normalized_observe_layers and use_cache
            else None
        )

        return LlavaModelOutputWithPastAndObservations(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            observed_hidden_states=observed_hidden_list,
            observed_attentions=observed_attn_list,
            observed_past_key_values=observed_cache_list,
            observed_layer_indices=observed_layer_indices,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        observe_layers: Optional[List[int]] = None,
        collect_attentions: bool = False,
        stop_at_last_observe_layer: bool = False,
    ) -> Union[Tuple, LlavaModelOutputWithPastAndObservations]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if observe_layers is not None and not return_dict:
            raise ValueError(
                "observe_layers is only supported when return_dict=True."
            )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds
        if not hasattr(self, "_logged_decoder_input_shape"):
            logger.warning(
                "Decoder input hidden_states shape=%s",
                tuple(hidden_states.shape),
            )
            self._logged_decoder_input_shape = True

        return self._decoder_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observe_layers=observe_layers,
            collect_attentions=collect_attentions,
            stop_at_last_observe_layer=stop_at_last_observe_layer,
            layer_start=0,
        )

    def forward_from_layer(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        observe_layers: Optional[List[int]] = None,
        collect_attentions: bool = False,
        stop_at_last_observe_layer: bool = False,
        layer_start: int = 0,
    ) -> Union[Tuple, LlavaModelOutputWithPastAndObservations]:
        if hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must be 3-D (batch, seq_len, hidden_size) when resuming from layer"
            )

        batch_size, seq_length, _ = hidden_states.shape

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )

        if position_ids is None:
            position_ids = torch.arange(seq_length, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(batch_size, seq_length)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length=0,
        )

        return self._decoder_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observe_layers=observe_layers,
            collect_attentions=collect_attentions,
            stop_at_last_observe_layer=stop_at_last_observe_layer,
            layer_start=layer_start,
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        observe_layers: Optional[List[int]] = None,
        collect_attentions: Optional[bool] = None,
        stop_at_last_observe_layer: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if observe_layers is not None and not return_dict:
            raise ValueError(
                "observe_layers is only supported when return_dict=True."
            )

        collect_attentions = bool(collect_attentions)
        stop_at_last_observe_layer = bool(stop_at_last_observe_layer)

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            observe_layers=observe_layers,
            collect_attentions=collect_attentions,
            stop_at_last_observe_layer=stop_at_last_observe_layer,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,  # outputs.hidden_states,
            attentions=outputs.attentions,
            observed_hidden_states=getattr(outputs, "observed_hidden_states", None),
            observed_attentions=getattr(outputs, "observed_attentions", None),
            observed_past_key_values=getattr(
                outputs, "observed_past_key_values", None
            ),
            observed_layer_indices=getattr(outputs, "observed_layer_indices", None),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs


try:
    AutoConfig.register("llava", LlavaConfig)
except ValueError:
    pass
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

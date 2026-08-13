# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer dense decoder layer."""

import ttnn

from models.demos.muse_glimmer.tt.attention import MuseGlimmerAttention, MuseGlimmerAttentionConfig
from models.demos.muse_glimmer.tt.rms_norm import RMSNorm
from models.demos.muse_glimmer.tt.shared_mlp import SharedMLP
from models.demos.muse_glimmer.utils.substate import substate


class MuseGlimmerDecoderLayer:
    def __init__(
        self,
        device,
        hf_config,
        state_dict,
        layer_idx,
        attention_dtype,
        mlp_dtype,
        tensor_cache_path,
        mlp_cache_path,
        max_seq_len,
        create_kv_cache=False,
        kv_cache_dtype=ttnn.bfloat16,
    ):
        layer_state = substate(state_dict, f"model.language_model.layers.{layer_idx}")

        def norm(name, eps):
            cache = f"{tensor_cache_path}/layer_{layer_idx}/{name}" if tensor_cache_path else None
            return RMSNorm(
                device,
                hf_config.hidden_size,
                substate(layer_state, name),
                eps,
                cache,
                centered=True,
            )

        self.input_layernorm = norm("input_layernorm", hf_config.rms_norm_eps)
        self.post_attention_layernorm = norm("post_attention_layernorm", hf_config.post_norm_eps)
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm", hf_config.rms_norm_eps)
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm", hf_config.post_norm_eps)

        attention_cache = f"{tensor_cache_path}/layer_{layer_idx}/self_attn" if tensor_cache_path else None
        self.self_attn = MuseGlimmerAttention(
            device,
            MuseGlimmerAttentionConfig(hf_config, layer_idx),
            substate(layer_state, "self_attn"),
            attention_cache,
            max_seq_len,
            create_kv_cache,
            kv_cache_dtype,
            attention_dtype,
        )
        mlp_cache = f"{mlp_cache_path}/layer_{layer_idx}/mlp" if mlp_cache_path else None
        self.mlp = SharedMLP(device, substate(layer_state, "mlp"), mlp_dtype, mlp_cache)

    def __call__(self, hidden_states, rope_mats, *, kv_cache=None, is_decode=False, packed=None):
        residual = hidden_states
        normalized = self.input_layernorm.forward(hidden_states)
        attention = self.self_attn(
            normalized,
            rope_mats,
            is_decode=is_decode,
            kv_cache=kv_cache,
            packed=packed,
        )
        normalized.deallocate(True)
        attention = self.post_attention_layernorm.forward(attention)
        hidden_states = ttnn.add(residual, attention)
        residual.deallocate(True)
        attention.deallocate(True)

        residual = hidden_states
        normalized = self.pre_feedforward_layernorm.forward(hidden_states)
        feed_forward = self.mlp(normalized)
        normalized.deallocate(True)
        feed_forward = self.post_feedforward_layernorm.forward(feed_forward)
        hidden_states = ttnn.add(residual, feed_forward)
        residual.deallocate(True)
        feed_forward.deallocate(True)
        return hidden_states

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core single-device Muse Glimmer attention operations."""

import ttnn


def apply_qkv_projection(hidden_states, weights, memory_config=None):
    return ttnn.linear(hidden_states, weights.wqkv, memory_config=memory_config)


def split_qkv_heads_decode(xqkv, config):
    if xqkv.memory_config().buffer_type == ttnn.BufferType.DRAM:
        xqkv = ttnn.to_memory_config(xqkv, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
    )


def split_qkv_heads_prefill(xqkv, config, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        transpose_k_heads=False,
        memory_config=memory_config,
    )


def apply_per_head_norm(tensor, weight, eps, with_scale=True, memory_config=None):
    shape = tensor.shape
    flat = ttnn.reshape(tensor, (1, 1, shape[1] * shape[2], shape[3]))
    normed = ttnn.rms_norm(
        flat,
        weight=weight if with_scale else None,
        epsilon=eps,
        memory_config=memory_config,
    )
    return ttnn.reshape(normed, shape)


def apply_rope(tensor, cos, sin, memory_config=None):
    return ttnn.experimental.rotary_embedding(tensor, cos, sin, None, memory_config=memory_config)


def concat_heads(tensor, is_decode_mode, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=memory_config)


def apply_output_projection(tensor, weights, normalized_hidden_states):
    gate = ttnn.linear(normalized_hidden_states, weights.gate_proj)
    gated = ttnn.multiply(tensor, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    tensor.deallocate(True)
    gate.deallocate(True)
    output = ttnn.linear(gated, weights.o_proj)
    gated.deallocate(True)
    return output

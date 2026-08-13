# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer attention weight loading."""

from dataclasses import dataclass

import torch
import ttnn

from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name


@dataclass(frozen=True)
class AttentionWeights:
    wqkv: ttnn.Tensor
    gate_proj: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(device, state_dict, weight_dtype=ttnn.bfloat8_b, tensor_cache_path=None):
    qkv_cache = get_cache_file_name(tensor_cache_path, "wqkv")
    gate_cache = get_cache_file_name(tensor_cache_path, "gate_proj")
    output_cache = get_cache_file_name(tensor_cache_path, "o_proj")
    qkv = cached_tensor_placeholder(qkv_cache, weight_dtype, ttnn.TILE_LAYOUT)
    gate = cached_tensor_placeholder(gate_cache, weight_dtype, ttnn.TILE_LAYOUT)
    output = cached_tensor_placeholder(output_cache, weight_dtype, ttnn.TILE_LAYOUT)
    if qkv is None:
        qkv = (
            torch.cat(
                [
                    state_dict["q_proj.weight"].transpose(-2, -1),
                    state_dict["k_proj.weight"].transpose(-2, -1),
                    state_dict["v_proj.weight"].transpose(-2, -1),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
    if gate is None:
        gate = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
    if output is None:
        output = state_dict["o_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)

    def as_weight(tensor, cache_name):
        return ttnn.as_tensor(
            tensor,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return AttentionWeights(
        wqkv=as_weight(qkv, qkv_cache),
        gate_proj=as_weight(gate, gate_cache),
        o_proj=as_weight(output, output_cache),
    )

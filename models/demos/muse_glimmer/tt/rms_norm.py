# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer RMSNorm weight loading and execution."""

import ttnn

from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder


class RMSNorm:
    def __init__(self, device, hidden_size, state_dict, eps, tensor_cache_path=None, centered=False):
        self.eps = eps
        cache_name = f"{tensor_cache_path}.weight" if tensor_cache_path else None
        weight = cached_tensor_placeholder(cache_name, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if weight is None:
            weight = state_dict["weight"]
            if centered:
                weight = weight + 1.0
            weight = weight.reshape(1, 1, -1, ttnn.TILE_SIZE).contiguous()
        self.weight = ttnn.as_tensor(
            weight,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=cache_name,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, hidden_states):
        return ttnn.rms_norm(hidden_states, weight=self.weight, epsilon=self.eps)

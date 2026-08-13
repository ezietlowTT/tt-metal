# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Muse Glimmer's dense SwiGLU feed-forward block."""

import ttnn

from models.demos.muse_glimmer.utils.general_utils import cached_tensor_placeholder, get_cache_file_name


class SharedMLP:
    def __init__(self, device, state_dict, dtype=ttnn.bfloat4_b, tensor_cache_path=None):
        def load(name):
            cache_name = get_cache_file_name(tensor_cache_path, f"{name}.weight")
            tensor = cached_tensor_placeholder(cache_name, dtype, ttnn.TILE_LAYOUT)
            if tensor is None:
                tensor = state_dict[f"{name}.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                tensor,
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        self.gate_proj = load("gate_proj")
        self.up_proj = load("up_proj")
        self.down_proj = load("down_proj")
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Packed verification always occupies one 32-token tile.  The
        # automatic 1-D matmul config uses a two-tile K block for the down
        # projection, which leaves the Blackhole DRAM readers under-filled.
        # Four K tiles nearly halves this projection's device time.
        self.decode_down_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(11, 10),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def __call__(self, hidden_states):
        gate = ttnn.linear(hidden_states, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        up = ttnn.linear(hidden_states, self.up_proj, compute_kernel_config=self.compute_kernel_config)
        intermediate = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        gate.deallocate(True)
        up.deallocate(True)
        down_program_config = self.decode_down_program_config if int(intermediate.shape[2]) == ttnn.TILE_SIZE else None
        output = ttnn.linear(
            intermediate,
            self.down_proj,
            program_config=down_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        intermediate.deallocate(True)
        return output

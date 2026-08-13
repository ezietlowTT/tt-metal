# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM serving adapter for Muse-Glimmer-30B (text-only, C-minimal).

Phase 1 of doc/vllm_integration/BUILD_PLAN.md: greedy, batch-1, HOST sampling, no
DFlash. Subclasses HybridAttentionForCausalLM to reuse get_kv_cache_spec (Muse's
config carries text_config.layer_types). vLLM owns the paged KV cache; the adapter
injects it into the model and drives prefill via the existing paged prefill path and
decode via the new page-table-aware vllm_decode_forward.

STATUS: first implementation, not yet device-verified end-to-end through vLLM.
Diagnostic logging is intentionally verbose to make the first bring-up informative.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM
from models.demos.muse_glimmer.tt.model import MuseGlimmerModel
from models.demos.muse_glimmer.tt.model_config import MuseGlimmerModelArgs

from loguru import logger

KV_PAGE_SIZE = 64


def _build_muse_model(mesh_device, model_path, max_seq_len):
    """Build MuseGlimmerModel with vLLM-owned KV cache (create_kv_cache=False).

    Honors MUSE_VLLM_N_LAYERS to build a reduced target (fast bring-up); when set,
    layer_types are trimmed to the first N (keeping their sliding/full pattern).
    """
    hf_config = MuseGlimmerModelArgs.load_hf_config(model_path)
    model_args = MuseGlimmerModelArgs.from_hf_config(hf_config)
    model_args._hf_text_config = hf_config.text_config

    n_layers_env = os.getenv("MUSE_VLLM_N_LAYERS")
    if n_layers_env:
        n = int(n_layers_env)
        lt = tuple(model_args.layer_types[:n])
        # Guarantee at least one of each attention kind exists in the reduced set.
        object.__setattr__(model_args, "layer_types", lt) if hasattr(model_args, "__setattr__") else None
        model_args.layer_types = lt
        model_args.num_hidden_layers = n
        if hasattr(model_args, "layer_rope_theta"):
            model_args.layer_rope_theta = tuple(model_args.layer_rope_theta[:n])
        logger.info(f"[muse-vllm] reduced target: {n} layers, layer_types={lt}")

    state_dict = MuseGlimmerModelArgs.load_state_dict(model_path)
    model = MuseGlimmerModel(
        device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        attention_dtype=ttnn.bfloat8_b,
        mlp_dtype=ttnn.bfloat4_b,
        lm_head_dtype=ttnn.bfloat8_b,
        tensor_cache_path=str(model_args.weight_cache_path(model_path, ttnn.bfloat8_b)),
        mlp_cache_path=str(model_args.weight_cache_path(model_path, ttnn.bfloat4_b)),
        max_seq_len=max_seq_len,
        create_kv_cache=False,  # vLLM owns the KV cache
        kv_cache_dtype=ttnn.bfloat16,
    )
    return model, model_args


class MuseGlimmerForConditionalGeneration(HybridAttentionForCausalLM):
    """C-minimal text-only vLLM adapter for Muse-Glimmer-30B (greedy, batch-1)."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }
    is_multimodal = False

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        from transformers import AutoTokenizer

        model_path = getattr(hf_config, "_name_or_path", None) or os.getenv("MUSE_TARGET_DIR")
        logger.info(f"[muse-vllm] initialize_vllm_model path={model_path} max_seq_len={max_seq_len}")
        model, model_args = _build_muse_model(mesh_device, model_path, max_seq_len)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self = cls([model], [model_args], mesh_device, tokenizer=tokenizer)
        self._model_path = model_path
        self._max_seq_len = max_seq_len
        return self

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(self._model_path, ttnn.bfloat8_b)

    # --- warmup: no-op (tt_transformers warmup assumes its Transformer internals) ---
    def warmup_model_prefill(self, *args, **kwargs):
        return

    def warmup(self, *args, **kwargs):
        return

    # --- KV cache: vLLM owns it; allocate Muse-native layout and inject into layers ---
    def allocate_kv_cache_per_layer(self, per_layer_specs):
        model = self.model[0]
        num_kv_heads = model.hf_config.num_key_value_heads
        head_dim = model.hf_config.head_dim
        # num_blocks from the spec (first dim is vLLM's block count); log to verify layout.
        first_shape = tuple(per_layer_specs[0][0])
        logger.info(f"[muse-vllm] allocate_kv_cache_per_layer: vLLM spec shape={first_shape} "
                    f"n_layers={len(per_layer_specs)} kv_heads={num_kv_heads} head_dim={head_dim}")
        num_blocks = int(first_shape[0])
        shape = [num_blocks, num_kv_heads, KV_PAGE_SIZE, head_dim]

        def _alloc():
            t = ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.fill(t, 0.0, output_tensor=t)
            return t

        per_layer = []
        for layer_idx in range(len(per_layer_specs)):
            kv = [_alloc(), _alloc()]
            model.layers[layer_idx].self_attn.kv_cache = kv
            per_layer.append(kv)
        model.tt_kv_cache = per_layer
        # Generator expects list[submesh][layer][k,v]; single submesh here.
        return [per_layer]

    # --- helpers ---
    def _page_table_tt(self, page_table):
        pt = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
        pt = pt.to(torch.int32)
        if pt.dim() == 1:
            pt = pt.unsqueeze(0)
        return ttnn.from_torch(
            pt, device=self.mesh_device, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # --- prefill: reuse the model's paged prefill path with vLLM's block table ---
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, start_pos=0,
                        prompt_lens=None, enable_trace=False, **kwargs):
        model = self.model[0]
        ids = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        ids = ids.reshape(-1)
        S = int(ids.shape[0])
        chunk_start = int(start_pos) if not torch.is_tensor(start_pos) else int(start_pos.reshape(-1)[0])
        logger.info(f"[muse-vllm] prefill_forward S={S} chunk_start={chunk_start}")
        hidden = model.embed_input_ids(ids.reshape(1, S))  # [1,1,S,hidden]
        packed = {
            "page_table": self._page_table_tt(page_table),
            "chunk_start": chunk_start,
            "logical_length": S,
        }
        logits = model(hidden, is_decode=False, packed=packed, last_token_only=True)
        out = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).squeeze(0).float()
        return out

    # --- decode: single-token, page-table-aware, host sampling ---
    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None,
                       enable_trace=False, read_from_device=True, sampling_params=None, **kwargs):
        model = self.model[0]
        ids = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        ids = ids.reshape(-1)
        pos = start_pos if torch.is_tensor(start_pos) else torch.as_tensor([start_pos])
        pos = pos.reshape(-1).to(torch.int32)
        cur = int(pos[0].item())
        logger.info(f"[muse-vllm] decode_forward token0={int(ids[0])} cur_pos={cur}")
        hidden = model.embed_input_ids(ids[:1].reshape(1, 1))  # [1,1,1,hidden]
        position_idx = ttnn.from_torch(
            torch.tensor([[cur]], dtype=torch.int32), device=self.mesh_device,
            dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cur_pos_tt = ttnn.from_torch(
            pos[:1], device=self.mesh_device, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        packed = {
            "vllm_mode": True,
            "position_idx": position_idx,
            "cur_pos": cur_pos_tt,
            "page_table": self._page_table_tt(page_table),
        }
        logits = model(hidden, is_decode=True, packed=packed, last_token_only=False)
        out = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).squeeze(0).float()
        return out

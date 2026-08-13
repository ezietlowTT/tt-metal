# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM serving adapter for Muse-Glimmer-30B (text-only path).

STATUS: SCAFFOLD — not yet functional. The environment (vLLM `empty` target +
vllm-tt-plugin) imports and this arch is registered, but the prefill/decode bridge
to vLLM-owned paged KV caches is not implemented. See
``doc/vllm_integration/work_log.md`` for the full plan.

Unlike Llama/Qwen/Mistral (thin wrappers over the shared tt_transformers.Transformer),
Muse is a separate model (``tt/model.py::MuseGlimmerModel``) with its own paged KV cache
and the native DFlash speculative-decode server. This adapter must call MuseGlimmerModel
directly and externalize its ``tt_kv_cache`` + ``page_table`` so vLLM owns them. Native
DFlash is intentionally dropped in the vLLM path.
"""

from __future__ import annotations

from models.tt_transformers.tt.generator import Generator


class MuseGlimmerForConditionalGeneration(Generator):
    """Text-only vLLM adapter for Muse-Glimmer-30B. Multimodal inputs are rejected."""

    # Keep in sync with get_kv_cache_spec: Muse mixes sliding-window and full-attention
    # layers, so serving must use vLLM hybrid per-layer KV specs.
    supports_async_decode = False  # do NOT flip until the async split + stale-input tests pass
    is_multimodal = False

    @classmethod
    def initialize_vllm_model(cls, vllm_config, mesh_device, *args, **kwargs):
        # TODO(bridge): build MuseGlimmerModel via tt.common.create_tt_model with KV-cache
        # allocation deferred to vLLM (allocate_vllm_kv_cache), load the datatype policy
        # (BFP8 attn/lm_head + BFP4 MLP), and wire the tokenizer / model args.
        raise NotImplementedError(
            "Muse vLLM adapter not implemented yet — see doc/vllm_integration/work_log.md"
        )

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        # TODO(bridge): emit per-layer KV specs. Muse layers are sliding-window OR full
        # attention (see tt/model_config.py); use vLLM hybrid attention specs, page size
        # KV_PAGE_SIZE=64, cache dtype BF16 (matches the standalone model).
        raise NotImplementedError

    def prefill_forward(self, *args, **kwargs):
        # TODO(bridge): embeddings + MuseGlimmerModel(..., is_decode=False,
        # packed={page_table=<vLLM block table>, chunk_start, logical_length}). Chunk to
        # PREFILL_CHUNK_SIZE=2048 internally but pass the logical length through masks/
        # positions/output slicing. Must accept non-chunk-aligned prompt lengths.
        raise NotImplementedError

    def decode_forward(self, *args, **kwargs):
        # TODO(bridge): single-token step, is_decode=True, over the vLLM page table +
        # positions, via the traced decode path. On-device split sampling; return device
        # tensors when read_from_device=False.
        raise NotImplementedError

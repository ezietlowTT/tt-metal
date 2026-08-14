# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM serving adapter for Muse-Glimmer-30B (text-only, C-minimal).

Phase 1 of doc/vllm_integration/BUILD_PLAN.md: greedy, batch-1, HOST sampling, no
DFlash. Subclasses HybridAttentionForCausalLM to reuse get_kv_cache_spec (Muse's
config carries text_config.layer_types). vLLM owns the paged KV cache; the adapter
injects it into the model and reuses the OG server's chunked+packed paged paths
(server.py _append_prompt_tokens / _packed_verify_inputs) for prefill and decode.

STATUS: prefill + decode forward plumbing validated on device against a reduced
2-layer target (tests/test_vllm_adapter_smoke.py). Not yet run through the full
vLLM server / full 52-layer accuracy path.
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


def _dbg(msg):
    # crash-survivable marker (survives a hard device abort that loses stdout).
    try:
        with open("/tmp/smoke_progress.txt", "a") as f:
            f.write(f"[adapter] {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


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

    def __init__(self, *args, **kwargs):
        # `*args, **kwargs` (not Generator's explicit signature) so vLLM's
        # _check_vllm_model_init(supports_kw "vllm_config") passes and
        # is_text_generation_model() classifies this bridge as generative.
        super().__init__(*args, **kwargs)

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
        _dbg(f"init: start path={model_path} max_seq_len={max_seq_len}")
        model, model_args = _build_muse_model(mesh_device, model_path, max_seq_len)
        _dbg("init: MuseGlimmerModel built")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        _dbg("init: tokenizer loaded; constructing Generator")
        self = cls([model], [model_args], mesh_device, tokenizer=tokenizer)
        _dbg("init: done")
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

    # --- protocol shims: exist only so vLLM's is_text_generation_model() classifies
    # this as a generative model (Muse isn't in vLLM's upstream registry, so — like
    # Gemma4 — the class itself must present the interface). The TT runner never calls
    # these; it drives prefill_forward / decode_forward. ---
    def embed_input_ids(self, input_ids, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; embeddings are computed on "
            "TT inside prefill_forward / decode_forward."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; the TT runner invokes "
            "prefill_forward / decode_forward, not forward()."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; logits are produced on TT "
            "and surfaced through prefill_forward / decode_forward."
        )

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
        _dbg(f"alloc: start specs={len(per_layer_specs)} num_blocks={num_blocks} spec0={first_shape}")
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
        _dbg(f"alloc: done ({len(per_layer)} layers x [k,v])")
        # Generator expects list[submesh][layer][k,v]; single submesh here.
        return [per_layer]

    # ============================================================================
    # Prefill/decode reuse the OG server's chunked+packed paged paths (server.py
    # _append_prompt_tokens / _packed_verify_inputs), minus the DFlash anchor logic.
    # The ONLY vLLM adaptation: (1) use vLLM's block table instead of Muse's identity
    # page table, and (2) make packed_kv_update's position_idx PHYSICAL
    # (page_table[pos//64]*64 + pos%64) since that op is not page-table-aware — while
    # rope_packed / cur_pos stay logical (rope lookup + page-table-aware SDPA read).
    # ============================================================================
    PHYS_VERIFY = 32  # PHYSICAL_VERIFY_TOKENS
    PAGE = KV_PAGE_SIZE
    PREFILL_CHUNK = 2048

    def _page_table_tt(self, page_table_torch):
        return ttnn.from_torch(
            page_table_torch.to(torch.int32), device=self.mesh_device, dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _phys_pos(self, page_table_torch, logical_pos):
        # physical flat position for a logical position under vLLM's block table.
        blk = page_table_torch[0, logical_pos // self.PAGE].item()
        return int(blk) * self.PAGE + (logical_pos % self.PAGE)

    def _normalized_embeddings(self, model, token_ids):
        emb = model.raw_token_embeddings(token_ids).float()
        emb = emb * torch.rsqrt(emb.square().mean(dim=-1, keepdim=True) + model.embed_norm_eps)
        return emb.to(torch.bfloat16)

    def _packed_inputs(self, model, token_ids, current_position, page_table_torch):
        """vLLM-adapted _packed_verify_inputs: physical write idx, logical rope/cur_pos.

        The packed decode treats the PHYS_VERIFY (32) positions as 32 decode users, so
        the SDPA page_table must have 32 rows (OG: decode_page_ids.repeat(32, 1)). Rows
        are identical here (single sequence).
        """
        real_p = int(token_ids.numel())
        P = self.PHYS_VERIFY
        page_table_tt = self._page_table_tt(page_table_torch.repeat(P, 1))
        emb = self._normalized_embeddings(model, token_ids.reshape(1, real_p))
        emb = torch.nn.functional.pad(emb, (0, 0, 0, P - real_p))
        hidden = ttnn.from_torch(emb.unsqueeze(0), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        logical = torch.arange(current_position, current_position + real_p, dtype=torch.int32)
        # physical positions for the KV write (packed_kv_update is not page-table-aware)
        phys = torch.zeros(1, P, dtype=torch.int32)
        for i in range(real_p):
            phys[0, i] = self._phys_pos(page_table_torch, current_position + i)
        position_idx = ttnn.from_torch(phys, device=self.mesh_device, dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # logical positions for rope + causal/sliding masks (SDPA read is page-table-aware)
        logical_full = torch.zeros(1, P, dtype=torch.int32)
        logical_full[0, :real_p] = logical
        logical_idx = ttnn.from_torch(logical_full, device=self.mesh_device, dtype=ttnn.uint32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cur = torch.full((P,), -1, dtype=torch.int32)
        cur[:real_p] = logical
        cur_pos = ttnn.from_torch(cur, device=self.mesh_device, dtype=ttnn.int32,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rope_packed = {}
        for lt, (cos_c, sin_c) in model.rope_caches_2d.items():
            cos = ttnn.unsqueeze_to_4D(ttnn.embedding(logical_idx, cos_c, layout=ttnn.TILE_LAYOUT))
            sin = ttnn.unsqueeze_to_4D(ttnn.embedding(logical_idx, sin_c, layout=ttnn.TILE_LAYOUT))
            rope_packed[lt] = (cos, sin)
        packed = {
            "p": P, "real_p": real_p, "position_idx": position_idx, "cur_pos": cur_pos,
            "page_index": current_position // self.PAGE, "page_offset": current_position % self.PAGE,
            "rope_packed": rope_packed, "page_table": page_table_tt, "retain_tail": False,
        }
        return hidden, packed

    def prefill_forward(self, tokens, page_table=None, kv_cache=None, start_pos=0,
                        prompt_lens=None, enable_trace=False, **kwargs):
        model = self.model[0]
        ids = (tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)).reshape(-1).to(torch.long)
        pt_torch = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
        if pt_torch.dim() == 1:
            pt_torch = pt_torch.unsqueeze(0)
        pt_tt = self._page_table_tt(pt_torch)
        base = int(start_pos) if not torch.is_tensor(start_pos) else int(start_pos.reshape(-1)[0])
        S = int(ids.shape[0])
        logger.info(f"[muse-vllm] prefill_forward S={S} base_pos={base}")

        offset = 0
        last_logits = None
        while offset < S:
            position = base + offset
            remaining = S - offset
            needs_next = None  # set per branch
            if position % self.PAGE == 0 and remaining >= self.PAGE:
                chunk_len = min(self.PREFILL_CHUNK, (remaining // self.PAGE) * self.PAGE)
                chunk_ids = ids[offset:offset + chunk_len].reshape(1, -1)
                needs_next = offset + chunk_len == S
                last_logits = model(
                    model.embed_input_ids(chunk_ids), is_decode=False,
                    packed={"page_table": pt_tt, "chunk_start": position, "logical_length": chunk_len},
                    last_token_only=needs_next, compute_logits=needs_next,
                )
                offset += chunk_len
            else:
                until_aligned = self.PAGE - (position % self.PAGE)
                append_len = min(self.PHYS_VERIFY, remaining, until_aligned)
                seg_ids = ids[offset:offset + append_len]
                hidden, packed = self._packed_inputs(model, seg_ids, position, pt_torch)
                needs_next = offset + append_len == S
                last_logits = model(hidden, is_decode=True, packed=packed,
                                    last_token_only=needs_next, compute_logits=needs_next)
                offset += append_len

        out = ttnn.to_torch(ttnn.get_device_tensors(last_logits)[0]).float().reshape(-1, model.vocab_size)
        # Plugin host sampler does tt_out[rows, -1, :], so return [batch=1, seq=1, vocab]
        # with the REAL last prompt token's logits at the (only) seq position.
        return out[-1].reshape(1, 1, model.vocab_size)

    def decode_forward(self, tokens, start_pos, page_table=None, kv_cache=None,
                       enable_trace=False, read_from_device=True, sampling_params=None, **kwargs):
        model = self.model[0]
        ids = (tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)).reshape(-1).to(torch.long)
        pos = (start_pos if torch.is_tensor(start_pos) else torch.as_tensor([start_pos])).reshape(-1)
        cur = int(pos[0].item())
        pt_torch = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
        if pt_torch.dim() == 1:
            pt_torch = pt_torch.unsqueeze(0)
        hidden, packed = self._packed_inputs(model, ids[:1], cur, pt_torch)
        logits = model(hidden, is_decode=True, packed=packed, last_token_only=True, compute_logits=True)
        out = ttnn.to_torch(ttnn.get_device_tensors(logits)[0]).float().reshape(-1, model.vocab_size)
        return out[0].reshape(1, 1, model.vocab_size)  # [batch=1, seq=1, vocab]

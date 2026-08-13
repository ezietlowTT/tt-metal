# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged batch-1 packed verification attention for Muse Glimmer."""

import ttnn

from .packed_cache import packed_kv_update

from .operations import (
    apply_output_projection,
    apply_per_head_norm,
    apply_qkv_projection,
    apply_rope,
    concat_heads,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights


PACKED_SDPA_SHORT_MAX_CORES = 16
PACKED_SDPA_LONG_MAX_CORES = 64
PACKED_SDPA_LONG_CONTEXT = 2048


def vllm_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_device,
    position_idx,
    cur_pos,
    page_table,
    rope_packed=None,
):
    """Single-token decode over a vLLM-owned paged KV cache.

    Unlike ``packed_decode_forward`` (which uses ``packed_kv_update`` and assumes
    an identity page table), this writes the new K/V with the page-table-aware
    ``ttnn.experimental.paged_update_cache`` so it respects vLLM's non-identity
    block table. No DFlash packed metadata and no sliding-window tail state — the
    read-side ``sliding_window_size`` on the SDPA op trims the window instead.
    """
    if kv_cache is None:
        raise ValueError("vLLM decode requires a KV cache")
    if int(hidden_states.shape[2]) != 1:
        raise ValueError("vllm_decode_forward expects a single token")

    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    l1 = ttnn.L1_MEMORY_CONFIG

    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=l1)
    q, k, v = split_qkv_heads_prefill(xqkv, config, memory_config=l1)
    ttnn.deallocate(xqkv)
    q = apply_per_head_norm(q, None, config.rms_norm_eps, with_scale=False, memory_config=l1)
    k = apply_per_head_norm(k, None, config.rms_norm_eps, with_scale=False, memory_config=l1)

    if rope_packed is None:
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    else:
        cos, sin = rope_packed
        owns_rope = False
    q = apply_rope(q, cos, sin, memory_config=l1)
    k = apply_rope(k, cos, sin, memory_config=l1)
    if owns_rope:
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

    k_cache, v_cache = kv_cache
    # [1, n_kv_heads, 1, head_dim] -> [1, 1, n_kv_heads, head_dim] (the "1BKD" layout
    # paged_update_cache expects for a single decode token).
    k_1bkd = ttnn.permute(k, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_1bkd = ttnn.permute(v, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(k)
    ttnn.deallocate(v)
    if k_1bkd.dtype != k_cache.dtype:
        old_k, old_v = k_1bkd, v_1bkd
        k_1bkd = ttnn.typecast(k_1bkd, k_cache.dtype)
        v_1bkd = ttnn.typecast(v_1bkd, v_cache.dtype)
        ttnn.deallocate(old_k)
        ttnn.deallocate(old_v)
    ttnn.experimental.paged_update_cache(
        k_cache, k_1bkd, update_idxs_tensor=cur_pos, page_table=page_table
    )
    ttnn.experimental.paged_update_cache(
        v_cache, v_1bkd, update_idxs_tensor=cur_pos, page_table=page_table
    )
    ttnn.deallocate(k_1bkd)
    ttnn.deallocate(v_1bkd)

    # q: [1, n_heads, 1, head_dim] -> [1, 1, n_heads, head_dim] (one decode user).
    q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=PACKED_SDPA_LONG_MAX_CORES,
    )
    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        cur_pos_tensor=cur_pos,
        scale=config.attention_scale,
        sliding_window_size=config.sliding_window if config.is_sliding else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
    )
    ttnn.deallocate(q)
    output = concat_heads(output, is_decode_mode=True, memory_config=l1)
    return apply_output_projection(output, weights, hidden_states)


def packed_decode_forward(
    hidden_states,
    cos_cache,
    sin_cache,
    weights: AttentionWeights,
    kv_cache,
    config,
    mesh_device,
    position_idx,
    cur_pos,
    page_table,
    page_index,
    page_offset,
    packed_p,
    real_p,
    rope_packed=None,
    retain_tail=True,
):
    """Verify a physical packed block while every query shares one paged cache."""
    if kv_cache is None:
        raise ValueError("Packed verification requires a KV cache")
    if int(hidden_states.shape[2]) != packed_p:
        raise ValueError("Packed verification metadata does not match its input")

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    l1 = ttnn.L1_MEMORY_CONFIG

    xqkv = apply_qkv_projection(hidden_states, weights, memory_config=l1)

    q, k, v = split_qkv_heads_prefill(xqkv, config, memory_config=l1)
    ttnn.deallocate(xqkv)
    q = apply_per_head_norm(q, None, config.rms_norm_eps, with_scale=False, memory_config=l1)
    k = apply_per_head_norm(k, None, config.rms_norm_eps, with_scale=False, memory_config=l1)

    if rope_packed is None:
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, cos_cache, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(position_idx, sin_cache, layout=ttnn.TILE_LAYOUT))
        owns_rope = True
    else:
        cos, sin = rope_packed
        owns_rope = False
    q = apply_rope(q, cos, sin, memory_config=l1)
    k = apply_rope(k, cos, sin, memory_config=l1)
    if owns_rope:
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

    k_cache, v_cache = kv_cache
    write_k = ttnn.slice(
        k,
        [0, 0, 0, 0],
        [1, num_kv_heads, real_p, head_dim],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    write_v = ttnn.slice(
        v,
        [0, 0, 0, 0],
        [1, num_kv_heads, real_p, head_dim],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if write_k.dtype != k_cache.dtype:
        old_write_k, old_write_v = write_k, write_v
        write_k = ttnn.typecast(write_k, k_cache.dtype)
        write_v = ttnn.typecast(write_v, v_cache.dtype)
        ttnn.deallocate(old_write_k)
        ttnn.deallocate(old_write_v)
    packed_kv_update(
        k_cache,
        v_cache,
        write_k,
        write_v,
        position_idx,
        count=real_p,
    )
    pending_tail = (write_k, write_v) if config.is_sliding and retain_tail else None
    if pending_tail is None:
        ttnn.deallocate(write_k)
        ttnn.deallocate(write_v)

    q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    # Treat packed positions as decode users which all map to the same physical
    # pages. Their individual cur_pos values supply exact causal/sliding masks,
    # eliminating the O(P * heads * max_seq_len) additive mask.
    q = ttnn.permute(q, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=64,
        exp_approx_mode=False,
        max_cores_per_head_batch=(
            PACKED_SDPA_LONG_MAX_CORES if page_index * 64 >= PACKED_SDPA_LONG_CONTEXT else PACKED_SDPA_SHORT_MAX_CORES
        ),
    )
    output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        cur_pos_tensor=cur_pos,
        scale=config.attention_scale,
        sliding_window_size=config.sliding_window if config.is_sliding else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=program_config,
    )
    ttnn.deallocate(q)
    output = concat_heads(output, is_decode_mode=True, memory_config=l1)
    return apply_output_projection(output, weights, hidden_states), pending_tail

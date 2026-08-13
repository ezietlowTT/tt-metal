# Muse × vLLM — build plan for all three contributions

Sequenced by dependency. Goal: land three upstream contributions so Muse serves through
vLLM **with** its OG wins (DFlash decode speed, sliding-window efficiency), then submit as
PRs. Repos: `tenstorrent/vllm@dev` (plugin + spec-decode + sliding-window) and
`tenstorrent/tt-metal` (Muse demo adapter). Auth: `ezietlowTT` (has `repo` scope; fork
pattern already used for tt-metal).

Dependency graph:  **C-minimal → C-full → A** ;  **B** is parallel (lands anytime after C for testing).

---

## Phase 0 — Shared test harness (0.5 day)
- **Reduced target**: a Muse config with 1 sliding + 1 full layer (one of each unique kind)
  for fast bring-up loops. Add an env/flag to `create_tt_model` to cap `num_hidden_layers`.
- **Launch path**: `run_vllm_server` is autoport-oriented; either adapt it to a
  `models/demos/` model or drive `vllm serve <HF dir>` directly with the plugin + on-device
  sampling config. Decide in Phase 1.
- **Device hygiene**: between runs kill leftover `EngineCore`/`vllm.entrypoints` procs
  (they hold chip locks); NO Tracy/profiler in vLLM stages (wedges machines).

## Phase 1 — C-minimal: non-speculative Muse vLLM server (2–4 days)  ← START HERE
Deliverable: a working OpenAI-compatible vLLM server for Muse, greedy, `max_num_seqs=1`,
host sampling. Proves the model in vLLM and unlocks A's target side.

Implement `MuseGlimmerForConditionalGeneration(HybridAttentionForCausalLM)` in
`models/demos/muse_glimmer/tt/generator_vllm.py`:
- `initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, ...)`:
  build `MuseGlimmerModel` with `create_kv_cache=False`; load `LazyStateDict` + tensor
  caches + tokenizer; return `cls([model], [model_args], mesh_device, tokenizer=tok)`.
- `get_kv_cache_spec`: REUSE hybrid base (Muse has `text_config.layer_types`; all
  `FullAttentionSpec`, kv_heads=2, head=128, block from cache_config).
- `allocate_kv_cache_per_layer(specs)`: allocate Muse layout
  `[num_blocks, 2, 64, 128]` bf16/TILE/DRAM per layer; assign into
  `layer.self_attn.kv_cache` and `model.tt_kv_cache`; return `list[submesh][layer][k,v]`.
- `prefill_forward(tokens, page_table, kv_cache, start_pos, prompt_lens, enable_trace, ...)`:
  inject caches; vLLM `page_table`→ttnn int32; embed (`model._embedding_rows`); call
  `model(hidden, is_decode=False, packed={page_table, chunk_start=start_pos,
  logical_length=S}, last_token_only=True)`; chunk internally to 2048; return host logits.
- `decode_forward(tokens, page_table, kv_cache, start_pos, ...)`: single token,
  `is_decode=True` via the **standard** `paged_scaled_dot_product_attention_decode` path
  already in `tt/attention/decode.py` (do NOT retain `prefill_tail`); return host logits.
- Stub `warmup*` to no-op; `model_capabilities = {supports_async_decode: False,
  supports_sample_on_device: False, supports_prefix_caching: False}`.
- Arch registration in `platform.py` (DONE).

**Gates (reduced target → full 52-layer):** server launches/loads/allocates without crash →
single coherent completion → non-page-aligned prompt length OK → multi-request smoke
(max_num_seqs=1) → qualitative + PCC vs HF and vs OG native server on shared prompts.
**Risks:** page_table dtype/shape mismatch; embed path; forward return-type contract;
autoport-oriented runner.

## Phase 2 — C-full: production serving contract (3–5 days)
- **On-device split sampling**: add `tt_out_tok` split to Muse (feed sampled token into the
  persistent decode input exactly once/token); implement `read_decode_output(async_read=)` +
  `process_decode_output_host`; set `supports_sample_on_device=True`.
- **Traced decode**: enable `enable_trace` replay via `$tt-enable-tracing`; add
  stale-token/current-position tests; only then `supports_async_decode=True`.
- **Batch/concurrency**: audit `packed_kv_update` / `decode.py` for batch dim. If hard
  batch-1, record as a physical limit with evidence; else serve up to `max_num_seqs=32`.
- **Evidence**: `run_vllm_server` sampling + qualitative + benchmark stages; degenerate-
  output check; TTFT/TPOT/ITL + decode t/s/u.

## Phase 3 — A: TT speculative decoding (flagship, 2–4 weeks)
- Register Muse assistant as a TT draft causal-LM (`TTMuseGlimmerAssistantForCausalLM` →
  `tt/dflash` wrapped).
- `platform.py:587`: replace `assert not speculative_config` with a capability gate
  (`model_capabilities["supports_tt_spec_decode"]`).
- `model_runner.py`: add draft-propose + **multi-token target-verify** decode branch;
  reuse existing `lane_scheduler` draft plumbing (`scheduled_spec_decode_tokens`,
  `update_draft_token_ids[_in_output]`).
- `worker.py`: host draft proposer beside target on the mesh. **DRAM budget check**:
  target ~17 GB caches + assistant ~1.7 GB on one p150a — may cap `max_model_len`.
- **A1 first** (standard `DraftModelProposer` + vLLM rejection sampler — general), then
  **A2** (native packed-verify extension point for bit-exact DFlash behavior).
- **Gates:** acceptance rate + decode t/s vs OG native; correctness + determinism through
  vLLM; qualitative parity.

## Phase 4 — B: sliding-window paged KV groups (1–2 weeks, parallelizable)
- Fix TT sliding-group block-table indexing to window-relative/rolling
  (`phys = (pos % window)//block`): either in plugin page-table construction
  (`model_runner._block_tables_per_layer`) or as a `sliding_window` arg to
  `paged_update_cache`/`paged_sdpa_decode`.
- Restore `SlidingWindowSpec` in `HybridAttentionForCausalLM.get_kv_cache_spec`; flip
  `_HYBRID_KV_CACHE_GROUPS_ENABLED=True`.
- **Gates:** correctness at positions > sliding_window across a page boundary; measured KV
  memory reduction; no regression on Gemma3/4 / Mistral. Careful — touches shared code.

---

## PR structure (submit after build, in order)
1. **tt-metal PR** — Muse demo vLLM adapter: `tt/generator_vllm.py` (functional),
   assistant draft wrapper, `doc/vllm_integration/*`. (Phase 1–2 output.)
2. **vllm PR #1 (C)** — Muse arch registration in `platform.py` + any plugin glue the
   adapter needs. Depends on (1). Mark ready once C-full gates pass.
3. **vllm PR #2 (A)** — capability-gated TT speculative decoding + model_runner draft/verify
   branch + worker draft hosting. Flagship; largest review.
4. **vllm PR #3 (B)** — sliding-window paged KV groups. Independent; can open in parallel.
Each PR carries the `vllm-integration` skill's "evidence to leave" (successful
`run_vllm_server` invocation, served context, non-aligned prompt evidence, sampling +
qualitative + benchmark artifacts, determinism, capability-flag proof).

## Effort summary
| Phase | Deliverable | Effort | Gates |
|---|---|---|---|
| 0 | Test harness / reduced target | 0.5 d | reduced config loads |
| 1 | C-minimal server (greedy, bs1, host-sample) | 2–4 d | coherent completion, PCC vs HF+OG |
| 2 | C-full (device sampling, trace, batch) | 3–5 d | run_vllm_server evidence |
| 3 | A — TT spec decode (DFlash-as-draft) | 2–4 wk | acceptance + decode t/s vs OG |
| 4 | B — sliding-window paged KV | 1–2 wk | correctness past window; mem savings |

Total: ~C = 1–1.5 wk to a solid serving PR; A + B ≈ 3–6 wk for the two upstream features.
Recommended kickoff: **Phase 1 (C-minimal)** — highest ratio of value to effort, unblocks
everything else, and is now known-tractable (Muse decode already uses the standard paged
SDPA op).

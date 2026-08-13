# Upstream contributions to tenstorrent/vllm — "best of all worlds" for Muse

Goal: keep the OG Muse repo's wins (native DFlash speculative decode, sliding-window
efficiency, high prefill throughput) **while** serving through vLLM. The path is to
contribute the two capabilities the TT plugin explicitly lacks today, so Muse (and every
TT model) gets them inside the standard serving stack.

All file references are in the checked-out fork at `/home/mando222/museglimmer/vllm`
(branch `dev`, vllm `0.1.dev1+g7c99bd3b8.empty`) unless noted. Evidence gathered
2026-08-12 against tt-metal @ `1c1b7c9c3` (2026-06-25).

---

## Contribution A — TT speculative decoding (FLAGSHIP)  🎯

Turn Muse's native DFlash draft+verify into an upstream TT capability: draft-model
speculative decoding on the TT backend, via vLLM's existing v1 spec-decode framework.

### Current state (evidence)
- **Hard-gated off**: `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py:587`
  ```python
  assert not vllm_config.speculative_config, \
      "Speculative decoding is not yet supported for TT backend"
  ```
- **Framework already present in the fork**: `vllm/v1/spec_decode/`
  (`DraftModelProposer`, `EagleProposer`/`SpecDecodeBaseProposer`, `ngram_proposer`,
  `medusa`, `suffix_decoding`). `DraftModelProposer._get_model()` loads the draft model
  through the standard `get_model(vllm_config)` — i.e. through the TT model registry.
- **Scheduler plumbing already present in the plugin**:
  `lane_scheduler.py` handles `scheduled_spec_decode_tokens`, `update_draft_token_ids`,
  `update_draft_token_ids_in_output`. `model_runner.py:134` already stores
  `self.speculative_config`.
- **Missing piece**: the TT worker/model-runner decode path assumes exactly one token
  per request (`model_runner.py:1119` "TT does not support speculative decoding in this
  path"). There is no draft-propose + multi-token target-verify execution on device.

### Why Muse is the ideal first draft-model
Muse **already implements native draft+verify on TT** in the OG repo — this is a working
existence proof that TT hardware does speculative decode efficiently:
- draft: the 5 GB assistant proposes a 16-token block (`tt/dflash/`);
- verify: the target checks them in a 32-token physical tile via the custom
  `packed_kv_update` kernel (`tt/attention/packed_cache.py`);
- accept/commit: anchor-commit replay.
Measured acceptance bursts of 8–14 tokens/verify → ~45–59 tok/s decode (see BENCHMARKS.md).

### Design (two options)
1. **Standard path (recommended first):** register Muse's assistant as a normal TT
   causal-LM draft model and let vLLM's `DraftModelProposer` + rejection sampler drive
   draft/verify. Requires the TT model-runner to (a) run the draft model N steps, (b) run
   a **multi-token target verify** decode (the target already supports a packed multi-token
   verify — expose it as an N-token decode over the vLLM page table), (c) return per-token
   logits for vLLM's rejection sampler. Loses Muse's *exact* native acceptance rule but is
   fully general and standards-compliant.
2. **Native-verify path (preserves exact DFlash behavior):** add a `TTProposer` +
   custom-verify extension point so a model supplies its own packed verify/accept step.
   Keeps Muse's numbers bit-for-bit; more custom surface.

### Concrete change set
- `platform.py:587` — replace the hard assert with a capability gate:
  allow `speculative_config` when the target model class declares
  `model_capabilities["supports_tt_spec_decode"] = True` (and method is `draft_model`/
  `ngram`).
- `model_runner.py` — add a decode branch that, per step: builds draft inputs, calls the
  draft proposer, builds an N-token verify decode input (page table + positions for the
  N speculated tokens), calls `target.decode_forward` in multi-token mode, and returns the
  per-position logits. Reuse the existing `scheduled_spec_decode_tokens` /
  `update_draft_token_ids` plumbing in `lane_scheduler.py`.
- `worker.py` — load/host the draft proposer alongside the target (both TT models on the
  same mesh; watch DRAM budget — Muse target ~17 GB caches + assistant ~1.7 GB).
- Draft-model registration: `TTMuseGlimmerAssistantForCausalLM` →
  `models/demos/muse_glimmer/tt/dflash` wrapped as a TT causal-LM.

### Effort / impact
- **Effort: large** (multi-week; touches worker, model-runner decode, sampling, KV for a
  second model). The scheduler layer being done is a real head-start.
- **Impact: very high** — every TT model gains spec decode; Muse keeps its headline decode
  win *inside* vLLM with continuous batching. It's on the plugin's own roadmap (the assert
  literally says "not yet supported").

---

## Contribution B — sliding-window paged KV groups on TT (memory optimization)

### Current state (evidence)
- `platform.py:544 support_hybrid_kv_cache() -> True` enables HMA, but
  `HybridAttentionForCausalLM.get_kv_cache_spec`
  (`models/tt_transformers/tt/generator_vllm.py:201`) **deliberately emits
  `FullAttentionSpec` for every layer**, with this comment:
  > SlidingWindowSpec temporarily disabled: TT decode passes the absolute position to
  > paged_update_cache / paged_sdpa_decode, but vLLM zero-pads the sliding group's
  > page_table past sliding_window/block_size entries, so positions beyond the window
  > collapse onto physical block 0 and silently corrupt the cache. … the SDPA op's own
  > sliding_window_size kwarg still trims attention correctly on the read side.
- **The workaround is correct**, just memory-heavy: sliding layers get a full
  `max_model_len` KV cache instead of a `sliding_window`-sized one.

### Root cause
TT decode indexes the block table by **absolute** position (`pos // block_size`). vLLM's
sliding-window KV manager only keeps `sliding_window // block_size + 1` physical blocks and
zero-pads block-table entries beyond the window, so an absolute index past the window reads
a zero → physical block 0 → corruption.

### Fix design
Make the TT sliding-window decode index the block table with a **window-relative / rolling**
slot: `phys_slot = (pos % effective_window) // block_size`, matching vLLM's rolling sliding
block-table layout — implemented either (a) in the plugin's per-group page-table
construction (`model_runner._block_tables_per_layer`) so the passed table is already
window-relative, or (b) as a `sliding_window` argument to `paged_update_cache` /
`paged_sdpa_decode` so the op computes the rolling slot. Then restore `SlidingWindowSpec`
in `get_kv_cache_spec` and flip `_HYBRID_KV_CACHE_GROUPS_ENABLED = True`.

### Effort / impact
- **Effort: medium** (TT-op + plugin page-table; needs on-device correctness tests at
  positions > sliding_window across a page boundary).
- **Impact: medium-high** — big KV memory savings for Muse (sliding_window 2048, most of 52
  layers sliding), Gemma3/4, Mistral, GPT-OSS. Enables longer context / more concurrent
  users per card. NOT a Muse serving blocker (workaround is correct), so lower urgency
  than A.

---

## Contribution C — the Muse model adapter itself

Once A (or even just the standard paged path) lands, contribute
`models/demos/muse_glimmer/tt/generator_vllm.py` +
`TTMuseGlimmerForConditionalGeneration` registration (already scaffolded here). Note for
the PR: Muse's OG stateful sliding-window "prefill tail" is an OG-server optimization and is
**bypassed** in vLLM mode — the adapter routes decode through the standard paged SDPA path
with the `sliding_window` read-side trim (same as tt_transformers Gemma/Mistral), so no
model-level tail-state change is required just to serve.

---

## Suggested sequencing
1. **C-minimal**: Muse adapter with standard paged full-attention + host sampling → get a
   working (non-speculative) vLLM Muse server. Proves the model in vLLM; unlocks A's target
   side.
2. **A**: TT draft-model spec decode with Muse's assistant → recover the DFlash decode win
   inside vLLM. The flagship.
3. **B**: sliding-window paged groups → memory efficiency for A's longer-context serving.

## Impact matrix
| # | Contribution | Effort | Upstream value | Recovers OG win |
|---|---|---|---|---|
| A | TT speculative decoding | Large | Very high (all TT models) | DFlash decode speed |
| B | Sliding-window paged KV | Medium | Medium-high (all sliding models) | KV memory efficiency |
| C | Muse vLLM adapter | Medium | Model addition | Serves Muse in vLLM |

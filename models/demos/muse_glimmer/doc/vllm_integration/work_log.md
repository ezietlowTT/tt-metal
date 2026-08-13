# Muse-Glimmer-30B — vLLM integration work log

Goal: serve Muse-Glimmer-30B through the Tenstorrent vLLM fork
(`github.com/tenstorrent/vllm@dev`) + `vllm-tt-plugin`, replacing the model's
bespoke DFlash server with the standard vLLM serving path.

## Status: Phase 1 forward plumbing VALIDATED on device (reduced 2-layer smoke passes)

### Phase 1 progress (C-minimal) — prefill + decode run on the p150a
Re-anchored on the OG server (server.py `_append_prompt_tokens` / `_packed_verify_inputs`)
after drifting into reinvented paths. The adapter now reuses the OG chunked+packed prefill
loop and `packed_decode_forward`; only two vLLM-specific adaptations were needed:
1. **`packed_kv_update` is not page-table-aware** → pass it a PHYSICAL position
   (`page_table[pos//64]*64 + pos%64`); keep `rope_packed`/`cur_pos` logical (rope lookup +
   page-table-aware SDPA read hit the same physical block).
2. **Packed decode treats the 32 physical-verify slots as 32 SDPA "users"** → the page table
   must be repeated to 32 rows (OG: `decode_page_ids.repeat(PHYSICAL_VERIFY_TOKENS, 1)`),
   else SDPA throws `cur_pos must have batch size equal to Q, got 32 and 1`.
Validated via `tests/test_vllm_adapter_smoke.py` (reduced `MUSE_VLLM_N_LAYERS=2`): initialize
-> allocate_kv_cache -> prefill -> decode all run and return `[1, vocab]` logits (gibberish by
design at 2 layers). Debug lessons: clear `__pycache__` between edits (stale bytecode masked
the real path); a hard device fault aborts the process AND dirties the device — `tt-smi -r`
between crashing runs; `faulthandler` + fsync'd progress markers survive the abort.
Remaining: full 52-layer accuracy, on-device sampling (C-full), then drive through the vLLM
server (`run_vllm_server`).

## (earlier) Status: env up + arch registered; Phase 0 done; Phase 1 (C-minimal) scoped at code level

### Phase 0 — DONE (V0 decisions locked)
- **V0 known-good base** = tt-metal @ `1c1b7c9c3` + current `vllm@dev` empty build. OG
  MuseGlimmer repo pins NO tt-metal commit (only transformers), so V0 is de-facto known-good.
- **Reduced target** (fast bring-up): build MuseGlimmerModel with `create_kv_cache=False` and
  override `model_args.num_hidden_layers=2`, `layer_types=("sliding_attention","full_attention")`
  (MuseGlimmerModelArgs is a dataclass; `__post_init__` accepts explicit layer_types).
- **Launch path**: `python -m models.common.readiness_check.run_vllm_server
  --model-dir models/demos/muse_glimmer --hf-model $MUSE_TARGET_DIR ...` (path-based; needs
  `tt/generator_vllm.py` + arch registered — both present). Device hygiene: kill leftover
  `EngineCore`/`vllm.entrypoints`; no profiler in vLLM stages.

### Phase 1 — code plan (C-minimal), exact
- **Decode needs a NEW helper** `vllm_decode_forward` in `tt/attention/decode.py`: Muse's
  existing `packed_decode_forward` is DFlash-specific (requires
  `position_idx/page_index/page_offset/p/real_p` + the custom `packed_kv_update` kernel + tail
  state). The C-minimal single-token path should mirror `packed_decode_forward`'s
  QKV→per-head-norm→rope→SDPA structure but: (1) write the single new K/V with the standard
  `ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos,
  page_table=page_table)` instead of `packed_kv_update`; (2) run
  `ttnn.transformer.paged_scaled_dot_product_attention_decode(q, k_cache, v_cache, page_table,
  cur_pos_tensor=cur_pos, sliding_window_size=config.sliding_window if config.is_sliding else
  None, ...)` (already used at decode.py:124); (3) no `pending_tail` (OG optimization bypassed).
- **Prefill** reuses the existing `prefill_forward` (`tt/attention/prefill.py`, already uses
  `paged_fill_cache` with a page_table) — the adapter passes the vLLM block table as
  `packed["page_table"]`, `chunk_start=start_pos`, `logical_length=S`.
- **Model wiring**: add a model-level decode that routes layers through `vllm_decode_forward`
  (either a `packed["vllm_mode"]=True` branch in `MuseGlimmerModel.__call__`/attention
  `__call__`, or a dedicated `vllm_decode(hidden, cur_pos, page_table)` method).
- **Adapter** (`tt/generator_vllm.py`): `initialize_vllm_model` builds the model
  (create_kv_cache=False) + tokenizer; `get_kv_cache_spec` reuse hybrid base;
  `allocate_kv_cache_per_layer` allocates `[num_blocks,2,64,128]` bf16/TILE/DRAM per layer and
  assigns into `layer.self_attn.kv_cache` + `model.tt_kv_cache`; `prefill_forward` embeds
  (`model.embed_input_ids`) → `model(..., is_decode=False, packed=...)` → host logits;
  `decode_forward` embeds single token → model vllm-decode → host logits (host sampling first).
- **Verify shapes on device** (needs iteration): `paged_update_cache` input K/V shape for a
  single decode token; rope indexing by `cur_pos` (model has `rope_caches_2d[layer_type]`);
  page_table dtype (int32) / shape from vLLM. Mirror tt_transformers/gemma decode for exact shapes.

### Gate 0 — vLLM installs & imports against our stack  ✅ DONE
- vLLM dev pins `torch==2.10.0`; our tt-metal venv has `torch 2.11.0+cpu`.
  Resolved by building the **`empty` target** with `--no-build-isolation --no-deps`
  so it uses our torch 2.11 (empty target compiles no device kernels).
- Build deps needed for `--no-build-isolation`: `setuptools-scm`, `cmake`, `ninja`,
  `jinja2`, `grpcio-tools==1.78.0`.
- `common.txt` runtime deps installed. Side effect: `grpcio-tools==1.78.0`
  downgraded `protobuf` 7.35.1 → 6.33.6 (still satisfies common.txt; ttnn still imports).
- Installed: `vllm 0.1.dev1+g7c99bd3b8.empty`, `vllm-tt-plugin 0.0.0`, `tblib`.
- **Verified all import together**: `ttnn`, `vllm`, `vllm_tt_plugin.platform`,
  `transformers.models.muse_glimmer`, `register_tt_models`. torch stays 2.11.
- Repro: `/home/mando222/museglimmer/vllm`, `env.sh` sourced, the two install scripts
  in scratchpad (`install_vllm.sh` + `install_vllm2.sh`).

### The contract (what a TT vLLM model must be)
TT vLLM models subclass `models.tt_transformers.tt.generator.Generator` and are thin
wrappers (`LlamaForCausalLM`, `QwenForCausalLM`, `MistralForCausalLM` in
`tt_transformers/tt/generator_vllm.py`). Required surface:
- `@classmethod initialize_vllm_model(cls, vllm_config, mesh_device, ...)` — build the model.
- `@classmethod get_kv_cache_spec(cls, vllm_config)` — per-layer KV specs
  (hybrid: sliding-window vs full-attention layers → different specs).
- `prefill_forward(...)` / `decode_forward(...)` — consume vLLM-owned paged KV caches
  + block/page tables, positions, batch dim.
- `read_decode_output(...)` / `process_decode_output_host(...)` — async decode split.
- On-device **split sampling** (`sample_on_device_mode=all`); traced decode replay.
- Registered in `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py::register_tt_models`.

**Key difficulty:** Llama/Qwen/Mistral all share the ONE `tt_transformers.Transformer`
model, so their wrappers are trivial. **Muse is a separate model implementation**
(`models/demos/muse_glimmer/tt/model.py::MuseGlimmerModel` + its own attention and
packed KV cache). It cannot reuse the shared Generator by delegation — the adapter must
call `MuseGlimmerModel` directly and bridge Muse's cache/page-table to vLLM's.

### What makes it tractable
`MuseGlimmerModel.__call__(hidden_states, *, is_decode, packed=..., last_token_only,
compute_logits)` already:
- takes `packed = {page_table, chunk_start, logical_length, real_p}`;
- uses **paged KV** via `ttnn.experimental.paged_fill_cache` (prefill) and
  `paged_update_cache` / the custom `packed_kv_update` kernel (decode);
- holds caches in `self.tt_kv_cache[layer_idx]` and a `self.page_table` it owns.

So the bridge is: **externalize** `tt_kv_cache` and `page_table` so vLLM owns/allocates
them and the adapter injects them per step — not a from-scratch attention rewrite.

## Implementation plan (remaining)

1. **generator_vllm.py adapter** (`models/demos/muse_glimmer/tt/generator_vllm.py`):
   - `MuseGlimmerForConditionalGeneration(Generator)` (text-only path; reject mm inputs).
   - `initialize_vllm_model`: build `MuseGlimmerModel` via `create_tt_model`, but with KV
     cache allocation deferred to vLLM (`allocate_vllm_kv_cache` producing Muse's
     per-layer `(k,v)` paged tensors in the block layout vLLM expects).
   - `get_kv_cache_spec`: emit per-layer specs — Muse mixes sliding-window and full
     attention (see `model_config` / layer config), so use vLLM hybrid attention specs.
   - `prefill_forward`: run embeddings + `MuseGlimmerModel(..., is_decode=False,
     packed={page_table=<vLLM block table>, chunk_start, logical_length})`, chunk
     internally to `PREFILL_CHUNK_SIZE` (2048) but pass the *logical* length through
     masks/positions/output slicing — do NOT require prompt length divisible by chunk/page.
   - `decode_forward`: `is_decode=True` single-token step over vLLM page table + positions;
     drive the traced decode path (see `$tt-enable-tracing`).
   - **Drop DFlash** — vLLM owns scheduling/KV; the assistant drafter is not used
     (native DFlash speculative decode is incompatible with vLLM's per-step KV mgmt).
     Optionally revisit via vLLM's own speculative-decode subsystem later.
   - On-device split sampling: Muse currently samples greedily inside decode; expose
     `tt_out_tok` split (`decode_forward(..., read_from_device=False)` +
     `read_decode_output(async_read=True)` + `process_decode_output_host`). If Muse's
     model lacks split sampling, add it in `tt/model.py` (small, contract-driven change).

2. **Batch / concurrency**: Muse's decode + `packed_kv_update`/`decode.py` assume
   **batch-1**. vLLM prefers `max_num_seqs` up to 32. First bring-up target is
   `--max-num-seqs 1` (Muse's honest hard constraint); batching would need batch-dim
   support in the decode kernels — record as a follow-up / hard-limit with evidence.

3. **Registration**: HF arch is `MuseGlimmerForConditionalGeneration`. Add
   `_register_model_if_missing(ModelRegistry, "TTMuseGlimmerForConditionalGeneration",
   "models.demos.muse_glimmer.tt.generator_vllm:MuseGlimmerForConditionalGeneration")`.
   Verify the platform's HF→TT arch remap actually resolves this name at server startup
   (Muse is multimodal-typed; confirm the text path is selected and mm inputs rejected).

4. **Bring-up (minimum-surface first)**: reduced 1-layer-per-kind config → server launch,
   trace capture/replay, vLLM cache ownership, page-table refresh, on-device sampling,
   stale-input tests. Then full 52-layer model for accuracy + benchmark evidence via
   `python -m models.common.readiness_check.run_vllm_server`.

## Honest effort estimate
This is a multi-session build. The hard parts are (a) matching Muse's paged cache tensor
layout to vLLM's block layout in `allocate_vllm_kv_cache` + `get_kv_cache_spec`, (b) the
traced decode + on-device split-sampling contract, and (c) the batch-1 → batched gap.
Correctness (PCC) is already proven for the standalone model, so this is a serving-path
adapter effort, not a model-correctness effort.

---

## Reverse-engineered contract (from the plugin source — use this to implement)

### What the plugin calls, in order
- `loader.py:38` → `MuseGlimmerForConditionalGeneration.initialize_vllm_model(hf_config,
  mesh_device, max_batch_size, max_seq_len, n_layers=None, tt_data_parallel=1,
  optimizations="performance")` → must return `cls([model], [model_args], mesh_device, tokenizer=...)`.
  (`self.model` and `self.model_args` are **lists**, one per DP submesh; `data_parallel = len(self.model)`.)
- `worker.py` → classmethod `get_kv_cache_spec(vllm_config)` → REUSE
  `HybridAttentionForCausalLM.get_kv_cache_spec` unchanged: Muse's HF `text_config.layer_types`
  exists (52 entries, sliding_attention/full_attention, sliding_window=2048). It currently
  emits `FullAttentionSpec(block_size, num_kv_heads=2, head_size=128, dtype)` for **every**
  layer (sliding disabled on purpose — SDPA trims on the read side), i.e. one KV group,
  single `page_table` path (`block_tables_per_layer` stays None).
- `model_runner.py:446` → `allocate_kv_cache_per_layer(per_layer_specs)` where each spec is
  `(kv_cache_shape, dtype, tensor_idx)`. OVERRIDE to allocate Muse's layout
  `[num_blocks, num_kv_heads=2, block_size=64, head_dim=128]` (bf16, TILE, DRAM — see
  `tt/attention/kv_cache.py::init_kv_cache`) and **assign into `layer.self_attn.kv_cache`**
  and `model.tt_kv_cache`. Return `list[submesh][layer_idx][k,v]`.

### The prefill/decode kwargs the runner passes (`model_runner.py:2562`)
```
prefill_forward(tokens=<torch [B,S] ids>, page_table=<vLLM block table>, kv_cache=self.kv_caches,
                enable_trace=<bool>, prompt_lens=<torch [B]>, start_pos=<torch positions>,
                [sampling_params=TTSamplingParams], [empty_slots=list], [page_tables_per_layer=None])
decode_forward(tokens=..., page_table=..., kv_cache=..., enable_trace=..., start_pos=...,
               prompt_lens=None, [sampling_params=...])   # prompt_lens None ⇒ decode
```
- If `self.request_specific_rope` is False (Muse: no mrope), `prefill_forward` returns just
  `tt_out`. Decode returns `tt_out`; if not host tensors, runner calls
  `read_decode_output(tt_out, async_read=)` then `process_decode_output_host(tt_out, is_tokens=)`.

### Recommended approach: OVERRIDE, don't delegate
`Generator.decode_forward`/`prefill_forward_text` call `self.model[i].switch_mode`,
`.prepare_inputs_decode`, `.ttnn_decode_forward`, `.sampling`, `.sampling_dp`,
`model_args[0].get_warmup_prefill_supported_seq_lens()` — all **tt_transformers.Transformer**
internals Muse lacks. So subclass `HybridAttentionForCausalLM` and fully OVERRIDE:
`initialize_vllm_model`, `allocate_kv_cache_per_layer`, `prefill_forward`, `decode_forward`,
`warmup_model_prefill`/`warmup` (stub to no-op), and provide `model_capabilities` +
`cache_path`. Return **host logits** first (host sampling) to get a server up; add on-device
split sampling later. This bypasses the tt_transformers coupling entirely.

- `prefill_forward`: `model.tt_kv_cache = kv_cache[0]`; embed `tokens` (`model._embedding_rows`/
  embed → hidden), call `model(hidden, is_decode=False,
  packed={page_table:<ttnn int32 from vLLM table>, chunk_start:int(start_pos), logical_length:S},
  last_token_only=True)` → logits → host. Chunk internally to `PREFILL_CHUNK_SIZE=2048`.

## BLOCKERS discovered (why this can't be a quick finish)

1. **Decode statefulness — DOWNGRADED after deeper reading (not a hard blocker).**
   Muse's decode ALREADY uses the standard paged op:
   `tt/attention/decode.py:124` calls
   `ttnn.transformer.paged_scaled_dot_product_attention_decode(..., page_table,
   cur_pos_tensor=cur_pos, sliding_window_size=config.sliding_window if config.is_sliding
   else None)` — the same paged-SDPA-decode + read-side sliding trim that
   tt_transformers/vLLM Gemma/Mistral use. The stateful `prefill_tail`
   (`reset_prefill_state`/`clone_prefill_state`/`commit_decode_tails`, `retain_tail`) is an
   **OG-server optimization layered on top**, not a replacement. In vLLM mode the adapter
   drives the standard paged path (page_table + cur_pos from vLLM, sliding trim on read) and
   simply does NOT retain the tail — no model-level change required. So a non-speculative
   Muse vLLM server (C-minimal) is tractable; the real remaining effort is the adapter glue
   + on-device iteration, not an attention rewrite.

2. **Generator base ↔ tt_transformers coupling.** As above; handled by the override approach,
   but it means Muse re-implements input prep, the forward call, sampling, and output
   formatting itself instead of reusing the base — more surface, more device-debug cycles.

3. **Batch-1 kernels.** `packed_kv_update` / `tt/attention/decode.py` assume batch-1. First
   serving target must be `--max-num-seqs 1`; batched continuous batching needs batch-dim
   support in those kernels (follow-up, record as hard limit).

## Status at end of session
- ✅ Native DFlash server still works post-vLLM-install (48-tok gen, ~29 AR tok/s, DFlash burst 14).
- ✅ vLLM `empty` + plugin installed & importing with ttnn/torch 2.11; arch
  `TTMuseGlimmerForConditionalGeneration` registered; scaffold imports & subclasses Generator.
- ⬜ Adapter bodies (`initialize_vllm_model`/`allocate_kv_cache_per_layer`/`prefill_forward`/
  `decode_forward`) not implemented — `generator_vllm.py` is a scaffold raising NotImplementedError.
- ⬜ Blocker #1 (stateful sliding-window decode) must be resolved for a working serve.
Next session: implement prefill path + host sampling first (server launch + load + cache-alloc +
first-token prefill is reachable), then tackle decode blocker #1.

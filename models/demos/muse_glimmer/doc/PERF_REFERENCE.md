# Muse-Glimmer-30B on p150a — performance reference

Consolidated perf record across serving paths, for future comparison. Single p150a
(Blackhole), batch-1, greedy, BFP8 attn/LM-head + BFP4 MLP, 64-token paged KV cache.
See `BENCHMARKS.md` for the full native-path methodology and `tt-metal/models/demos/
muse_glimmer/doc/vllm_integration/` for the vLLM work.

## 1. Native DFlash server (the OG repo path) — authoritative baseline

Measured 2026-08-12/14. Native `models.demos.muse_glimmer.server.server`, DFlash 16-token
speculative decode, on-device sampling.

**Prefill throughput** (scales with prompt length; chunked prefill amortizes fixed cost):

| prompt tokens | 266 | 1546 | 3026 | 5986 | 11906 | 23706 |
|---|---|---|---|---|---|---|
| prefill tok/s | 27 | 136 | 241 | 394 | 744 | **1411** |

Isolated-phase prefill (no tokenization/DFlash overhead): ~996 tok/s @2K, ~1396 tok/s @8K.
Trends toward the repo's ~1549 tok/s claim at 128K.

**Decode (AR, DFlash speculative)** — acceptance-bound, content-dependent, **22–59 tok/s**:

| workload | avg accepted/verify | AR decode tok/s |
|---|---|---|
| math / code (structured) | 4.0–4.6 | 43–45 |
| "ocean" 1-sentence | ~3.75 (bursts 8–11) | ~40–59 |
| summarize free text | 2.4–3.9 | 27–37 |
| hard free-form | 1.5–1.9 | 22–24 |

Repo headline best case: 120 tok/s. **Decode t/s varies ~3× with content** — measure on
representative prompts.

**KV cache**: single-session exact prefix reuse → identical repeat request skips ALL prefill
(cached=full, prefill=0s); multi-turn continuation re-prefills only the new suffix. Decode
latency grows only ~34% (1.38s→1.85s per 16-tok block) from 64→8192 context.

**Stable load**: max_seq_len 32768 loads in ~30s; host RSS ~1 GB (weights on-device via
lazy streaming). 128K context available (opt-in).

**Correctness (PCC vs HF reference)**: bf16 ~0.9999, BFP8 0.991–0.999, BFP4-MLP 0.90–0.99
(all pass). Full parity suite 22/22.

**Regression check (2026-08-14, after vLLM integration work)**: native model code is
byte-identical to original (empty git diff); "ocean" gen reproduced verbatim @ **40 AR
tok/s** — no perf/accuracy loss from the vLLM-adapter work.

## 2. vLLM C-minimal serving path (new, 2026-08-14)

First end-to-end generation through the real vLLM v1 engine → TT plugin → adapter → HOST
sampler. Full 52-layer model, greedy, `max_num_seqs=1`, `enforce_eager=True`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `gpu_memory_utilization=0.55`,
`additional_config={"trace_region_size": 256000000}`. vLLM allocated 2049 KV blocks.

**Functional**: ✅ "The capital of France is" → " Paris and it is one of the most visited
cities in the world. Due to" (coherent, correct).

**Perf (C-minimal, UNOPTIMIZED)**: not yet benchmarked with a clean number — the fork-stack
timing run hung (device thrashing during the session), and we're changing over to the
standalone `vllm-tt-plugin` + upstream vLLM 0.24.0 (see below), so vLLM perf will be measured
on that target stack rather than the soon-retired fork. Functional path is confirmed
(coherent multi-token generation). Expected to be well below native decode until C-full
(on-device sampling + traced decode) and A (DFlash-as-spec-decode) land — host sampling +
eager + no trace + no speculation is a lower bound, not a serving verdict.

**Stack change (2026-08-14)**: the TT plugin moved to the standalone
`tenstorrent/vllm-tt-plugin` (upstream vLLM 0.24.0 built `empty` + the plugin), replacing the
monolithic `tenstorrent/vllm` fork. Our tt-metal adapter carries over; re-verification +
perf on the new stack are in progress.

> ⚠️ These are a functional-path lower bound, NOT a serving-perf verdict. C-minimal uses
> host sampling + eager execution + no traced decode + no DFlash speculation. The optimized
> path (Phase 2 C-full: on-device sampling + traced decode; Phase 3/4: DFlash-as-spec-decode)
> should approach the native decode numbers. Compare like-for-like only after C-full.

## How to reproduce
- Native: `source env.sh; $PY -m models.demos.muse_glimmer.server.server --model-path
  "$MUSE_TARGET_DIR" ... --prompt "..." --max-new-tokens N`
- vLLM offline: `VLLM_ENABLE_V1_MULTIPROCESSING=0 $PY scratchpad/vllm_perf.py` (from tt-metal
  root, env sourced).

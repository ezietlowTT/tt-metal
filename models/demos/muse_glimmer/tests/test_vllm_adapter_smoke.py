# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimum-surface device smoke for the C-minimal vLLM adapter (Phase 1).

Drives initialize_vllm_model -> allocate_kv_cache_per_layer -> prefill_forward ->
decode_forward directly on a reduced model (set MUSE_VLLM_N_LAYERS=2), bypassing the
full vLLM server/runner. Not a correctness gate (reduced layers); it validates the
adapter's forward plumbing on device.

Run:
  MUSE_VLLM_N_LAYERS=2 pytest -q -s \
    models/demos/muse_glimmer/tests/test_vllm_adapter_smoke.py
"""

import faulthandler
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import ttnn

# Dump a Python traceback to a file on any fatal signal (SIGABRT/SIGSEGV) so a hard
# device fault still tells us which op/line crashed.
faulthandler.enable(open("/tmp/fault.txt", "w"), all_threads=True)

from models.demos.muse_glimmer.tt.common import DEFAULT_MODEL_PATH
from models.demos.muse_glimmer.tt.generator_vllm import MuseGlimmerForConditionalGeneration

MODEL_PATH = os.getenv("MUSE_TARGET_DIR", DEFAULT_MODEL_PATH)
MAX_LEN = 512
BLOCK = 64

_PROG = "/tmp/smoke_progress.txt"


def mark(msg):
    # crash-survivable progress marker (unbuffered append), so a device segfault
    # still leaves the last stage on disk.
    with open(_PROG, "a") as f:
        f.write(msg + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[smoke] {msg}", flush=True)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.DISABLED}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_vllm_adapter_prefill_decode(mesh_device):
    if not Path(MODEL_PATH, "config.json").exists():
        pytest.skip(f"Muse Glimmer checkpoint not available at {MODEL_PATH}")

    open(_PROG, "w").close()
    mark("start; building adapter")
    hf_config = SimpleNamespace(_name_or_path=MODEL_PATH)
    gen = MuseGlimmerForConditionalGeneration.initialize_vllm_model(
        hf_config, mesh_device, max_batch_size=1, max_seq_len=MAX_LEN
    )
    mark("initialize_vllm_model done")

    n_layers = len(gen.model[0].layers)
    num_blocks = MAX_LEN // BLOCK
    per_layer_specs = [([num_blocks, 2, BLOCK, 128], torch.bfloat16, i) for i in range(n_layers)]
    gen.allocate_kv_cache_per_layer(per_layer_specs)
    mark(f"allocate_kv_cache_per_layer done (n_layers={n_layers}, num_blocks={num_blocks})")

    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prompt = "The capital of France is"
    ids = gen.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    S = int(ids.shape[0])
    mark(f"tokenized prompt_len={S}; calling prefill_forward")

    logits = gen.prefill_forward(ids, page_table=page_table, start_pos=0, prompt_lens=torch.tensor([S]))
    mark(f"prefill_forward returned shape={tuple(logits.shape)}")
    next_tok = int(logits.reshape(-1, logits.shape[-1])[-1].argmax().item())
    mark(f"prefill argmax token id={next_tok} -> {gen.tokenizer.decode([next_tok])!r}")
    assert logits.shape[-1] == gen.model[0].vocab_size

    if os.getenv("MUSE_SMOKE_PREFILL_ONLY") == "1":
        mark("PREFILL_ONLY: PASS")
        return

    mark("calling decode_forward")
    dec = gen.decode_forward(torch.tensor([next_tok]), start_pos=torch.tensor([S]), page_table=page_table)
    mark(f"decode_forward returned shape={tuple(dec.shape)}")
    dtok = int(dec.reshape(-1, dec.shape[-1])[-1].argmax().item())
    mark(f"decode argmax token id={dtok} -> {gen.tokenizer.decode([dtok])!r}")
    assert dec.shape[-1] == gen.model[0].vocab_size
    mark("PASS: prefill+decode ran on device and produced vocab-sized logits")

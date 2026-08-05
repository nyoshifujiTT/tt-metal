# Qwen3-Embedding standalone demo (single-device, non-vLLM)

`demo.py` runs the Qwen3-Embedding model directly on a single TT device, without
vLLM or tt-inference-server. It drives the exact TT forward path the vLLM adapter
uses and compares the resulting embedding against a Hugging Face reference by
cosine similarity, so it is the standard way to smoke-test the metal model on its
own (mirroring `models/demos/wormhole/bge_m3/demo/demo.py`).

The same demo covers the whole Qwen3-Embedding family. The checkpoint is fixed by
the test parametrization (`DEFAULT_MODEL_NAME`, default `Qwen/Qwen3-Embedding-0.6B`);
`Qwen/Qwen3-Embedding-8B` works the same way once selected (see "Run" below). It is
single-device only (WH or BH, e.g. a single P150).

## What it verifies

Qwen3-Embedding is a decoder model whose sentence embedding is the *last* token's
hidden state (after the final norm, before the LM head), L2-normalized. The demo
produces that embedding three equivalent ways and asserts each is unit-norm and
matches the HF reference (mean `cos > 0.95`, typically ~0.97):

- `test_qwen3_embedding_demo` — pooled last-token forward + host L2 normalize.
  This is the path the serving runner consumes.
- `test_qwen3_embedding_flat_contract` — flat per-token forward
  (`return_full_hidden_states=True`) + host LAST pooling + L2 normalize. Asserts
  it equals the pooled path bit-for-bit (`cos > 0.9999`), guarding the layout the
  vLLM pooling runner relies on.
- `test_qwen3_embedding_single_trace` — `embed_single_trace=True` folds last-token
  slice + final norm + L2 normalize into one prefill trace, returning the finished
  normalized embedding from a single `execute_trace` replay. Asserts it equals the
  pooled path (`cos > 0.999`) and is unit-norm on device.

## Prerequisites

- A built tt-metal Python environment on a host with one TT device visible
  (`ttnn` importable; see the top-level tt-metal build instructions).
- Python deps used by the demo: `transformers`, `torch`, `scikit-learn`, `numpy`,
  `loguru` (all part of the tt-metal dev requirements).
- Model weights. The checkpoint is resolved through standard Hugging Face rules,
  so either allow an online fetch (`HF_TOKEN` set if required) or pre-stage the
  repo in the local HF cache / a directory. No repo id is hardcoded in the model;
  the demo hands the parametrized model id to `AutoConfig`/`AutoModel`/`AutoTokenizer`.

## Run

From the tt-metal repo root:

```bash
# 0.6B (default) - pooled last-token accuracy check
pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py::test_qwen3_embedding_demo

# all three checks (pooled, flat-contract, single-trace)
pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py
```

`device` and `model_location_generator` are supplied by the shared tt-metal
pytest fixtures (root `conftest.py`); you do not pass them yourself.

The checkpoint (`model_name`) and `sequence_length` come from the
`@pytest.mark.parametrize` on each `test_*` function, defaulting to
`Qwen/Qwen3-Embedding-0.6B` at `8192`. To exercise the 8B checkpoint, change
`DEFAULT_MODEL_NAME` to `Qwen/Qwen3-Embedding-8B` (or add an `8B` entry to the
parametrize) and rerun the same command.

## Expected output

The demo logs, per prompt, the embedding dimension (`1024` for 0.6B, `4096` for
8B), the TT L2 norms (all `~1.0`), and `cos(TT, HF)` per prompt plus the mean. A
passing run ends with the selected `test_*` cases green; the accuracy assertion is
mean `cos(TT, HF) > 0.95` (observed ~0.97 on P150).

## Known constraints

- Single device only. The demo raises if more than one device is presented.
- One prompt per call (`batch_size=1`). The embedding is the last-token hidden
  state, and the TT forward derives that index from a single `prompt_lens` value
  shared across the batch, so mixing variable-length prompts in one batched call
  would pick the wrong token for the shorter rows. Serving feeds one request at a
  time for the same reason.
- Max sequence length defaults to `8192` (`DEFAULT_SEQUENCE_LENGTH`). This is the
  device-memory-capped context used for the P150 bring-up, not the model's full
  positional limit; longer inputs are truncated by the tokenizer.

## Reference

- `models/demos/wormhole/qwen3_embedding_8b/demo/demo.py` — this demo.
- `models/demos/qwen3_embedding/tt/generator_vllm.py` — the vLLM pooling adapter
  that wraps the same forward path for serving.

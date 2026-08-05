# Qwen3-Embedding standalone demo (single-device, non-vLLM)

`demo.py` runs the Qwen3-Embedding model directly on a single TT device, without
vLLM or tt-inference-server. It drives the exact TT forward path the vLLM adapter
uses and compares the result against a Hugging Face reference by cosine
similarity, so it is the standard way to smoke-test the metal model on its own
(mirroring `models/demos/wormhole/bge_m3/demo/demo.py`).

The same demo covers the whole Qwen3-Embedding family. The accuracy test is
parametrized over `Qwen/Qwen3-Embedding-0.6B` (default) and `Qwen/Qwen3-Embedding-8B`
(the `EMBEDDING_MODELS` matrix); every checkpoint shares this metal
implementation, is single-device (WH or BH, e.g. a single P150), and caps the
context at 8192 for the P150 bring-up.

## Low-level model creation

`Qwen3ForEmbedding` (from `generator_vllm.py`) is the metal embedding model. For a
single device, construct it directly with an explicit checkpoint name; the config
is resolved through standard Hugging Face rules (local cache honoured, no
hardcoded repo id):

```python
import ttnn
from models.demos.wormhole.qwen3_embedding_8b.demo.generator_vllm import Qwen3ForEmbedding

device = ttnn.open_device(device_id=0)

model = Qwen3ForEmbedding(
    device=device,
    max_batch_size=1,          # >1 enables the batched path (see below)
    max_seq_len=8192,          # device-memory-capped context for P150
    model_name="Qwen/Qwen3-Embedding-0.6B",
    # act_dtype / weight_dtype default to bfloat16 / bfloat8_b
)
```

The model lazily initializes its KV cache / paged-attention config on the first
`forward`, so no separate setup call is needed for the demo path.

## Embedding API

`forward(input_ids, attention_mask, ...)` is prefill-only (embedding models have
no decode step). It returns the pooled last-token hidden state `[batch, hidden]`
(after the final norm, before the LM head). Qwen3-Embedding's only pooling
directive on top is L2 normalization, which the official Hugging Face usage
applies as `F.normalize(embeddings, p=2, dim=1)`.

`forward` exposes three equivalent ways to obtain that embedding:

- default (`[batch, hidden]` pooled last-token) — normalize on the host. This is
  what the serving pooling runner consumes.
- `return_full_hidden_states=True` — return the flat per-token
  `[total_tokens, hidden]` layout (final norm, no last-token slice) that an
  upstream-conforming pooling runner would hand to a `Pooler`; do LAST pooling +
  normalize on the host.
- `embed_single_trace=True` — fold last-token slice + final norm + L2 normalize
  into the prefill trace, so a single `execute_trace` replay returns the finished,
  already-normalized embedding (no host post-processing).

Accessors: `get_embedding_dim()` (1024 for 0.6B, 4096 for 8B),
`get_max_seq_len()`, and `get_max_batch_size()` report the configured sizes the
serving stack uses to size buffers.

Not covered here: `initialize_vllm_model()` is a vLLM-only classmethod (it builds
the model from a `vllm_config` device); the standalone demo constructs
`Qwen3ForEmbedding` directly instead, so that path is exercised by the vLLM/plugin
tests rather than this demo.

## Run inference (Example)

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", padding_side="left")

def embed(prompt):
    batch = tokenizer([prompt], padding=False, truncation=True, max_length=8192, return_tensors="pt")
    hidden = model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).to(torch.float32)
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)
    return F.normalize(hidden, p=2, dim=1)   # finished embedding

emb = embed("Artificial intelligence is transforming how we interact with technology.")
```

## Batched inference

Set `max_batch_size > 1` and pass a multi-row `input_ids` / `attention_mask` to
embed several prompts in one `forward` (returns `[batch, hidden]`). The metal
forward derives the last-token index from a single `prompt_lens` value shared
across the batch, so **every row in a batch must share one real length** for its
last token to be correct — batch same-length prompts (right-padded to a shared
length), exactly how the serving pooling runner batches same-length requests.
Slice the result back to your real batch size and L2-normalize.

## Tests

Run from the tt-metal repo root. `device` and `model_location_generator` come
from the shared root `conftest.py` fixtures; you do not pass them yourself.

```bash
# pooled last-token accuracy check, 0.6B (default)
pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py::test_qwen3_embedding_demo -k 0.6B

# the same check on the 8B checkpoint
pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py::test_qwen3_embedding_demo -k 8B

# everything (all paths, 0.6B unless noted)
pytest models/demos/wormhole/qwen3_embedding_8b/demo/demo.py
```

The demo's tests, and what each proves:

- `test_qwen3_embedding_demo[0.6B|8B]` — pooled last-token forward + host L2
  normalize; unit-norm and `cos(TT, HF) > 0.95` (observed ~0.97 on P150).
- `test_qwen3_embedding_flat_contract` — flat per-token forward equals the pooled
  path bit-for-bit (`cos > 0.9999`) and matches HF, guarding the layout the vLLM
  pooling runner relies on.
- `test_qwen3_embedding_single_trace` — the on-device single-trace embedding
  equals the pooled path (`cos > 0.999`) and is unit-norm on device.
- `test_qwen3_embedding_batched` — a real `batch_size=4` forward; every row is
  unit-norm, matches its `batch_size=1` embedding (`cos > 0.99`; batched prefill
  is not bit-exact vs B=1 under bf8, ~0.996) and matches HF (`cos > 0.95`). Uses
  one prompt repeated so every row shares the batch-wide `prompt_lens` length.
- `test_qwen3_embedding_long_context` — a ~4500-token prompt exercises the long
  single-chunk prefill (>4096); unit-norm and `cos(TT, HF) > 0.95`.
- `test_qwen3_embedding_accessors` — `get_embedding_dim` / `get_max_seq_len` /
  `get_max_batch_size` agree with the config/constructor and the real embedding
  width.

## Expected output

Each test logs the embedding dimension (`1024` for 0.6B, `4096` for 8B), TT L2
norms (all `~1.0`), and the relevant cosine similarities. A passing run ends with
the selected `test_*` cases green.

## Known constraints

- Single device only. The demo raises if more than one device is presented.
- Batched calls require equal-length prompts (see "Batched inference"); the
  default per-prompt tests run one prompt per call for the same reason.
- Max sequence length defaults to `8192` (`DEFAULT_SEQUENCE_LENGTH`). This is the
  device-memory-capped context used for the P150 bring-up, not the model's full
  positional limit; longer inputs are truncated by the tokenizer.

## Reference

- `models/demos/wormhole/qwen3_embedding_8b/demo/demo.py` — this demo.
- `models/demos/wormhole/qwen3_embedding_8b/demo/generator_vllm.py` — the
  `Qwen3ForEmbedding` metal model.
- `models/demos/qwen3_embedding/tt/generator_vllm.py` — the vLLM pooling adapter
  that wraps the same forward path for serving.

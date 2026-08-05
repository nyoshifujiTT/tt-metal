# BGE-Reranker-v2-M3

## Introduction
This directory contains the Tenstorrent implementation of
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3), a
multilingual cross-encoder reranker. Given a `(query, document)` pair it returns
a single relevance logit; higher means more relevant. The model reuses the
shared XLM-RoBERTa encoder backbone from `models/demos/wormhole/bge_m3`, adds an
on-device sequence-classification head, and scores end-to-end on device (the CLS
extraction and the classification head both run on the Tenstorrent device, so
only the small `[batch, 1]` logit returns to host).

Ranking a list of documents (sorting by score, `top_n` selection) is a
serving-layer concern handled by the vLLM `/rerank` endpoint, not by the device
model; the device model produces per-pair relevance logits only.

## Platforms
- Blackhole (p150)

## Structure
- `demo/demo.py` - Standalone single-device demo (scoring, reranking, long
  sequences); no vLLM required.
- `demo/generator_vllm.py` - `BgeRerankerV2M3`, the cross-encoder execution
  wrapper and vLLM model adapter (`forward` scores a request to `[batch, 1]`).
- `tt/xlm_roberta_classification_head_tt.py` - On-device (ttnn) classification
  head (`dense -> tanh -> out_proj`, fp32).
- `tt/xlm_roberta_classification_head.py` - Host (PyTorch, fp32) reference head.
- `tt/reranker_pooler.py` - Device CLS extraction / scoring helpers and the
  `RerankerClassifierPooler` used by the canonical vLLM pooling path.
- `tt/model_config.py` - Reranker sequence-classification weight loading.
- `tests/` - Device and device-free unit tests.

## Model Specifications
- Model: BAAI/bge-reranker-v2-m3 (XLM-RoBERTa large, `XLMRobertaForSequenceClassification`)
- Hidden Size: 1024
- Layers: 24
- Attention Heads: 16
- Intermediate Size: 4096
- Max Sequence Length: 8192
- Output: single relevance logit per `(query, document)` pair

## Prerequisites
- A single Tenstorrent device.
- The `BAAI/bge-reranker-v2-m3` weights (downloaded from Hugging Face, or provided
  via the test `model_location_generator`).

## How to Run

### Demo (single device)
Run the standalone demo, which scores positive/negative pairs against the Hugging
Face reference, ranks a query against several documents, and exercises the
8192-token long-sequence path:

```bash
pytest models/demos/bge_reranker_v2_m3/demo/demo.py
```

The weights are resolved from the Hugging Face Hub by default. To run offline
against a local checkpoint, point `HF_MODEL` at the snapshot directory (the same
convention other demos use):

```bash
export HF_MODEL=/path/to/bge-reranker-v2-m3
pytest models/demos/bge_reranker_v2_m3/demo/demo.py
```

Individual scenarios:

```bash
# Positive/negative scoring, checked against the HF reference logit
pytest models/demos/bge_reranker_v2_m3/demo/demo.py::test_reranker_score

# One query vs several documents, ranked on host
pytest models/demos/bge_reranker_v2_m3/demo/demo.py::test_reranker_rerank

# 8192-token request through the chunked long-sequence encoder path
pytest models/demos/bge_reranker_v2_m3/demo/demo.py::test_reranker_long_seq
```

### Tests
```bash
# End-to-end device logit vs Hugging Face reference
pytest models/demos/bge_reranker_v2_m3/tests/test_model.py

# Device classification head and pooler
pytest models/demos/bge_reranker_v2_m3/tests/test_xlm_roberta_classification_head_tt.py
pytest models/demos/bge_reranker_v2_m3/tests/test_reranker_pooler.py
```

## Scoring API
Score a `(query, document)` pair on device with a single `forward` call. This is
exactly the path the deployed pooling runner uses -- the encoder, on-device CLS
extraction, and the device head run in one forward and return the `[batch, 1]`
relevance logit:

```python
import ttnn
from models.demos.bge_reranker_v2_m3.demo.generator_vllm import BgeRerankerV2M3

device = ttnn.open_device(device_id=0)

reranker = BgeRerankerV2M3(
    device=device,
    max_batch_size=1,
    max_seq_len=8192,
    model_name="BAAI/bge-reranker-v2-m3",
)
reranker._initialize_model()  # builds the encoder + device head (forward also does this lazily)

query = "what is tenstorrent"
document = "Tenstorrent builds AI processors."
enc = reranker.tokenizer(
    [[query, document]], padding=True, truncation=True, return_tensors="pt", max_length=512
)
logit = reranker.forward(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
print(float(logit.view(-1)[0]))  # relevance logit (higher = more relevant)

ttnn.close_device(device)
```

To rank several documents, score each `(query, document)` pair and sort by logit
on host (as the demo's `run_reranker_rerank` does); the device model itself does
not sort.

## Relationship to bge-m3
The reranker reuses the bge-m3 XLM-RoBERTa encoder backbone (`create_tt_model`
and the shared `XlmRobertaEncoder` execution template) unchanged, and adds the
sequence-classification head on top. See
`models/demos/wormhole/bge_m3/USER_GUIDE.md` for the encoder/embedding backbone.

# pyannote speaker-diarization-community-1

## Platforms:
    Blackhole (p150)

## Introduction

[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) is a speaker-diarization pipeline: it segments an audio recording into "who spoke when". The pipeline runs two neural nets — a **WeSpeaker ResNet34** speaker-embedding model and a **PyanNet** (SincNet + BiLSTM) local-segmentation model — around host-side clustering. This demo ports both neural nets to ttnn so they execute on a Tenstorrent Blackhole p150, matching the torch reference within tolerance. The tests assert the bounds: WeSpeaker embedding cosine > 0.99, PyanNet segmentation logit cosine > 0.99 with powerset argmax agreement > 0.95, and an end-to-end diarization error rate < 0.05 against the same pipeline run entirely on host. On the 30 s sample the measured frame agreement is 0.998.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `pip install "pyannote.audio==4.*" soundfile` (community-1 requires pyannote 4.x)

## Weights

Weights are fetched on demand, so there is nothing to place by hand. Model identity and model location are separate, as elsewhere in tt-metal: the repo id comes from `HF_MODEL` (default `pyannote/speaker-diarization-community-1`) and the location is resolved by the `model_location_generator` fixture, falling back to a Hugging Face download.

`pyannote/speaker-diarization-community-1` is gated, so accept the terms on the model page and export a token:

```sh
export HF_TOKEN=hf_...
```

Without a token, point `HF_MODEL` at the ungated mirror. It carries the same checkpoints: the Hub tree API reports identical git object ids and sizes for `config.yaml`, `embedding/pytorch_model.bin`, `segmentation/pytorch_model.bin` and both `plda/*.npz` files.

```sh
export HF_MODEL=pyannote-community/speaker-diarization-community-1
```

## How to Run

### Demo

Diarize pyannote's bundled 30 s sample with the WeSpeaker embedding on device:

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/demo/demo.py::test_demo
```

Diarize your own recording with both neural nets on device:

```sh
pytest --disable-warnings --input-path=/path/to/audio.wav models/demos/audio/pyannote_diarization/demo/demo.py::test_demo_both_nets
```

### Device-independent reference parity (no p150 needed)

Validates the numpy op-graph against the real torch models, plus the weight-resolution helper:

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_common_weight_resolution.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_wespeaker_numpy_ref_parity.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_pyannet_numpy_ref_parity.py
```

### On-device parity (p150)

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_ttnn_wespeaker_ondevice.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_ttnn_pyannet_ondevice.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_resident_narrow_ondevice.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_diarization_e2e_ondevice.py
```

Pass `--device-id` to pick a device; every test takes it from the shared `device` fixture.

## Details

### Structure

- `common.py` — weight resolution shared by the tests and the demo (`HF_MODEL` for identity, `model_location_generator` for location).
- `tt/` — ttnn implementations:
  - `ttnn_wespeaker.py` — `TTNNWeSpeaker`, the ResNet34 speaker embedding.
  - `ttnn_wespeaker_resident.py` — device-resident fast path: the activation stays on device across the whole ResNet34 (input uploaded once, every conv/relu/residual runs on device, only the final feature map is downloaded); TSTP pooling and the `seg_1` linear stay on host so pyannote's exact (optionally weighted) pooling is preserved bit-for-bit.
  - `ttnn_pyannet.py` — the PyanNet SincNet + BiLSTM local-segmentation net.
- `reference/` — numpy reference op-graphs (`wespeaker_numpy_ref.py`, `pyannet_numpy_ref.py`), each parity-checked against the real torch model.
- `tests/` — the parity tests listed above.
- `demo/demo.py` — the runnable demo.

### What runs on device

Both neural nets are executed with ttnn: WeSpeaker convolutions, residual-adds, relu and reductions; PyanNet SincNet convolutions and the BiLSTM (implemented as a device-resident batched recurrence, since ttnn has no fused LSTM). Host keeps only the pyannote clustering/pooling glue, matching the CPU pipeline numerically.

### Test inputs

No fixture data is checked in. The parity tests build their inputs from a fixed seed and compute the torch reference in process; the end-to-end test diarizes pyannote's bundled sample twice — host-only and on-device — and compares the two with diarization error rate. That keeps the assertion recording-independent: bf16 arithmetic can split a turn at a boundary without changing who spoke when.

## Notes

- The last time-chunk of a recording can be very narrow (time-width `W` down to 1). `ttnn_wespeaker_resident` zero-pads such conv inputs up to a safe width and crops the output back, so every conv runs on device with no host fallback. `tests/test_resident_narrow_ondevice.py` checks that path for `W = 1, 2, 4, 8, 12`: the cropped output keeps the exact unpadded shape and stays within bf16 tolerance of the numpy backbone (cosine > 0.99). This sidesteps a ttnn conv2d auto-shard reader-index assert (tenstorrent/tt-metal#35207, #43193).
- ttnn prints harmless `leaked function/type` noise on exit; filter it with `grep -viE 'leaked|nanobind'`.

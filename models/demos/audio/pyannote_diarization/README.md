# pyannote speaker-diarization-community-1

## Platforms:
    Blackhole (p150)

## Introduction

[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) is a speaker-diarization pipeline: it segments an audio recording into "who spoke when". The pipeline runs two neural nets — a **WeSpeaker ResNet34** speaker-embedding model and a **PyanNet** (SincNet + BiLSTM) local-segmentation model — around host-side clustering. This demo ports both neural nets to ttnn so they execute on a Tenstorrent Blackhole p150, matching the torch reference bit-for-bit within tolerance (WeSpeaker embedding cosine > 0.99; PyanNet frame agreement 0.998 vs CPU).

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `pip install "pyannote.audio==4.*" soundfile` (community-1 requires pyannote 4.x)
- community-1 weights on disk (`config.yaml` + `embedding/` + `segmentation/` + `plda/`); the non-gated mirror [`pyannote-community/speaker-diarization-community-1`](https://huggingface.co/pyannote-community/speaker-diarization-community-1) matches the official SHAs and needs no HF token.

## How to Run

### Device-independent reference parity (no p150 needed)

Validates the numpy op-graph against the real torch models (skips cleanly without torch/pyannote/weights):

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_wespeaker_numpy_ref_parity.py
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_pyannet_numpy_ref_parity.py
```

### On-device parity (p150)

Runs the device-resident WeSpeaker backbone and checks it matches the numpy backbone across a sweep of narrow time-widths (`W = 1, 2, 4, 8, 12`):

```sh
pytest --disable-warnings models/demos/audio/pyannote_diarization/tests/test_resident_narrow_ondevice.py
```

### Full diarization demo (p150)

`demo/tt_full_diarization.py` runs the whole community-1 pipeline with the TT neural nets and prints the `{start, end, speaker}` segments. By default the WeSpeaker embedding runs on device and PyanNet segmentation runs on host (cheap); set `DIARIZATION_TT_SEGMENTATION=1` to run both neural nets on device.

## Details

### Structure

- `tt/` — ttnn implementations:
  - `ttnn_wespeaker.py` — `TTNNWeSpeaker`, the ResNet34 speaker embedding.
  - `ttnn_wespeaker_resident.py` — device-resident fast path: the activation stays on device across the whole ResNet34 (input uploaded once, every conv/relu/residual runs on device, only the final feature map is downloaded); TSTP pooling and the `seg_1` linear stay on host so pyannote's exact (optionally weighted) pooling is preserved bit-for-bit.
  - `ttnn_pyannet.py` — the PyanNet SincNet + BiLSTM local-segmentation net.
- `reference/` — numpy reference op-graphs (`wespeaker_numpy_ref.py`, `pyannet_numpy_ref.py`), each parity-checked against the real torch model.
- `tests/` — the parity tests listed above.
- `demo/` — runnable parity scripts (`run_ttnn_parity*.py`, `run_ttnn_seg_parity*.py`), `gen_golden.py`, and the end-to-end `tt_full_diarization.py` / `tt_diarization_integration.py`.

### What runs on device

Both neural nets are executed with ttnn: WeSpeaker convolutions, residual-adds, relu and reductions; PyanNet SincNet convolutions and the BiLSTM (implemented as a device-resident batched recurrence, since ttnn has no fused LSTM). Host keeps only the pyannote clustering/pooling glue, matching the CPU pipeline numerically.

## Notes

- Open the device with `ttnn.open_device(device_id=0, l1_small_size=32768)`; `l1_small_size` is required.
- The last time-chunk of a recording can be very narrow (time-width `W` down to 1). `ttnn_wespeaker_resident` zero-pads such conv inputs up to a safe width and crops the output back, so every conv runs on device (no host fallback) and the result is numerically identical. This sidesteps a ttnn conv2d auto-shard reader-index assert (tenstorrent/tt-metal#35207, #43193).
- ttnn prints harmless `leaked function/type` noise on exit; filter it with `grep -viE 'leaked|nanobind'`.

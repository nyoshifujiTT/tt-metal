# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Score community-1 over a whole diarization corpus and report the DER.

This is a benchmark, not a test: it runs for hours over hundreds of recordings
and prints a number, so it lives here as a CLI script rather than under
``tests/`` (the same split mamba uses for its lm-eval harness). The pytest
suite keeps a short fixed-size regression check instead.

Prepare a corpus as

    <root>/audio/<id>.wav
    <root>/rttm/<id>.rttm

VoxConverse is the practical choice: audio and RTTMs are both CC-BY downloads.
AMI is open after registration; DIHARD needs an LDC licence.

Use the split the published figure was measured on -- pyannote reports
VoxConverse on *test*, and scoring *dev* against that number lands well under
it because dev is the easier half, which reads as a pass while measuring the
wrong thing. Splits with no published figure of their own are reported without
a verdict.

Usage:
    python models/demos/audio/pyannote_diarization/benchmarks/corpus_der.py \
        --corpus /path/to/voxconverse-test --split voxconverse-test

    # quick check over the first few recordings
    ... --limit 5

    # embedding on device only (segmentation stays on host)
    ... --no-offload-segmentation
"""

import argparse
import json
import sys
import time

import torch

from models.demos.audio.pyannote_diarization import accuracy
from models.demos.audio.pyannote_diarization.pipeline import (
    load_pipeline,
    offload_embedding,
    offload_segmentation,
)


def diarize_with(pipeline):
    """Return ``diarize(wav_path) -> turns`` for :func:`accuracy.corpus_der`."""
    import soundfile as sf

    def diarize(wav_path):
        # Hand pyannote an in-memory waveform: decoding a path goes through
        # torchcodec, whose wheels are tied to a torch release the image the
        # service ships does not carry.
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = {
            "waveform": torch.from_numpy(samples.T.copy()),
            "sample_rate": sample_rate,
        }
        annotation = pipeline(audio).speaker_diarization
        return [
            {"speaker": speaker, "start": segment.start, "end": segment.end}
            for segment, _, speaker in annotation.itertracks(yield_label=True)
        ]

    return diarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        required=True,
        help="corpus root holding audio/<id>.wav and rttm/<id>.rttm",
    )
    parser.add_argument(
        "--split",
        default="voxconverse-test",
        help="split name, which selects the published DER to compare against",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N recordings (sorted by id, so reproducible)",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--no-offload-segmentation",
        action="store_true",
        help="run only the embedding net on device, as the service does by default",
    )
    parser.add_argument("--output", help="write the full result as JSON to this path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import ttnn

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768)
    try:
        pipeline = load_pipeline()
        offload_embedding(pipeline, device)
        if not args.no_offload_segmentation:
            offload_segmentation(pipeline, device)

        started = time.monotonic()
        scored = accuracy.corpus_der(diarize_with(pipeline), args.corpus, limit=args.limit)
        elapsed = time.monotonic() - started
    finally:
        ttnn.close_device(device)

    published = accuracy.published_corpus_der(args.split)
    worst = sorted(scored["per_recording"].items(), key=lambda kv: -kv[1])[:3]

    print(f"{args.split}: DER={scored['der']:.5f} over " f"{scored['num_recordings']} recordings in {elapsed:.0f}s")
    print(f"worst: {[(name, round(der, 4)) for name, der in worst]}")
    if published is None:
        # Deliberately not compared against another split's figure: that is how
        # a dev-split run gets mistaken for a pass.
        print(f"no published DER for split {args.split!r}; nothing to compare against")
        verdict = None
    else:
        ceiling = published + accuracy.CORPUS_DER_TOLERANCE
        verdict = scored["der"] <= ceiling
        print(f"published={published:.4f} ceiling={ceiling:.4f} -> " f"{'PASS' if verdict else 'FAIL'}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "der": scored["der"],
                    "num_recordings": scored["num_recordings"],
                    "elapsed_seconds": elapsed,
                    "published_der": published,
                    "verdict": verdict,
                    "per_recording": scored["per_recording"],
                },
                f,
                indent=2,
            )
        print(f"wrote {args.output}")

    return 0 if verdict is not False else 1


if __name__ == "__main__":
    sys.exit(main())

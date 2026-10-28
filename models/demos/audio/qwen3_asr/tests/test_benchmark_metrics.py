# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Speed metrics must follow the standard ASR definition.

RTF = processing_time / audio_duration and RTFx = audio_duration /
processing_time, where audio_duration is the ORIGINAL waveform length. vLLM's own
ASR benchmark computes rtfx = input_audio_duration / duration, and the Open ASR
Leaderboard reports RTFx the same way.

Deriving the duration from the mel frame count instead is a different number: the
extractor pads every clip to a 30s window and the eval was summing padded frames,
so 1892.0 s of real audio was reported as 1649.4 s - a 1.147x error that made the
demo-side and served-side numbers incomparable even though both looked like
"audio seconds per second".
"""

import os

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "eval", "corpus_eval.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


def test_audio_duration_comes_from_the_waveform():
    src = _read(EVAL)
    assert "clip_seconds = len(wav) / float(sr)" in src, "duration must be the waveform length"
    assert "audio_total += clip_seconds" in src


def test_mel_frame_count_is_not_used_as_a_duration():
    src = _read(EVAL)
    assert "audio_total += nf / 100.0" not in src, "mel frames are not the audio duration"


def test_report_exposes_rtfx():
    # The name matters: rtfx is what vLLM's benchmark and the Open ASR
    # Leaderboard report, so a reader can compare against published numbers.
    src = _read(EVAL)
    assert '"rtfx"' in src, "the report must expose rtfx"
    assert "audio_total / wall" in src, "rtfx is audio duration over wall time"


def test_readme_documents_the_metric_definition():
    readme = _read(os.path.join(HERE, "..", "README.md"))
    assert "rtfx" in readme, "the README must name the metric it reports"
    assert "ORIGINAL waveform duration" in readme, "and pin down the denominator"


def test_readme_warns_the_two_evals_measure_different_workloads():
    # Same metric name, different workload: one is a direct single-stream driver,
    # the other goes over HTTP and may be concurrent. Comparing them numerically
    # is the mistake this note exists to prevent.
    readme = _read(os.path.join(HERE, "..", "README.md"))
    assert "single-stream" in readme
    assert "must not be compared" in readme

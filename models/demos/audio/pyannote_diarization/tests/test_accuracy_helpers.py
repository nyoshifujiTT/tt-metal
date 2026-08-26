# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Unit tests for the shared diarization scoring helpers (no device needed).

tt-inference-server's eval workflow scores the served model through these same
functions, so a change here moves both reported numbers; they are pinned.
"""
import pytest

pytest.importorskip("pyannote.core")
pytest.importorskip("pyannote.metrics")

from models.demos.audio.pyannote_diarization import accuracy  # noqa: E402


def test_rttm_onset_duration_becomes_onset_offset(tmp_path):
    """RTTM stores duration; the Annotation must hold the end time."""
    path = tmp_path / "ref.rttm"
    path.write_text(
        "SPEAKER sample 1 6.690 0.430 <NA> <NA> speaker90 <NA> <NA>\n"
        "SPEAKER sample 1 7.550 0.800 <NA> <NA> speaker91 <NA> <NA>\n"
        "\n"
    )

    annotation = accuracy.load_rttm(str(path))

    assert sorted(annotation.labels()) == ["speaker90", "speaker91"]
    first = next(iter(annotation.itertracks(yield_label=True)))
    assert first[0].start == pytest.approx(6.690)
    assert first[0].end == pytest.approx(7.120)


def test_served_turns_and_rttm_are_directly_comparable():
    """The API's turn shape must score against an RTTM without conversion."""
    turns = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0},
    ]
    hypothesis = accuracy.turns_to_annotation(turns)

    assert accuracy.diarization_error_rate(hypothesis, hypothesis) == pytest.approx(0.0)
    assert accuracy.speaker_count(hypothesis) == 2


def test_scoring_reports_speaker_counts_separately_from_the_der():
    """A merged speaker can still score a low DER, so the count is its own field."""
    reference = accuracy.turns_to_annotation(
        [
            {"speaker": "A", "start": 0.0, "end": 1.0},
            {"speaker": "B", "start": 1.0, "end": 2.0},
        ]
    )
    merged = accuracy.turns_to_annotation([{"speaker": "A", "start": 0.0, "end": 2.0}])

    scored = accuracy.score_against_reference(merged, reference)

    assert scored["num_speakers"] == 1
    assert scored["reference_num_speakers"] == 2
    assert scored["speaker_count_matches"] is False


def test_thresholds_keep_fidelity_stricter_than_absolute_accuracy():
    """Device-vs-host must reproduce exactly; annotation agreement never does.

    Both host and device score ~0.05 against the shipped annotation, which is
    the ordinary gap between a diarizer and a human transcript rather than a
    porting error, so the accuracy gate has to be the looser of the two.
    """
    assert accuracy.FIDELITY_DER_MAX < accuracy.ACCURACY_DER_MAX
    assert accuracy.ACCURACY_DER_MAX < accuracy.PUBLISHED_DER

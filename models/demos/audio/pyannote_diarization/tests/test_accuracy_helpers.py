# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Unit tests for the shared diarization scoring helpers (no device needed).

tt-inference-server's eval workflow scores the served model through these same
functions, so a change here moves both reported numbers; they are pinned.
"""
import pytest

from models.demos.audio.pyannote_diarization import accuracy


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


def _write_rttm(path, turns):
    lines = [
        f"SPEAKER rec 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>\n" for speaker, start, end in turns
    ]
    path.write_text("".join(lines))


def test_corpus_pairs_audio_with_its_annotation_in_a_stable_order(tmp_path):
    """A limited run must pick the same recordings every time."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "rttm").mkdir()
    for name in ("b", "a", "c"):
        (tmp_path / "audio" / f"{name}.wav").write_bytes(b"RIFF")
        _write_rttm(tmp_path / "rttm" / f"{name}.rttm", [("s1", 0.0, 1.0)])
    # An audio file with no annotation must be dropped, not scored as empty.
    (tmp_path / "audio" / "unannotated.wav").write_bytes(b"RIFF")

    pairs = accuracy.corpus_files(str(tmp_path))

    assert [recording_id for recording_id, _, _ in pairs] == ["a", "b", "c"]
    assert [r for r, _, _ in accuracy.corpus_files(str(tmp_path), limit=2)] == ["a", "b"]


def test_corpus_der_accumulates_over_recordings_rather_than_averaging(tmp_path):
    """Long recordings must weigh more, which is how published DERs are computed."""
    (tmp_path / "audio").mkdir()
    (tmp_path / "rttm").mkdir()
    # 'long' is 100 s and scored perfectly; 'short' is 1 s and scored entirely
    # wrong. A per-file mean would be ~0.5; accumulating gives ~1/101.
    _write_rttm(tmp_path / "rttm" / "long.rttm", [("s1", 0.0, 100.0)])
    _write_rttm(tmp_path / "rttm" / "short.rttm", [("s1", 0.0, 1.0)])
    for name in ("long", "short"):
        (tmp_path / "audio" / f"{name}.wav").write_bytes(b"RIFF")

    def diarize(wav_path):
        if wav_path.endswith("long.wav"):
            return [{"speaker": "s1", "start": 0.0, "end": 100.0}]
        return []  # missed the whole recording

    scored = accuracy.corpus_der(diarize, str(tmp_path))

    assert scored["num_recordings"] == 2
    assert scored["der"] == pytest.approx(1.0 / 101.0, abs=1e-6)
    assert scored["per_recording"]["short"] == pytest.approx(1.0)


def test_corpus_root_prefers_the_per_corpus_variable(tmp_path, monkeypatch):
    generic = tmp_path / "generic"
    specific = tmp_path / "specific"
    generic.mkdir()
    specific.mkdir()
    monkeypatch.setenv("DIARIZATION_CORPUS_DIR", str(generic))
    monkeypatch.setenv("DIARIZATION_VOXCONVERSE_DIR", str(specific))

    assert accuracy.corpus_root("voxconverse") == str(specific)

    monkeypatch.delenv("DIARIZATION_VOXCONVERSE_DIR")
    assert accuracy.corpus_root("voxconverse") == str(generic)

    monkeypatch.delenv("DIARIZATION_CORPUS_DIR")
    assert accuracy.corpus_root("voxconverse") is None


def test_published_figures_are_keyed_by_split():
    """A split without its own published figure must not borrow another's.

    VoxConverse dev scores well under the published 0.112 because that figure
    is measured on test; gating dev against it would report a pass while
    measuring the easier half.
    """
    assert accuracy.published_corpus_der("voxconverse-test") == pytest.approx(0.112)
    assert accuracy.published_corpus_der("voxconverse-dev") is None
    assert accuracy.published_corpus_der("nonexistent-corpus") is None
    # The single-sample gate is unrelated to these and must not be reused.
    assert accuracy.published_corpus_der("voxconverse-test") < accuracy.ACCURACY_DER_MAX


def test_corpus_root_accepts_a_hyphenated_split_name(tmp_path, monkeypatch):
    """Split names carry a hyphen; the env var spelling must still resolve."""
    root = tmp_path / "vc-test"
    root.mkdir()
    monkeypatch.delenv("DIARIZATION_CORPUS_DIR", raising=False)
    monkeypatch.setenv("DIARIZATION_VOXCONVERSE_TEST_DIR", str(root))

    assert accuracy.corpus_root("voxconverse-test") == str(root)

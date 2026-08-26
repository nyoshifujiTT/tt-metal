# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""End-to-end community-1 diarization with the neural nets on the p150.

Two things are checked, and they answer different questions:

* *fidelity* -- the pipeline is run twice over the same recording, once
  entirely on host and once with the WeSpeaker embedding (and optionally the
  PyanNet segmentation) executed through ttnn, and the device run must
  reproduce the host run. This catches a ttnn kernel drifting from the
  reference implementation.
* *accuracy* -- the device run is scored against the human annotation shipped
  with the sample. This catches the pipeline being wrong in absolute terms, in
  a way host and device would share and fidelity alone would never reveal.

Both use diarization error rate, the standard metric, differing only in what
serves as the reference; the scoring lives in ``common``'s sibling
``accuracy`` module so tt-inference-server's eval workflow reports the same
number for the served model. Raw segment counts are not used: bf16 arithmetic
can split or merge a turn at a boundary without changing who is speaking when.
The audio and its annotation both ship with pyannote.audio, so no external
fixture is needed.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("ttnn")
pytest.importorskip("pyannote.audio")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.demos.audio.pyannote_diarization import accuracy, common  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import TTNNPyanNet  # noqa: E402
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker  # noqa: E402


def _install_torch_load_shim():
    """pyannote checkpoints hold non-tensor objects; lightning>=2.6 flips the default."""
    original = torch.load
    if getattr(original, "_diar_shim", False):
        return

    def patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    patched._diar_shim = True
    torch.load = patched


def _load_pipeline(model_location_generator):
    _install_torch_load_shim()
    from pyannote.audio import Pipeline

    config = common.resolve_weights(common.PIPELINE_RELPATH, model_location_generator)
    pipeline = Pipeline.from_pretrained(config)
    pipeline.to(torch.device("cpu"))
    return pipeline


def _sample_audio():
    from pyannote.audio.sample import SAMPLE_FILE

    return str(SAMPLE_FILE["audio"])


def _speakers(diarization):
    return {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}


def _offload_embedding(pipeline, device):
    """Run the WeSpeaker ResNet34 conv backbone through ttnn."""
    wespeaker = pipeline._embedding.model_
    resnet = wespeaker.resnet
    tt = TTNNWeSpeaker(wespeaker.state_dict(), device)
    tt.use_device_elementwise = True

    def backbone(feats1):
        x = tt._relu_dev(tt._conv(feats1, tt.folded["conv1"], 1))
        for layer, blocks in enumerate(tt.BLOCKS, start=1):
            for block in range(blocks):
                stride = 2 if (block == 0 and layer > 1) else 1
                x = tt._block(x, f"resnet.layer{layer}.{block}", stride)
        return x

    def forward(fbank, weights=None):
        feats = fbank.permute(0, 2, 1).unsqueeze(1).float()
        outs = [backbone(feats[i : i + 1]) for i in range(feats.shape[0])]
        frames = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :frames] for o in outs], dim=0)
        return torch.tensor(0.0), resnet.seg_1(resnet.pool(x, weights=weights))

    resnet.forward = forward


def _offload_segmentation(pipeline, device):
    """Run the PyanNet segmentation net through ttnn."""
    model = pipeline._segmentation.model
    sinc = model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    tt = TTNNPyanNet(model.state_dict(), sinc, device)

    def forward(waveforms):
        logits = tt.forward_batch(waveforms.detach().numpy())
        return F.log_softmax(torch.from_numpy(logits).float(), dim=-1)

    model.forward = forward


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("offload_segmentation", [False, True], ids=["embedding_only", "both_nets"])
def test_diarization_matches_host_pipeline(device, model_location_generator, offload_segmentation):
    audio = _sample_audio()

    host = _load_pipeline(model_location_generator)(audio).speaker_diarization

    pipeline = _load_pipeline(model_location_generator)
    _offload_embedding(pipeline, device)
    if offload_segmentation:
        _offload_segmentation(pipeline, device)
    on_device = pipeline(audio).speaker_diarization

    assert len(_speakers(host)) >= 2, "host pipeline should find multiple speakers in the sample recording"
    assert len(_speakers(on_device)) == len(_speakers(host)), (
        f"speaker count differs: device={sorted(_speakers(on_device))} " f"host={sorted(_speakers(host))}"
    )

    der = accuracy.diarization_error_rate(host, on_device)
    assert der < accuracy.FIDELITY_DER_MAX, f"diarization error rate against the host pipeline too high: {der}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("offload_segmentation", [False, True], ids=["embedding_only", "both_nets"])
def test_diarization_matches_human_annotation(device, model_location_generator, offload_segmentation):
    """The device pipeline must be right in absolute terms, not just consistent.

    Fidelity against the host run cannot catch a pipeline that is misconfigured
    identically on both sides -- wrong clustering threshold, wrong checkpoint --
    because both runs would agree while both are wrong. Scoring against the
    human annotation that ships with the sample catches that.

    This is the same measurement tt-inference-server's eval workflow reports for
    the served model, through the same ``accuracy`` helpers.
    """
    pipeline = _load_pipeline(model_location_generator)
    _offload_embedding(pipeline, device)
    if offload_segmentation:
        _offload_segmentation(pipeline, device)
    on_device = pipeline(_sample_audio()).speaker_diarization

    reference = accuracy.load_rttm(accuracy.sample_reference_path())
    scored = accuracy.score_against_reference(on_device, reference)

    assert scored["speaker_count_matches"], (
        f"speaker count differs from the annotation: device={scored['num_speakers']} "
        f"reference={scored['reference_num_speakers']}"
    )
    assert scored["der"] < accuracy.ACCURACY_DER_MAX, (
        f"diarization error rate against the human annotation too high: {scored['der']} "
        f"(published DER for this model is {accuracy.PUBLISHED_DER}, see {accuracy.PUBLISHED_DER_REF})"
    )

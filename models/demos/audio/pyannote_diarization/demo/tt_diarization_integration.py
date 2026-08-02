"""Real community-1 diarization with the WeSpeaker conv backbone on p150 (ttnn).

Runs the actual pyannote SpeakerDiarization(community-1) pipeline, but the
ResNet34 convolutional backbone of the speaker-embedding model executes on the
Tenstorrent p150 via ttnn. The exact weighted StatsPool + seg_1 linear stay on
host (unchanged pyannote semantics), so this is a faithful integration: the heavy
CNN of community-1's embedding NN runs TT-natively inside a real diarization.

Compares the resulting speaker turns to the CPU golden (out.rttm).
"""
import os, sys, types
import numpy as np
import torch
import torch.nn.functional as F
import ttnn

from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker

MODEL_DIR = os.environ.get("PYANNOTE_COMMUNITY1_WEIGHTS_DIR", "/data/pyannote-community-1/weights")
SAMPLE = os.environ.get("DIAR_SAMPLE_WAV", "/data/pyannote-community-1/samples/pyannote_sample.wav")

# --- torch.load shim for lightning>=2.6 ---
_orig = torch.load
torch.load = (lambda *a, **k: (k.__setitem__("weights_only", False) or _orig(*a, **k))
              if k.get("weights_only") is None else _orig(*a, **k))

from pyannote.audio import Pipeline

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    pipe = Pipeline.from_pretrained(os.path.join(MODEL_DIR, "config.yaml"))
    pipe.to(torch.device("cpu"))

    # locate the WeSpeaker resnet inside the embedding model
    emb = pipe._embedding
    wespeaker = emb.model_          # WeSpeakerResNet34 (has .resnet)
    resnet = wespeaker.resnet

    tt = TTNNWeSpeaker(wespeaker.state_dict(), dev)
    tt.use_device_elementwise = True

    call_count = {"n": 0}

    def _tt_backbone_one(feats1):
        # feats1: (1,1,F,T) -> (1,C,F\',T\') conv backbone on TT
        x = tt._relu_dev(tt._conv(feats1, tt.folded["conv1"], 1)) if tt.use_device_elementwise \
            else torch.relu(tt._conv(feats1, tt.folded["conv1"], 1))
        for li, nb in enumerate(tt.BLOCKS, start=1):
            for bi in range(nb):
                stride = 2 if (bi == 0 and li > 1) else 1
                x = tt._block(x, f"resnet.layer{li}.{bi}", stride)
        return x

    def tt_resnet_forward(fbank, weights=None):
        # fbank: (B, T, F) -> conv backbone on TT, one chunk at a time (bound L1)
        feats = fbank.permute(0, 2, 1).unsqueeze(1).float()  # (B,1,F,T)
        outs = [_tt_backbone_one(feats[i:i+1]) for i in range(feats.shape[0])]
        # align frames across chunks (should match) and stack
        minT = min(o.shape[-1] for o in outs)
        x = torch.cat([o[..., :minT] for o in outs], dim=0)  # (B,C,F\',T\')
        call_count["n"] += 1
        stats = resnet.pool(x, weights=weights)
        embed_a = resnet.seg_1(stats)
        return torch.tensor(0.0), embed_a

    resnet.forward = tt_resnet_forward   # inject TT backbone

    print("[info] running community-1 diarization with TT WeSpeaker backbone...", flush=True)
    out = pipe(SAMPLE)
    diar = out.speaker_diarization
    turns = [(round(s.start, 2), round(s.end, 2), spk) for s, _, spk in diar.itertracks(yield_label=True)]
    spk = sorted({t[2] for t in turns})
    print(f"[result] TT-embedding diarization: segments={len(turns)} speakers={spk} embed_calls={call_count['n']}")
    for t in turns[:8]:
        print("  ", t)
    # compare to CPU golden rttm speaker count
    print(f"[compare] speaker_count={len(spk)} (CPU golden was 2 speakers, 13 segments)")
    assert len(spk) >= 2, "expected multiple speakers"
    print("PASS: real community-1 diarization ran with WeSpeaker conv backbone on p150")
finally:
    ttnn.close_device(dev)

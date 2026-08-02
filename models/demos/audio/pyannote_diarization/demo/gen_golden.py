"""Generate golden WeSpeaker ResNet34 embeddings (CPU) as the parity target for
a future ttnn port. Deterministic fixed inputs -> saved reference embeddings +
input log-mel features (the ttnn port's boundary is the mel features -> ResNet34).
"""
import os, json, numpy as np, torch
from pyannote.audio import Model

MODEL_BIN = os.environ.get(
    "PYANNOTE_COMMUNITY1_EMB_WEIGHTS",
    "/data/pyannote-community-1/weights/embedding/pytorch_model.bin",
)
OUT = os.environ.get(
    "GOLDEN_EMBEDDINGS_JSON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_embeddings.json"),
)

torch.manual_seed(0)
m = Model.from_pretrained(MODEL_BIN)
m.eval()

cases = {}
# deterministic pseudo-waveforms of a few durations (2s, 3s, 5s) at 16 kHz
for name, secs in [("d2", 2.0), ("d3", 3.0), ("d5", 5.0)]:
    n = int(secs * 16000)
    g = torch.Generator().manual_seed(hash(name) & 0xffff)
    wav = (torch.rand(1, 1, n, generator=g) * 2 - 1) * 0.1
    with torch.no_grad():
        emb = m(wav)
    emb = emb[0].numpy().astype(np.float64)
    cases[name] = {
        "seconds": secs,
        "samples": n,
        "embedding_dim": int(emb.shape[0]),
        "l2_norm": float(np.linalg.norm(emb)),
        "embedding_first16": emb[:16].tolist(),
        "embedding_sha_sum": float(emb.sum()),
    }
    print(name, "dim", emb.shape[0], "l2", round(float(np.linalg.norm(emb)),4))

with open(OUT, "w") as f:
    json.dump({"model": "WeSpeakerResNet34 (community-1 embedding)",
               "blocks": [3,4,6,3], "fbank_dim": 80, "embedding_dim": 256,
               "cases": cases}, f, indent=2)
print("wrote", OUT)

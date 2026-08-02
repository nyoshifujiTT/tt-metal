"""On-device (p150) parity: TTNN WeSpeaker embedding vs torch reference emb.

Fixture paths are overridable with PARITY_INPUT_NPZ / STATE_DICT_NPZ; the module
dir is added to sys.path relatively so this runs from any checkout location.
"""
import os, sys, numpy as np, torch, ttnn

d = np.load(os.environ.get("PARITY_INPUT_NPZ", "parity_input.npz"))
feats = torch.from_numpy(d["feats"]).float()
emb_torch = d["emb_torch"]
sd_npz = np.load(os.environ.get("STATE_DICT_NPZ", "state_dict.npz"))
# WeSpeakerNumpyRef expects objects with .numpy(); wrap numpy arrays as torch tensors
state_dict = {k: torch.from_numpy(sd_npz[k]) for k in sd_npz.files}

from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    model = TTNNWeSpeaker(state_dict, dev)
    model.use_device_elementwise = True  # relu/add on device
    model.use_device_pool = True  # TSTP+linear on device
    emb_tt = model.forward(feats).numpy()
finally:
    ttnn.close_device(dev)

cos = float(np.dot(emb_tt[0], emb_torch[0]) / (np.linalg.norm(emb_tt[0]) * np.linalg.norm(emb_torch[0])))
maxabs = float(np.max(np.abs(emb_tt - emb_torch)))
print(f"PARITY cos={cos:.5f} max_abs={maxabs:.4f}")
assert cos > 0.99, f"cosine too low {cos}"
print("PASS: ttnn WeSpeaker embedding matches torch reference on p150")

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""ttnn implementation of PyanNet (community-1 segmentation) for p150.

Parity target = pyannet_numpy_ref (which matches torch, cos ~1.0). The heavy
neural-net compute runs on device via ttnn:
  - SincNet conv1d layers: host numpy conv (materialized sinc filters), ~2ms and
    numerically identical; the expensive part is the recurrence, not this conv.
  - BiLSTM x4: **device-resident, batched over all sliding windows**. Weights and
    the (h, c) state stay on device; the per-timestep input projection is a single
    batched matmul, and every gate (sigmoid/tanh), the cell update and the hidden
    update run via ttnn ops. There is NO per-timestep host<->device transfer, and
    all windows of a chunk are processed together as one batch. This removes the
    ~896 host round-trips/window that made the naive port ~15x slower than CPU.
  - linear x2 + classifier: ttnn.linear on device, batched.
InstanceNorm / maxpool / abs / leaky_relu stay on host (cheap pointwise/pooling
that ttnn lacks a direct fused op for). The boundary is documented; the heavy
linear algebra (LSTM matmuls + linears) runs on the p150.

`forward` keeps the original single-window path (used by parity tests). The
diarization accelerator uses `forward_batch` to run every window at once.
"""
from __future__ import annotations

import numpy as np
import torch
import ttnn

from models.demos.audio.pyannote_diarization.reference.pyannet_numpy_ref import (
    instance_norm_1d, leaky_relu, conv1d, maxpool1d,
)


def _tt_linear(device, x_np, w_np, b_np):
    """y = x @ w.T + b on device. x:(N,in) w:(out,in) b:(out,) -> (N,out)."""
    tx = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(x_np)).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(torch.from_numpy(w_np.T.copy()).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(torch.from_numpy(b_np.reshape(1, -1).copy()).to(torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=device)
    ty = ttnn.linear(tx, tw, bias=tb)
    return ttnn.to_torch(ty).float().numpy()


class _DevBiLSTM:
    """One BiLSTM layer, device-resident, batched over windows AND directions.

    Both LSTM directions are stacked into a leading dim of size 2 so the whole
    layer runs a single recurrence: per timestep there is one 3D batched matmul
    (dir,B,h)@(dir,h,4h) for the recurrent term instead of two separate matmuls,
    halving the per-step op-dispatch count. Weights/biases/state stay on device;
    the per-timestep input projection is one big matmul done up front; only the
    final (T,2,B,h) output is downloaded once (no per-timestep host transfer).
    """

    def __init__(self, sd, li, device, hidden=128):
        self.dev = device
        self.h = hidden

        def dv(a):
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT, device=device)

        self.wih = dv(sd[f"lstm.weight_ih_l{li}"].T)  # (in,4h)
        self.whh = dv(sd[f"lstm.weight_hh_l{li}"].T)  # (h,4h)
        self.bih = dv(sd[f"lstm.bias_ih_l{li}"].reshape(1, -1))
        self.bhh = dv(sd[f"lstm.bias_hh_l{li}"].reshape(1, -1))
        self.wihr = dv(sd[f"lstm.weight_ih_l{li}_reverse"].T)
        self.whhr = dv(sd[f"lstm.weight_hh_l{li}_reverse"].T)
        self.bihr = dv(sd[f"lstm.bias_ih_l{li}_reverse"].reshape(1, -1))
        self.bhhr = dv(sd[f"lstm.bias_hh_l{li}_reverse"].reshape(1, -1))
        h = hidden
        # 3D (direction-batched) recurrent weights/bias for the fused loop:
        #   dir 0 = forward, dir 1 = reverse
        self.whh3 = dv(np.stack([sd[f"lstm.weight_hh_l{li}"].T,
                                 sd[f"lstm.weight_hh_l{li}_reverse"].T], axis=0))     # (2,h,4h)
        self.bhh3 = dv(np.stack([sd[f"lstm.bias_hh_l{li}"],
                                 sd[f"lstm.bias_hh_l{li}_reverse"]], axis=0).reshape(2, 1, 4 * h))

    def _dir(self, GXtb, B, T, whh, bhh):
        """Recurrence, fully device-resident. GXtb: device (T*B,4h) time-major
        (block t = rows [t*B:(t+1)*B]); input projection precomputed by caller.
        Only the final (T,B,h) output is downloaded once (no per-step transfer)."""
        h = self.h
        ht = ttnn.from_torch(torch.zeros(B, h, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=self.dev)
        ct = ttnn.from_torch(torch.zeros(B, h, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=self.dev)
        outs = []
        for t in range(T):
            gx = ttnn.slice(GXtb, [t * B, 0], [t * B + B, 4 * h])  # (B,4h) outer-dim slice
            gh = ttnn.linear(ht, whh, bias=bhh)               # (B,4h) device
            g = ttnn.add(gx, gh)
            i = ttnn.sigmoid(g[:, 0 * h:1 * h])
            f = ttnn.sigmoid(g[:, 1 * h:2 * h])
            gg = ttnn.tanh(g[:, 2 * h:3 * h])
            o = ttnn.sigmoid(g[:, 3 * h:4 * h])
            ct = ttnn.add(ttnn.multiply(f, ct), ttnn.multiply(i, gg))
            ht = ttnn.multiply(o, ttnn.tanh(ct))
            outs.append(ht)
        stacked = ttnn.concat(outs, dim=0)                    # (T*B,h)
        return ttnn.to_torch(stacked).float().reshape(T, B, h)

    def forward(self, X):
        """X: np (B,T,in) -> np (B,T,2h).

        Both directions run in a SINGLE device recurrence: the (h,c) state has
        2*B rows (rows [0:B]=forward, [B:2B]=reverse-time), so each timestep is
        one recurrent 3D batched matmul (2,B,h)@(2,h,4h) plus lean gate ops.
        The input projection is done up front (one matmul per direction), stacked
        time-major, and sliced per step from device. Only the final output is
        downloaded once."""
        B, T, _ = X.shape
        h = self.h
        # forward-time and reverse-time input sequences, time-major (T,B,in)
        Xf = np.ascontiguousarray(np.transpose(X, (1, 0, 2)).reshape(T * B, -1))
        Xr = np.ascontiguousarray(np.transpose(X[:, ::-1, :], (1, 0, 2)).reshape(T * B, -1))
        Xf_d = ttnn.from_torch(torch.from_numpy(Xf).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.dev)
        Xr_d = ttnn.from_torch(torch.from_numpy(Xr).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.dev)
        GXf = ttnn.linear(Xf_d, self.wih, bias=self.bih)      # (T*B,4h)
        GXr = ttnn.linear(Xr_d, self.wihr, bias=self.bihr)    # (T*B,4h)
        GXf = ttnn.to_torch(GXf).float().reshape(T, B, 4 * h)
        GXr = ttnn.to_torch(GXr).float().reshape(T, B, 4 * h)
        # interleave to (T,2,B,4h) -> (T*2*B, 4h): block order per t = [fwd(B), rev(B)]
        GX = np.concatenate([GXf[:, None], GXr[:, None]], axis=1).reshape(T * 2 * B, 4 * h)
        GX_d = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(GX)).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=self.dev)
        ht = ttnn.from_torch(torch.zeros(2, B, h, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.dev)
        ct = ttnn.from_torch(torch.zeros(2, B, h, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.dev)
        outs = []
        for t in range(T):
            gx = ttnn.slice(GX_d, [t * 2 * B, 0], [t * 2 * B + 2 * B, 4 * h])  # (2B,4h)
            gh = ttnn.matmul(ht, self.whh3)                   # (2,B,4h) recurrent, per-direction
            gh = ttnn.add(gh, self.bhh3)
            gh = ttnn.reshape(gh, (2 * B, 4 * h))             # back to 2D for lean gate slicing
            g = ttnn.add(gx, gh)                              # (2B,4h)
            R = 2 * B
            sg = ttnn.sigmoid(g)                              # i,f,o in slots 0,1,3
            tg = ttnn.tanh(ttnn.slice(g, [0, 2 * h], [R, 3 * h]))   # gg
            i = ttnn.slice(sg, [0, 0 * h], [R, 1 * h])
            f = ttnn.slice(sg, [0, 1 * h], [R, 2 * h])
            o = ttnn.slice(sg, [0, 3 * h], [R, 4 * h])
            ct2 = ttnn.reshape(ct, (2 * B, h))
            ct2 = ttnn.add(ttnn.multiply(f, ct2), ttnn.multiply(i, tg))
            ht2 = ttnn.multiply(o, ttnn.tanh(ct2))           # (2B,h)
            ct = ttnn.reshape(ct2, (2, B, h))
            ht = ttnn.reshape(ht2, (2, B, h))
            outs.append(ttnn.reshape(ht2, (1, 2 * B, h)))
        stacked = ttnn.concat(outs, dim=0)                    # (T,2B,h)
        y = ttnn.to_torch(stacked).float().numpy().reshape(T, 2, B, h)
        fwd = np.transpose(y[:, 0], (1, 0, 2))                # (B,T,h)
        rev = np.transpose(y[:, 1], (1, 0, 2))[:, ::-1, :]    # (B,T,h) restore order
        return np.concatenate([np.ascontiguousarray(fwd), np.ascontiguousarray(rev)], axis=2)


class TTNNPyanNet:
    def __init__(self, state_dict, sinc_kernel, device):
        self.device = device
        self.sd = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in state_dict.items()}
        self.sinc_kernel = sinc_kernel
        self._bilstm = None  # lazily-built device-resident BiLSTM layers

    # ---- SincNet frontend (cheap; host numpy convs + instancenorm/pool) -------
    def sincnet(self, wav):
        sd = self.sd
        x = instance_norm_1d(wav, sd["sincnet.wav_norm1d.weight"], sd["sincnet.wav_norm1d.bias"])
        x = conv1d(x, self.sinc_kernel, None, stride=10)
        x = np.abs(x); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.0.weight"], sd["sincnet.norm1d.0.bias"]); x = leaky_relu(x)
        x = conv1d(x, sd["sincnet.conv1d.1.weight"], sd["sincnet.conv1d.1.bias"]); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.1.weight"], sd["sincnet.norm1d.1.bias"]); x = leaky_relu(x)
        x = conv1d(x, sd["sincnet.conv1d.2.weight"], sd["sincnet.conv1d.2.bias"]); x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.2.weight"], sd["sincnet.norm1d.2.bias"]); x = leaky_relu(x)
        return x  # (B,60,T)

    def _ensure_bilstm(self):
        if self._bilstm is None:
            self._bilstm = [_DevBiLSTM(self.sd, li, self.device) for li in range(4)]
        return self._bilstm

    # ---- naive single-timestep LSTM kept for reference/parity single-window ---
    def _lstm_dir(self, x, w_ih, w_hh, b_ih, b_hh, hidden, reverse=False):
        seq = np.ascontiguousarray(x[::-1]) if reverse else x
        T = seq.shape[0]
        h = np.zeros((1, hidden), dtype=np.float32)
        c = np.zeros((1, hidden), dtype=np.float32)
        out = np.zeros((T, hidden), dtype=np.float32)
        def sig(z): return 1.0 / (1.0 + np.exp(-z))
        for t in range(T):
            xt = seq[t:t + 1]
            gx = _tt_linear(self.device, xt, w_ih, b_ih)
            gh = _tt_linear(self.device, h, w_hh, b_hh)
            g = gx + gh
            i, f, gg, o = np.split(g[0], 4)
            i = sig(i); f = sig(f); gg = np.tanh(gg); o = sig(o)
            c = f * c + i * gg
            h = (o * np.tanh(c)).reshape(1, hidden)
            out[t] = h[0]
        return np.ascontiguousarray(out[::-1]) if reverse else out

    def bilstm(self, x, li, hidden=128):
        sd = self.sd
        fwd = self._lstm_dir(x, sd[f"lstm.weight_ih_l{li}"], sd[f"lstm.weight_hh_l{li}"],
                             sd[f"lstm.bias_ih_l{li}"], sd[f"lstm.bias_hh_l{li}"], hidden, reverse=False)
        rev = self._lstm_dir(x, sd[f"lstm.weight_ih_l{li}_reverse"], sd[f"lstm.weight_hh_l{li}_reverse"],
                             sd[f"lstm.bias_ih_l{li}_reverse"], sd[f"lstm.bias_hh_l{li}_reverse"], hidden, reverse=True)
        return np.concatenate([fwd, rev], axis=1)

    def forward(self, wav):
        """Single-window path (parity tests). wav: (1,1,S) -> logits (1,T,7)."""
        sd = self.sd
        feat = self.sincnet(wav)
        x = np.ascontiguousarray(feat[0].T)  # (T,60)
        for li in range(4):
            x = self.bilstm(x, li)
        x = leaky_relu(_tt_linear(self.device, x, sd["linear.0.weight"], sd["linear.0.bias"]))
        x = leaky_relu(_tt_linear(self.device, x, sd["linear.1.weight"], sd["linear.1.bias"]))
        logits = _tt_linear(self.device, x, sd["classifier.weight"], sd["classifier.bias"])
        return logits[None]

    # ---- fast batched path used by the diarization accelerator ---------------
    def forward_batch(self, wav_batch):
        """wav_batch: np (B,1,S) -> logits np (B,T,7). Device-resident batched.

        SincNet runs per window on host (cheap), then all windows share one
        device-resident batched BiLSTM x4 and batched device linears.
        """
        sd = self.sd
        B = wav_batch.shape[0]
        feats = [self.sincnet(wav_batch[i:i + 1])[0].T for i in range(B)]  # list of (T,60)
        T = min(f.shape[0] for f in feats)
        X = np.stack([np.ascontiguousarray(f[:T]) for f in feats], axis=0)  # (B,T,60)
        for L in self._ensure_bilstm():
            X = L.forward(X)                                   # (B,T,256)
        N = B * T
        X2 = X.reshape(N, -1)
        X2 = leaky_relu(_tt_linear(self.device, X2, sd["linear.0.weight"], sd["linear.0.bias"]))
        X2 = leaky_relu(_tt_linear(self.device, X2, sd["linear.1.weight"], sd["linear.1.bias"]))
        logits = _tt_linear(self.device, X2, sd["classifier.weight"], sd["classifier.bias"])
        return logits.reshape(B, T, -1)

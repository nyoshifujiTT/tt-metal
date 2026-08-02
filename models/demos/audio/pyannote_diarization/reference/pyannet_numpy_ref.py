"""Device-independent numpy reference of PyanNet (community-1 segmentation).

Op graph (verified from real model):
  wav_norm1d (InstanceNorm affine)
  SincNet:
    conv1d[0] = fixed materialized sinc filters (80,1,251), stride1 -> abs
                -> maxpool(3,3) -> InstanceNorm(80) -> leaky_relu
    conv1d[1] = Conv1d(80->60,k5) -> maxpool(3,3) -> InstanceNorm(60) -> leaky_relu
    conv1d[2] = Conv1d(60->60,k5) -> maxpool(3,3) -> InstanceNorm(60) -> leaky_relu
  BiLSTM x4 (hidden 128, monolithic bidirectional)
  linear(256->128) leaky_relu ; linear(128->128) leaky_relu
  classifier(128->7) ; activation (log/softmax? -> we compare pre-activation logits)

Uses torch only for conv1d/maxpool numerics (parity is what we validate); the
sinc kernel is materialized once and treated as a fixed conv weight, so the whole
net is standard conv1d + instancenorm + maxpool + leaky_relu + LSTM + linear ->
portable to ttnn. This module implements LSTM explicitly (matmul + gates) so the
recurrence is defined independently of torch.nn.LSTM.
"""
import numpy as np
import torch
import torch.nn.functional as F


def instance_norm_1d(x, weight, bias, eps=1e-5):
    # x: (B, C, T); normalize over T per (B,C), affine per channel
    mean = x.mean(axis=2, keepdims=True)
    var = x.var(axis=2, keepdims=True)  # biased (population), matches InstanceNorm
    xn = (x - mean) / np.sqrt(var + eps)
    return xn * weight[None, :, None] + bias[None, :, None]


def leaky_relu(x, slope=0.01):
    return np.where(x >= 0, x, slope * x)


def conv1d(x, w, b, stride=1):
    xt = torch.from_numpy(x).float()
    wt = torch.from_numpy(w).float()
    bt = torch.from_numpy(b).float() if b is not None else None
    return F.conv1d(xt, wt, bt, stride=stride).numpy()


def maxpool1d(x, k=3, s=3):
    return F.max_pool1d(torch.from_numpy(x).float(), k, s).numpy()


def lstm_layer(x, w_ih, w_hh, b_ih, b_hh, hidden):
    """Single-direction LSTM. x: (T, in) -> (T, hidden). Standard gates i,f,g,o."""
    T = x.shape[0]
    h = np.zeros(hidden, dtype=np.float64)
    c = np.zeros(hidden, dtype=np.float64)
    out = np.zeros((T, hidden), dtype=np.float64)
    def sig(z): return 1.0 / (1.0 + np.exp(-z))
    for t in range(T):
        g = x[t] @ w_ih.T + b_ih + h @ w_hh.T + b_hh  # (4*hidden,)
        i, f, gg, o = np.split(g, 4)
        i = sig(i); f = sig(f); gg = np.tanh(gg); o = sig(o)
        c = f * c + i * gg
        h = o * np.tanh(c)
        out[t] = h
    return out


def bilstm_layer(x, sd, li, hidden=128):
    """Bidirectional LSTM layer li. x: (T, in) -> (T, 2*hidden)."""
    fwd = lstm_layer(
        x, sd[f"lstm.weight_ih_l{li}"], sd[f"lstm.weight_hh_l{li}"],
        sd[f"lstm.bias_ih_l{li}"], sd[f"lstm.bias_hh_l{li}"], hidden,
    )
    xr = x[::-1]
    rev = lstm_layer(
        xr, sd[f"lstm.weight_ih_l{li}_reverse"], sd[f"lstm.weight_hh_l{li}_reverse"],
        sd[f"lstm.bias_ih_l{li}_reverse"], sd[f"lstm.bias_hh_l{li}_reverse"], hidden,
    )[::-1]
    return np.concatenate([fwd, rev], axis=1)


class PyanNetNumpyRef:
    def __init__(self, state_dict, sinc_kernel):
        self.sd = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in state_dict.items()}
        self.sinc_kernel = sinc_kernel  # (80,1,251)

    def sincnet(self, wav):
        sd = self.sd
        x = instance_norm_1d(wav, sd["sincnet.wav_norm1d.weight"], sd["sincnet.wav_norm1d.bias"])
        # layer 0: sinc conv -> abs -> pool -> norm -> lrelu
        x = conv1d(x, self.sinc_kernel, None, stride=10)  # SincNet stride=10
        x = np.abs(x)
        x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.0.weight"], sd["sincnet.norm1d.0.bias"])
        x = leaky_relu(x)
        # layer 1
        x = conv1d(x, sd["sincnet.conv1d.1.weight"], sd["sincnet.conv1d.1.bias"])
        x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.1.weight"], sd["sincnet.norm1d.1.bias"])
        x = leaky_relu(x)
        # layer 2
        x = conv1d(x, sd["sincnet.conv1d.2.weight"], sd["sincnet.conv1d.2.bias"])
        x = maxpool1d(x)
        x = instance_norm_1d(x, sd["sincnet.norm1d.2.weight"], sd["sincnet.norm1d.2.bias"])
        x = leaky_relu(x)
        return x  # (B, 60, frames)

    def forward(self, wav):
        sd = self.sd
        feat = self.sincnet(wav)          # (B,60,T)
        assert feat.shape[0] == 1
        x = feat[0].T                     # (T,60)
        for li in range(4):
            x = bilstm_layer(x, sd, li)   # (T,256)
        # linear layers with leaky_relu
        x = leaky_relu(x @ sd["linear.0.weight"].T + sd["linear.0.bias"])
        x = leaky_relu(x @ sd["linear.1.weight"].T + sd["linear.1.bias"])
        logits = x @ sd["classifier.weight"].T + sd["classifier.bias"]  # (T,7)
        return logits[None]               # (1,T,7)

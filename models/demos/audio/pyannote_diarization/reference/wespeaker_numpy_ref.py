"""Device-independent reference of WeSpeaker ResNet34 embedding forward.

Purpose: (1) prove the exact op sequence to port to ttnn, (2) provide a
bn-folded weight set and a numpy forward that matches the torch model within
tolerance, so the ttnn port can be parity-checked op-by-op against numpy (which
is itself parity-checked against the real torch model here).

This uses torch only for conv2d (numerical parity is what we validate); the ttnn
port replaces each op with its ttnn equivalent. The value
here is the *folded weights + explicit op graph*, not avoiding torch.
"""
import numpy as np
import torch
import torch.nn.functional as F


def fold_bn_into_conv(conv_w, bn_w, bn_b, bn_rm, bn_rv, eps=1e-5):
    """Return (folded_w, folded_b) so that conv+bn == conv(folded_w)+folded_b."""
    scale = bn_w / np.sqrt(bn_rv + eps)
    folded_w = conv_w * scale[:, None, None, None]
    folded_b = bn_b - bn_rm * scale
    return folded_w.astype(np.float32), folded_b.astype(np.float32)


class WeSpeakerNumpyRef:
    """ResNet34 [3,4,6,3] + TSTP + Linear, from a bn-folded state dict."""

    BLOCKS = [3, 4, 6, 3]
    PLANES = [32, 64, 128, 256]

    def __init__(self, state_dict):
        self.sd = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in state_dict.items()}
        self._fold_all()

    def _get(self, name):
        return self.sd[name]

    def _fold_conv_bn(self, conv_prefix, bn_prefix):
        w = self._get(conv_prefix + ".weight")
        return fold_bn_into_conv(
            w,
            self._get(bn_prefix + ".weight"),
            self._get(bn_prefix + ".bias"),
            self._get(bn_prefix + ".running_mean"),
            self._get(bn_prefix + ".running_var"),
        )

    def _fold_all(self):
        self.folded = {}
        # stem
        self.folded["conv1"] = self._fold_conv_bn("resnet.conv1", "resnet.bn1")
        # layers
        for li, nblocks in enumerate(self.BLOCKS, start=1):
            for bi in range(nblocks):
                p = f"resnet.layer{li}.{bi}"
                self.folded[f"{p}.c1"] = self._fold_conv_bn(f"{p}.conv1", f"{p}.bn1")
                self.folded[f"{p}.c2"] = self._fold_conv_bn(f"{p}.conv2", f"{p}.bn2")
                # downsample present on first block of layers 2..4
                ds_w = f"{p}.shortcut.0.weight"
                if ds_w in self.sd:
                    self.folded[f"{p}.ds"] = self._fold_conv_bn(
                        f"{p}.shortcut.0", f"{p}.shortcut.1"
                    )
        self.seg_w = self._get("resnet.seg_1.weight")
        self.seg_b = self._get("resnet.seg_1.bias")

    @staticmethod
    def _conv(x, wb, stride=1, pad=1):
        w, b = wb
        xt = torch.from_numpy(x)
        wt = torch.from_numpy(w)
        bt = torch.from_numpy(b)
        y = F.conv2d(xt, wt, bt, stride=stride, padding=pad)
        return y.numpy()

    def _block(self, x, p, stride):
        identity = x
        out = np.maximum(self._conv(x, self.folded[f"{p}.c1"], stride=stride), 0.0)
        out = self._conv(out, self.folded[f"{p}.c2"], stride=1)
        if f"{p}.ds" in self.folded:
            identity = self._conv(x, self.folded[f"{p}.ds"], stride=stride, pad=0)
        out = out + identity
        return np.maximum(out, 0.0)

    def backbone_numpy(self, feats):
        """feats: (B,1,80,T) -> conv feature map (B,C,H,W) (pre-pool).

        Mirrors TTNNWeSpeakerResident.backbone so tiny-width chunks (which
        trip ttnn conv2d auto-shard asserts) can be produced on host with an
        identical result."""
        x = np.maximum(self._conv(feats, self.folded["conv1"], stride=1), 0.0)
        for li, nblocks in enumerate(self.BLOCKS, start=1):
            for bi in range(nblocks):
                stride = 2 if (bi == 0 and li > 1) else 1
                x = self._block(x, f"resnet.layer{li}.{bi}", stride)
        return x.astype(np.float32)

    def forward(self, feats):
        """feats: (batch, 1, n_mels=80, time) float32 -> (batch, 256)."""
        x = np.maximum(self._conv(feats, self.folded["conv1"], stride=1), 0.0)
        for li, nblocks in enumerate(self.BLOCKS, start=1):
            for bi in range(nblocks):
                stride = 2 if (bi == 0 and li > 1) else 1
                x = self._block(x, f"resnet.layer{li}.{bi}", stride)
        # x: (B, 256, F', T'); TSTP: stats over time of (channel*freq)
        b, c, fdim, t = x.shape
        x2 = x.reshape(b, c * fdim, t)
        mean = x2.mean(axis=2)
        std = x2.std(axis=2)
        pooled = np.concatenate([mean, std], axis=1)  # (B, 5120)
        emb = pooled @ self.seg_w.T + self.seg_b
        return emb.astype(np.float32)

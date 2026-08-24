# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""ttnn implementation of WeSpeaker ResNet34 embedding (community-1) for p150.

Op graph follows wespeaker_numpy_ref, and tests/test_ttnn_wespeaker_ondevice.py
checks this against the torch model. BN is folded into conv weights on host
(fold_bn_into_conv in wespeaker_numpy_ref). Convs run on device via ttnn.conv2d;
residual add / relu via ttnn; TSTP (mean+std over time) via ttnn reductions;
seg_1 via ttnn.linear.

Input boundary: log-mel Fbank (B,1,80,T) float (frontend stays on host).
"""
from __future__ import annotations

import numpy as np
import torch
import ttnn

from models.demos.audio.pyannote_diarization.reference.wespeaker_numpy_ref import WeSpeakerNumpyRef


class TTNNWeSpeaker:
    BLOCKS = [3, 4, 6, 3]

    def __init__(self, state_dict, device):
        self.device = device
        self.use_device_elementwise = False
        self.use_device_pool = False
        # reuse the verified bn-folding + layout from the numpy ref
        self._ref = WeSpeakerNumpyRef(state_dict)
        self.folded = self._ref.folded
        self.seg_w = self._ref.seg_w  # (256, 5120)
        self.seg_b = self._ref.seg_b  # (256,)

    def _conv(self, x_nchw, wb, stride, pad=1):
        """x_nchw: torch (B,Cin,H,W) -> torch (B,Cout,H',W') via ttnn.conv2d."""
        w, b = wb
        B, Cin, H, W = x_nchw.shape
        Cout, _, kh, kw = w.shape
        # ttnn conv2d takes NHWC flattened as (1,1,B*H*W,Cin) with batch_size=B
        x_nhwc = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, Cin)
        tx = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tw = ttnn.from_torch(torch.from_numpy(w).to(torch.bfloat16))
        tb = ttnn.from_torch(torch.from_numpy(b).reshape(1, 1, 1, Cout).to(torch.bfloat16))
        conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)
        compute_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        out = ttnn.conv2d(
            input_tensor=tx, weight_tensor=tw, bias_tensor=tb, device=self.device,
            in_channels=Cin, out_channels=Cout, batch_size=B,
            input_height=H, input_width=W, kernel_size=(kh, kw),
            stride=(stride, stride), padding=(pad, pad),
            conv_config=conv_cfg, compute_config=compute_cfg,
        )
        ot = ttnn.to_torch(out if not isinstance(out, (tuple, list)) else out[0])
        Hout = (H + 2 * pad - kh) // stride + 1
        Wout = (W + 2 * pad - kw) // stride + 1
        return ot.reshape(B, Hout, Wout, Cout).permute(0, 3, 1, 2).float()

    @staticmethod
    def _relu(x):
        return torch.relu(x)

    def _relu_dev(self, x_nchw):
        """ReLU on device (ttnn.relu), host<->device round-trip on a tiled tensor."""
        t = ttnn.from_torch(x_nchw.to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.device)
        t = ttnn.relu(t)
        return ttnn.to_torch(t).float()

    def _add_dev(self, a_nchw, b_nchw):
        """Residual add on device (ttnn.add)."""
        ta = ttnn.from_torch(a_nchw.to(torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        tb = ttnn.from_torch(b_nchw.to(torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=self.device)
        return ttnn.to_torch(ttnn.add(ta, tb)).float()

    def _block(self, x, prefix, stride):
        relu = self._relu_dev if self.use_device_elementwise else self._relu
        identity = x
        out = relu(self._conv(x, self.folded[f"{prefix}.c1"], stride))
        out = self._conv(out, self.folded[f"{prefix}.c2"], 1)
        if f"{prefix}.ds" in self.folded:
            identity = self._conv(x, self.folded[f"{prefix}.ds"], stride, pad=0)
        # align spatial dims if conv rounding differs between paths
        h = min(out.shape[2], identity.shape[2])
        w = min(out.shape[3], identity.shape[3])
        out = out[:, :, :h, :w]
        identity = identity[:, :, :h, :w]
        if self.use_device_elementwise:
            return relu(self._add_dev(out, identity))
        return relu(out + identity)

    def forward(self, feats_nchw):
        """feats_nchw: torch (B,1,80,T) -> (B,256) embedding."""
        relu = self._relu_dev if self.use_device_elementwise else self._relu
        x = relu(self._conv(feats_nchw, self.folded["conv1"], 1))
        for li, nb in enumerate(self.BLOCKS, start=1):
            for bi in range(nb):
                stride = 2 if (bi == 0 and li > 1) else 1
                x = self._block(x, f"resnet.layer{li}.{bi}", stride)
        b, c, fdim, t = x.shape
        x2 = x.reshape(b, c * fdim, t)
        if self.use_device_pool:
            return self._tstp_seg_dev(x2)
        mean = x2.mean(axis=2)
        std = x2.std(axis=2)
        pooled = torch.cat([mean, std], dim=1)  # (B,5120)
        emb = pooled @ torch.from_numpy(self.seg_w).float().T + torch.from_numpy(self.seg_b).float()
        return emb

    def _tstp_seg_dev(self, x2):
        """TSTP (mean+std over time) + seg_1 linear on device (ttnn)."""
        # x2: torch (B, 2560, T). Compute mean/std over last dim on device.
        tx = ttnn.from_torch(x2.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device)
        mean = ttnn.mean(tx, dim=2)            # (B,2560)
        # std = sqrt(mean(x^2) - mean^2)  (population std, matches numpy .std())
        sq = ttnn.multiply(tx, tx)
        msq = ttnn.mean(sq, dim=2)             # (B,2560)
        m2 = ttnn.multiply(mean, mean)
        var = ttnn.subtract(msq, m2)
        std = ttnn.sqrt(ttnn.relu(var))        # relu guards tiny negatives
        mean_t = ttnn.to_torch(mean).float().reshape(x2.shape[0], -1)
        std_t = ttnn.to_torch(std).float().reshape(x2.shape[0], -1)
        pooled = torch.cat([mean_t, std_t], dim=1)  # (B,5120)
        # seg_1 linear on device
        tp = ttnn.from_torch(pooled.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device)
        tw = ttnn.from_torch(torch.from_numpy(self.seg_w.T.copy()).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device)
        tb = ttnn.from_torch(torch.from_numpy(self.seg_b.reshape(1, -1).copy()).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device)
        emb = ttnn.linear(tp, tw, bias=tb)
        return ttnn.to_torch(emb).float()

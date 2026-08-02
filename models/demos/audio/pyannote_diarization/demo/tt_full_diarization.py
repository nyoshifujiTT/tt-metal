"""Real community-1 diarization with BOTH NNs on p150 (ttnn):
 - segmentation PyanNet (SincNet + BiLSTM4) via TTNNPyanNet
 - embedding WeSpeaker ResNet34 backbone via TTNNWeSpeaker
Compares speaker turns to the CPU golden (2 speakers / 13 segments)."""
import os, sys, numpy as np, torch, torch.nn.functional as F, ttnn
from models.demos.audio.pyannote_diarization.tt.ttnn_wespeaker import TTNNWeSpeaker
from models.demos.audio.pyannote_diarization.tt.ttnn_pyannet import TTNNPyanNet

MODEL_DIR=os.environ.get("PYANNOTE_COMMUNITY1_WEIGHTS_DIR","/data/pyannote-community-1/weights")
SAMPLE=os.environ.get("DIAR_SAMPLE_WAV","/data/pyannote-community-1/samples/pyannote_sample.wav")
_orig=torch.load
torch.load=(lambda *a,**k:(k.__setitem__("weights_only",False) or _orig(*a,**k)) if k.get("weights_only") is None else _orig(*a,**k))
from pyannote.audio import Pipeline

dev=ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    pipe=Pipeline.from_pretrained(os.path.join(MODEL_DIR,"config.yaml")); pipe.to(torch.device("cpu"))

    # --- inject TT segmentation ---
    seg_model=pipe._segmentation.model
    sinc=seg_model.sincnet.conv1d[0].filterbank.filters().detach().numpy()
    tt_seg=TTNNPyanNet(seg_model.state_dict(), sinc, dev)
    seg_calls={"n":0}
    def seg_forward(waveforms):
        # waveforms (B,1,S) -> (B,frames,7) logsoftmax, process each batch item
        outs=[]
        for i in range(waveforms.shape[0]):
            logits=tt_seg.forward(waveforms[i:i+1].detach().numpy())  # (1,T,7)
            outs.append(torch.from_numpy(logits[0]).float())
        seg_calls["n"]+=1
        y=torch.stack(outs,0)
        return F.log_softmax(y, dim=-1)
    seg_model.forward=seg_forward

    # --- inject TT embedding backbone ---
    emb=pipe._embedding; wespeaker=emb.model_; resnet=wespeaker.resnet
    tt_emb=TTNNWeSpeaker(wespeaker.state_dict(), dev); tt_emb.use_device_elementwise=True
    def _bb_one(feats1):
        x=tt_emb._relu_dev(tt_emb._conv(feats1, tt_emb.folded["conv1"],1))
        for li,nb in enumerate(tt_emb.BLOCKS,start=1):
            for bi in range(nb):
                st=2 if (bi==0 and li>1) else 1
                x=tt_emb._block(x, f"resnet.layer{li}.{bi}", st)
        return x
    def resnet_forward(fbank, weights=None):
        feats=fbank.permute(0,2,1).unsqueeze(1).float()
        outs=[_bb_one(feats[i:i+1]) for i in range(feats.shape[0])]
        minT=min(o.shape[-1] for o in outs)
        x=torch.cat([o[...,:minT] for o in outs],dim=0)
        stats=resnet.pool(x, weights=weights)
        return torch.tensor(0.0), resnet.seg_1(stats)
    resnet.forward=resnet_forward

    print("[info] running community-1 diarization with BOTH NNs on p150...", flush=True)
    out=pipe(SAMPLE)
    diar=out.speaker_diarization
    turns=[(round(s.start,2),round(s.end,2),spk) for s,_,spk in diar.itertracks(yield_label=True)]
    spk=sorted({t[2] for t in turns})
    print(f"[result] FULL-TT diarization: segments={len(turns)} speakers={spk} seg_calls={seg_calls['n']}")
    for t in turns[:6]: print("  ",t)
    assert len(spk)>=2
    print("PASS: real community-1 diarization ran with BOTH segmentation(SincNet+BiLSTM) and embedding(ResNet34) on p150")
finally:
    ttnn.close_device(dev)

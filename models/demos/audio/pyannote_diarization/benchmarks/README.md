# Corpus DER benchmark

Scores `pyannote/speaker-diarization-community-1` over a whole diarization
corpus and reports the diarization error rate, so the port can be held against
the DER pyannote publishes for this model.

This is a benchmark rather than a test: a full split is hundreds of recordings
and runs for hours. The pytest suite keeps a short fixed-size regression check
(`tests/test_diarization_corpus_ondevice.py`, 3 recordings) and this script is
what you run when you want the comparable number.

## Preparing a corpus

```
<root>/audio/<id>.wav
<root>/rttm/<id>.rttm
```

VoxConverse is the practical choice -- audio and annotations are both CC-BY
downloads. AMI is open after registration; DIHARD needs an LDC licence.

```sh
curl -LO https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip
curl -L -o voxconverse.zip https://codeload.github.com/joonson/voxconverse/zip/refs/heads/master
python3 -c 'import zipfile; [zipfile.ZipFile(z).extractall(".") for z in ("voxconverse_test_wav.zip","voxconverse.zip")]'
ln -sfn voxconverse_test_wav audio
cp -r voxconverse-master/test rttm
```

## Running

```sh
python models/demos/audio/pyannote_diarization/benchmarks/corpus_der.py \
    --corpus /path/to/voxconverse-test --split voxconverse-test
```

Add `--limit N` to score only the first N recordings (sorted by id, so a
limited run is reproducible), `--no-offload-segmentation` to run only the
embedding net on device as the service does by default, and `--output x.json`
to keep the per-recording breakdown.

## Which split

Use the one the published figure was measured on. pyannote reports VoxConverse
on the **test** split; scoring **dev** against that number lands well under it
because dev is the easier half, which reads as a pass while measuring the wrong
thing. Splits with no published figure of their own are reported without a
verdict rather than compared against a neighbour's number.

## Measured

| Split | Recordings | DER | Published | Time |
|---|---|---|---|---|
| voxconverse-test | 232 | **0.1113** | 0.112 | 8 h |
| voxconverse-dev | 216 | 0.0705 | — (no published figure) | 3.5 h |

Both with the embedding net on device on a p150. Per-recording DERs can exceed
1.0 where a recording holds only seconds of annotated speech -- the denominator
is small, so one false alarm dominates -- which is why the metric is
accumulated by speech time rather than averaged per file.

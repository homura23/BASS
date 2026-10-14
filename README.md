# CARLA

CARLA is a public audio deepfake detection pipeline. It contains the R0/R1
model framework, training and evaluation entry points, frozen-encoder cache
utilities, and support-only adaptation code.

## Encoder architecture

CARLA uses four frozen audio encoder branches during feature extraction. The
encoder wrappers and pretrained files are supplied by an external SOFIA
checkout; CARLA supplies the bridge, configuration, fusion model, density flow,
training code, and testing scripts.

| Branch | Input | Output | Role |
| --- | --- | --- | --- |
| FxEncoder++ | stereo, 44.1 kHz | 128 dimensions | timbral and music-texture representation |
| MuQ | mono, 24 kHz | 1024 dimensions | music representation |
| MERT | mono, 24 kHz | 768 dimensions | transformer music/audio representation |
| Wav2Vec2 | mono, 16 kHz | 768 dimensions | general speech/audio representation |

The four embeddings are passed to a multi-branch mixture-of-experts fusion
module and then to the CARLA density flow. The public R0/R1 models use these
four branches only.

## External encoder setup

CARLA does not vendor the external encoder source or pretrained files. The
external SOFIA checkout is expected as a sibling directory of CARLA:

~~~text
CARLA/
../sofia/
~~~

From the CARLA repository root, set:

~~~bash
export SOFIA_ROOT=../sofia
~~~

SOFIA_ROOT is the root directory used by carla/utils/sofia_bridge.py to
locate the external encoder wrappers and pretrained files. The paths below
are resolved relative to SOFIA_ROOT; they are not paths inside CARLA. If
SOFIA_ROOT is not set, the bridge uses the sibling ../sofia directory.

These paths must exist relative to SOFIA_ROOT:

~~~text
third_party/fxencoder_plusplus/fxenc_plusplus_default.pt
third_party/muq_weights/model.safetensors
third_party/muq_weights/pytorch_model.bin
third_party/mert/pytorch_model.bin
third_party/mert/MERT-v1-95M_fairseq.pt
third_party/wav2vec/pytorch_model.bin
~~~

CARLA resolves these paths through SOFIA_ROOT. The public checkpoints use the
checkpoint_repaired encoder compatibility mode. Install the external
checkout's Python dependencies and model files before inference.

Download references:

- fx: https://huggingface.co/yytung/fxencoder-plusplus/tree/main
- muq: https://huggingface.co/OpenMuQ/MuQ-MuLan-large
- mert: https://huggingface.co/m-a-p/MERT-v1-95M
- wave2: https://huggingface.co/facebook/wav2vec2-base

## Install

The exact versions validated in the CARLA environment are in
requirements.txt:

~~~bash
conda activate carla
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
~~~

The lock file records the CUDA 12.1 PyTorch and torchaudio pair used for
validation. Replace those two packages as a matched pair for another target.

## Dataset CSV

Training, evaluation, and batch inference manifests require:

~~~text
full_path,label
audio/example_real.wav,0
audio/example_fake.wav,1
~~~

An optional source column can identify a generator. Keep private audio and
manifests outside this repository. Files under data/examples/ are placeholders.

## R0 and R1

R0 means CARLA without audio RandAugment during training stage.

R1 means CARLA with RandAugment to improve robustness to audio corruptions.

Train R0:

~~~bash
scripts/train_r0.sh \
  --train-csv data/train.csv \
  --val-csv data/validation.csv \
  --out_dir runs/r0
~~~

Train R1 from R0:

~~~bash
scripts/train_r1.sh \
  --base runs/r0/carla_r0_best.pt \
  --train-csv data/real_train.csv \
  --val-csv data/real_validation.csv \
  --out-dir runs/r1
~~~

Optional resource directories can be set in a local copy of config/r1.yaml.
Do not commit those resources or their manifests.

## Evaluation

CSV evaluation:

~~~bash
scripts/test_model.sh \
  --base weights/carla_r1_final.pt \
  --csv data/test.csv \
  --out runtime_outputs/carla_r1_test.jsonl \
  --device cuda:0
~~~

One audio file:

~~~bash
scripts/test_single_audio.py \
  --checkpoint weights/carla_r1_final.pt \
  --audio data/example.wav \
  --device cuda:0
~~~

Batch audio:

~~~bash
scripts/test_batch_audio.py \
  --checkpoint weights/carla_r1_final.pt \
  --csv data/test.csv \
  --output runtime_outputs/carla_r1_batch.jsonl \
  --device cuda:0
~~~

## Support-only adaptation

~~~bash
scripts/train_lora.sh \
  --base weights/carla_r1_final.pt \
  --support data/support.csv \
  --generator example-generator \
  --out-dir runtime_outputs/carla_lora/example-generator
~~~

Validation and test splits must not be passed to this support-only trainer.

## Layout

- carla/models/: R0/R1 model, fusion, density flow, and checkpoint code
- carla/data/: CSV, audio, augmentation, and embedding-cache utilities
- carla/fewshot/: support-only LoRA training
- carla/eval/: metrics and checkpoint evaluation
- config/: portable R0/R1 configurations
- scripts/: training, testing, and cache entry points
- tests/: public contract tests
- weights/: final public R0/R1 checkpoints

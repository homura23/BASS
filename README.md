# CARLA

CARLA is a public audio deepfake detection pipeline. It contains the R0/R1
model framework, training and evaluation entry points, frozen-encoder cache
utilities, and support-only adaptation code.

Private datasets, manifests, execution logs, intermediate caches, experiment
reports, ablation variants, and paper-result tables are excluded. Only the
final public R0 and R1 base checkpoints are stored in weights/.

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

CARLA does not vendor the external encoder source or pretrained files. Place a
SOFIA checkout next to this repository as ../sofia, or set SOFIA_ROOT:

~~~bash
export SOFIA_ROOT=/path/to/sofia
~~~

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

## Install

The exact versions validated in the MONICA environment are in
requirements.txt:

~~~bash
conda activate monica
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

R0 is the clean baseline. It trains the shared fusion and density-flow model
without waveform RandAugment.

R1 uses the same four encoders, fusion topology, and density-flow contract. It
starts from R0 and applies real-only waveform RandAugment during training to
improve robustness to recording and transmission conditions. R1 adds no new
public encoder branch.

Train R0:

~~~bash
scripts/train_r0.sh \
  --train-csv /path/to/train.csv \
  --val-csv /path/to/validation.csv \
  --out_dir runs/r0
~~~

Train R1 from R0:

~~~bash
scripts/train_r1.sh \
  --base runs/r0/carla_r0_best.pt \
  --train-csv /path/to/real_train.csv \
  --val-csv /path/to/real_validation.csv \
  --out-dir runs/r1
~~~

Optional resource directories can be set in a local copy of config/r1.yaml.
Do not commit those resources or their manifests.

## Evaluation

CSV evaluation:

~~~bash
scripts/test_model.sh \
  --base weights/carla_r1_final.pt \
  --csv /path/to/test.csv \
  --out /tmp/carla_r1_test.jsonl \
  --device cuda:0
~~~

One audio file:

~~~bash
scripts/test_single_audio.py \
  --checkpoint weights/carla_r1_final.pt \
  --audio /path/to/example.wav \
  --device cuda:0
~~~

Batch audio:

~~~bash
scripts/test_batch_audio.py \
  --checkpoint weights/carla_r1_final.pt \
  --csv /path/to/test.csv \
  --output /tmp/carla_r1_batch.jsonl \
  --device cuda:0
~~~

The inference outputs are written outside the repository. Both audio scripts
use the full_path,label contract and the four external encoder branches.
Embedding caches are local runtime artifacts and should remain outside Git.

## Support-only adaptation

~~~bash
scripts/train_lora.sh \
  --base weights/carla_r1_final.pt \
  --support /path/to/support.csv \
  --generator example-generator \
  --out-dir /tmp/carla_lora/example-generator
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

Do not commit personal paths, private data, runtime outputs, caches, or logs.

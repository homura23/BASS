# CARLA / FREYA-PRO

CARLA is the public R0/R1 implementation of the FREYA-NF v1 audio deepfake detection pipeline. It contains the model framework, training code, frozen-encoder cache utilities, support-only LoRA adaptation, and evaluation entry points.

The repository intentionally excludes private datasets, checkpoints, logs, intermediate caches, experiment reports, and paper-result tables. Unpublished internal model variants are not part of the public release.

## Dependency on SOFIA

CARLA uses the official SOFIA encoders and fusion implementation as an external dependency. SOFIA source code and pretrained weights are not vendored here.

1. Clone the official repository:

   ```bash
   git clone https://github.com/homura23/SOFIA ../sofia
   export SOFIA_ROOT="$PWD/../sofia"
   ```

2. Follow the official SOFIA README to install its transitive dependencies and place the pretrained files under its `third_party/` tree.

CARLA expects these paths relative to `SOFIA_ROOT`:

- `third_party/fxencoder_plusplus/fxenc_plusplus_default.pt`
- `third_party/muq_weights/`
- `third_party/mert/`
- `third_party/wav2vec/`

The compatibility configuration defaults to the legacy SOFIA feature initialization used by the original pipeline. The repaired checkpoint mode is available separately and represents a different feature space.

See the [official SOFIA README](https://github.com/homura23/SOFIA#readme) for the authoritative dependency and weight instructions.

## Install

From the CARLA repository root:

```bash
python -m pip install -e ".[sofia,dev]"
```

Use a PyTorch installation appropriate for the target CPU/GPU and CUDA runtime.

## Dataset CSV

Training and evaluation CSV files must contain at least:

```text
full_path,label
audio/example_real.wav,0
audio/example_fake.wav,1
```

An optional `source` column identifies the generator. Keep private datasets and their manifests outside this repository. The files in `data/examples/` are placeholders only.

## R0 and R1

R0 trains the clean FREYA-NF v1 density model:

```bash
scripts/train_r0.sh \
  --train-csv $DATA_ROOT/train.csv \
  --val-csv $DATA_ROOT/validation.csv \
  --out_dir runs/r0
```

R1 uses the same model framework and real-only RandAugment during training:

```bash
scripts/train_r1.sh \
  --base runs/r0/carla_r0_best.pt \
  --train-csv $DATA_ROOT/real_train.csv \
  --val-csv $DATA_ROOT/real_validation.csv \
  --out-dir runs/r1
```

Configure optional noise and room-response directories in a local copy of `config/r1.yaml when those resources are available. Do not commit the resources or their manifests.

Evaluate a checkpoint on a CSV split:

```bash
scripts/test_model.sh \
  --base runs/r1/carla_r1_best.pt \
  --csv $DATA_ROOT/test.csv \
  --out runs/r1/test.jsonl
```

Embedding caches can be built with `scripts/build_csv_embedding_cache.py`; cache files are local runtime artifacts and are ignored by Git.

## Support-only LoRA

The public Stage-2 path adapts a frozen R0 or R1 checkpoint using support data only:

```bash
scripts/train_lora.sh \
  --base runs/r1/carla_r1_best.pt \
  --support $DATA_ROOT/support.csv \
  --generator example-generator \
  --out-dir runs/lora/example-generator
```

Validation and test splits must not be supplied to this support-only candidate trainer. The cumulative bank evaluator is available as:

```bash
python -m freya_pro.eval.bank --help
```

## Layout

- `freya_pro/models/`: R0/R1 model and checkpoint code
- `freya_pro/data/`: CSV, audio, RandAugment, and embedding-cache utilities
- `freya_pro/fewshot/`: support-only LoRA training
- `freya_pro/eval/`: metrics, single-adapter evaluation, and bank routing
- `config/`: portable public configurations
- `scripts/`: main training, testing, and cache entry points
- `tests/`: dependency-light public contract tests

No personal paths, private data, intermediate results, execution logs, or model weights should be committed.

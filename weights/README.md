# Public CARLA checkpoints

The two files in this directory are the final public base checkpoints:

- carla_r0_final.pt: clean R0 baseline.
- carla_r1_final.pt: R1 trained from R0 with real-only waveform RandAugment.

Both contain CARLA fusion and density-flow parameters plus a portable inference
configuration. The four frozen encoder branches are external:

- FxEncoder++: third_party/fxencoder_plusplus/fxenc_plusplus_default.pt
- MuQ: third_party/muq_weights/model.safetensors and pytorch_model.bin
- MERT: third_party/mert/pytorch_model.bin and MERT-v1-95M_fairseq.pt
- Wav2Vec2: third_party/wav2vec/pytorch_model.bin

These paths are relative to SOFIA_ROOT. The embedded configuration uses the
checkpoint_repaired encoder compatibility mode. Use
scripts/test_single_audio.py or scripts/test_batch_audio.py for inference.

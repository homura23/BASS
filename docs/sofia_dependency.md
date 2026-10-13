# External encoder dependency

CARLA uses encoder wrappers and pretrained resources from an external SOFIA
checkout. CARLA does not redistribute that source tree, pretrained weights, or
local caches.

Set SOFIA_ROOT to the checkout, or place it next to CARLA as ../sofia:

~~~bash
export SOFIA_ROOT=/path/to/sofia
~~~

CARLA resolves these paths relative to SOFIA_ROOT:

- FxEncoder++: third_party/fxencoder_plusplus/fxenc_plusplus_default.pt
- MuQ: third_party/muq_weights/model.safetensors and pytorch_model.bin
- MERT: third_party/mert/pytorch_model.bin and MERT-v1-95M_fairseq.pt
- Wav2Vec2: third_party/wav2vec/pytorch_model.bin

The active branches are FxEncoder++, MuQ, MERT, and Wav2Vec2. Their inputs
are stereo 44.1 kHz, mono 24 kHz, mono 24 kHz, and mono 16 kHz. Their output
sizes are 128, 1024, 768, and 768 dimensions.

The CARLA bridge loads the frozen branches, sends their embeddings to the
multi-branch fusion module, and scores the fused representation with the CARLA
density flow. R0 and R1 use the same encoder contract. R0 is clean baseline
training; R1 starts from R0 and uses real-only waveform RandAugment.

Released configurations use relative paths only. Set SOFIA_ROOT instead of
editing a CARLA config when the external checkout is elsewhere.

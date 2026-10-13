# Public CARLA checkpoints

The two files in this directory are the final public base checkpoints:

- carla_r0_final.pt: clean R0 model.
- carla_r1_final.pt: real-only RandAugment R1 model.

They contain only the CARLA fusion/flow parameters and a portable inference
configuration. The frozen SOFIA encoder weights are external dependencies and
must be installed according to the official SOFIA README:
https://github.com/homura23/SOFIA#readme

The embedded configuration uses the checkpoint_repaired encoder contract.
Use scripts/test_single_audio.py or scripts/test_batch_audio.py for inference.

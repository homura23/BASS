# SOFIA dependency contract

CARLA treats SOFIA as an external checkout. Set SOFIA_ROOT to the checkout, or place the checkout next to the CARLA directory as `../sofia`.

The public CARLA configuration never stores a machine-specific SOFIA path. The bridge resolves the following values relative to `SOFIA_ROOT`:

- FXEncoder++ repository and default checkpoint
- MuQ repository and weight directory
- MERT checkpoint directory
- wav2vec checkpoint directory

The exact repositories, weight files, and installation steps are maintained by the official SOFIA project:

https://github.com/homura23/SOFIA#readme

CARLA does not copy SOFIA source files, checkpoints, or local dependency caches. This keeps the repository small and prevents external model artifacts from being redistributed accidentally.

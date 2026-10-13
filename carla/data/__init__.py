"""CARLA data interfaces.

The lightweight corruption/QC modules remain importable without an installed
audio backend, which lets registry and tensor-only tests run in minimal
environments.  Audio loaders are imported lazily because they require
torchaudio and the external SOFIA package.
"""

from .corruptions import AudioRandAugment, apply_operation, core_condition_registry
from .audio_io import load_audio_cached, resolve_audio_path

__all__ = [
    "AudioRandAugment",
    "PairedEmbeddingDataset",
    "RandAugmentWaveformDataset",
    "apply_operation",
    "core_condition_registry",
    "load_audio_cached",
    "make_pair_loader",
    "make_randaug_loader",
    "resolve_audio_path",
]


def __getattr__(name):
    if name in {
        "PairedEmbeddingDataset",
        "RandAugmentWaveformDataset",
        "make_pair_loader",
        "make_randaug_loader",
    }:
        from .dataset import (
            PairedEmbeddingDataset,
            RandAugmentWaveformDataset,
            make_pair_loader,
            make_randaug_loader,
        )

        return {
            "PairedEmbeddingDataset": PairedEmbeddingDataset,
            "RandAugmentWaveformDataset": RandAugmentWaveformDataset,
            "make_pair_loader": make_pair_loader,
            "make_randaug_loader": make_randaug_loader,
        }[name]
    raise AttributeError(name)

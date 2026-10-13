from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from freya_pro.models.robustness import FusionWrapper


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, embeddings):
        return self.projection(embeddings["x"])


@pytest.mark.parametrize("version", ["r0", "r1"])
def test_only_public_model_versions_are_supported(version):
    model = FusionWrapper(TinyFusion(), version=version)
    output = model({"x": torch.ones(2, 2)})
    assert model.variant == version
    assert output.shape == (2, 2)


def test_removed_model_versions_are_rejected():
    with pytest.raises(ValueError):
        FusionWrapper(TinyFusion(), version="r2")


def test_public_configs_are_relative():
    for path in sorted((PROJECT_ROOT / "config").rglob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))

        def visit(item):
            if isinstance(item, dict):
                for child in item.values():
                    yield from visit(child)
            elif isinstance(item, list):
                for child in item:
                    yield from visit(child)
            elif isinstance(item, str):
                yield item

        values = list(visit(value))
        assert not any(item.startswith("/") for item in values), path
        assert not any("25AW30" in item for item in values), path


def test_source_has_no_machine_specific_absolute_paths():
    candidates = [
        path for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts
    ]
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        machine_path = "/" + "25AW30"
        assert machine_path not in text, path
        root_cache = "/" + "root/"
        assert root_cache not in text, path

"""Convert an existing FREYA-NF v1 checkpoint to a CARLA R0 checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

from freya_pro.models.loader import load_base_v1, save_pro_checkpoint
from freya_pro.utils.config import load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default="config/r0.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["model_version"] = "r0"
    model = load_base_v1(args.source, cfg, resolve_device(args.device), "r0")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_pro_checkpoint(str(out), model, cfg, stage="migrated_base", epoch=0)
    print(f"migrated={out}")


if __name__ == "__main__":
    main()

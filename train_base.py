"""Train the clean CARLA R0 density model."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from carla.data.dataset import make_audio_loader
from carla.data.embedding_cache import make_embedding_cache_loader
from carla.data.embedding_cache import embedding_cache_artifact_sha256
from carla.eval.metrics import binary_metrics
from carla.models.loader import build_model, load_carla_checkpoint, save_carla_checkpoint
from carla.utils.config import load_config, resolve_device
from carla.utils.run_io import (
    append_jsonl, capture_rng_state, environment_manifest, public_path, restore_rng_state,
    prune_epoch_checkpoints, sha256_file, start_telemetry, telemetry_snapshot,
    utc_now, write_json_atomic,
)
from carla.utils.sofia_bridge import build_encoders, encoder_device, set_requires_grad


@torch.no_grad()
def encode(encoders, batch, device):
    if "embeddings" in batch:
        return {name: value.to(device, non_blocking=True) for name, value in batch["embeddings"].items()}
    return {
        name: encoder(
            batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
            sample_rate=encoder.target_sr,
        ).to(device)
        for name, encoder in encoders.items()
    }


def run_epoch(model, encoders, loader, optimizer, device, train: bool, max_grad_norm: float):
    model.fusion.train(train)
    model.flow.train(train)
    total = count = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            labels = batch["label"].to(device)
            embeddings = encode(encoders, batch, device)
            fused = model.base_fuse(embeddings)
            _, nll = model.flow(fused, labels)
            loss = nll.mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.parameters()) + list(model.flow.parameters()),
                    max_grad_norm,
                )
                optimizer.step()
            total += loss.item() * labels.shape[0]
            count += labels.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate_detection(model, encoders, loader, device):
    scores, labels = [], []
    model.eval()
    for batch in loader:
        scores.extend(model.score(encode(encoders, batch, device)).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    if set(labels) != {0, 1}:
        raise ValueError("Selection CSV must contain both real=0 and fake=1")
    return binary_metrics(scores, labels)


def audit_clean_selection_csv(path: str) -> None:
    """Audit the legacy/post-freeze clean detection diagnostic CSV.

    Formal Stage-0 launchers intentionally do not call this path: checkpoint
    selection must remain blind to synthetic generators.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("R0 selection CSV must contain clean real=0 and fake=1")
    splits = {row.get("split", "validation").lower() for row in rows}
    if splits != {"validation"}:
        raise ValueError(f"R0 selection must be validation-only, got splits={sorted(splits)}")
    for key in ("operation_id", "condition_id"):
        values = {row.get(key, "clean") for row in rows if row.get(key)}
        if values and values != {"clean"}:
            raise ValueError(f"R0 selection must be clean-only, got {key}={sorted(values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-duration-seconds", type=float, default=None)
    parser.add_argument("--crop-mode", choices=["random", "start", "center"], default=None)
    parser.add_argument("--eval-crop-mode", choices=["random", "start", "center"], default=None)
    parser.add_argument("--train-csv", default=None, help="Override the configured training CSV")
    parser.add_argument("--val-csv", default=None, help="Override the configured validation CSV")
    parser.add_argument("--selection-csv", default=None,
                        help="Legacy/post-freeze clean real+fake diagnostic only; "
                             "formal R0 launchers must omit it")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--resume", default=None,
                        help="Resume an epoch checkpoint from the same output directory/config")
    parser.add_argument("--train-embedding-cache", default=None)
    parser.add_argument("--val-embedding-cache", default=None)
    parser.add_argument("--selection-embedding-cache", default=None)
    parser.add_argument("--out_dir", default="runs/r0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"] = dict(cfg["data"])
    if args.train_csv is not None:
        cfg["data"]["train_csv"] = str(Path(args.train_csv).resolve())
    if args.val_csv is not None:
        cfg["data"]["val_csv"] = str(Path(args.val_csv).resolve())
    if args.max_batch_duration_seconds is not None:
        cfg["data"]["max_batch_duration_seconds"] = args.max_batch_duration_seconds
    if args.crop_mode is not None:
        cfg["data"]["crop_mode"] = args.crop_mode
    if args.eval_crop_mode is not None:
        cfg["data"]["eval_crop_mode"] = args.eval_crop_mode
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.selection_csv is not None:
        cfg["data"]["selection_csv"] = str(Path(args.selection_csv).resolve())
    for key, value in (
        ("train_embedding_cache", args.train_embedding_cache),
        ("val_embedding_cache", args.val_embedding_cache),
        ("selection_embedding_cache", args.selection_embedding_cache),
    ):
        if value is not None:
            cfg["data"][key] = str(Path(value).resolve())
    cfg["training"] = dict(cfg["training"])
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["training"]["weight_decay"] = args.weight_decay
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed
    device = resolve_device(args.device or cfg.get("device"))
    telemetry = start_telemetry(device)
    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu")
        model, checkpoint_cfg = load_carla_checkpoint(args.resume, device)
        if checkpoint_cfg != cfg:
            raise ValueError("Resume checkpoint resolved config does not match requested config")
        if resume_state.get("stage") != "base":
            raise ValueError(f"Cannot resume R0 from stage={resume_state.get('stage')!r}")
    else:
        model = build_model(cfg, device, variant=str(cfg.get("model_version", "r0")))
    data = cfg["data"]
    training = cfg["training"]
    batch_size = int(args.batch_size or data.get("batch_size", 8))
    workers = int(data.get("num_workers", 48))
    if args.train_embedding_cache and str(data.get("crop_mode", "random")) == "random":
        raise ValueError("A fixed train embedding cache cannot represent random crop across epochs")
    train_loader = (
        make_embedding_cache_loader(
            args.train_embedding_cache, csv_path=data["train_csv"], cfg=cfg,
            crop_mode=str(data.get("crop_mode", "random")), train=True,
            batch_size=batch_size, num_workers=workers,
        ) if args.train_embedding_cache else
        make_audio_loader(data["train_csv"], cfg, train=True, batch_size=batch_size, num_workers=workers)
    )
    valid_loader = (
        make_embedding_cache_loader(
            args.val_embedding_cache, csv_path=data["val_csv"], cfg=cfg,
            crop_mode=str(data.get("eval_crop_mode", "center")), train=False,
            batch_size=batch_size, num_workers=workers, drop_last=False,
        ) if args.val_embedding_cache else
        make_audio_loader(data["val_csv"], cfg, train=False, batch_size=batch_size,
                          num_workers=workers)
    )
    selection_loader = None
    if args.selection_csv:
        audit_clean_selection_csv(args.selection_csv)
        selection_loader = (
            make_embedding_cache_loader(
                args.selection_embedding_cache, csv_path=args.selection_csv, cfg=cfg,
                crop_mode=str(data.get("eval_crop_mode", "center")), train=False,
                batch_size=batch_size, num_workers=workers, drop_last=False,
            ) if args.selection_embedding_cache else
            make_audio_loader(args.selection_csv, cfg, train=False, batch_size=batch_size,
                              num_workers=workers, drop_last=False)
        )
    cache_covers_all = bool(
        args.train_embedding_cache and args.val_embedding_cache
        and (not args.selection_csv or args.selection_embedding_cache)
    )
    encoders = {} if cache_covers_all else build_encoders(
        cfg["audio_branches"], device, cfg.get("encoder_compatibility")
    )
    for encoder in encoders.values():
        set_requires_grad(encoder, False)
    parameters = list(model.fusion.parameters()) + list(model.flow.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["lr"]),
        betas=tuple(training.get("betas", (0.9, 0.98))),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    data_sha256 = {
        "train_csv": sha256_file(data["train_csv"]),
        "real_validation_csv": sha256_file(data["val_csv"]),
        "diagnostic_selection_csv": (
            sha256_file(args.selection_csv) if args.selection_csv else None
        ),
    }
    embedding_cache_sha256 = {
        "train": (
            embedding_cache_artifact_sha256(args.train_embedding_cache)
            if args.train_embedding_cache else None
        ),
        "real_validation": (
            embedding_cache_artifact_sha256(args.val_embedding_cache)
            if args.val_embedding_cache else None
        ),
        "diagnostic_selection": (
            embedding_cache_artifact_sha256(args.selection_embedding_cache)
            if args.selection_embedding_cache else None
        ),
    }
    if args.resume:
        if Path(args.resume).resolve().parent != out_dir:
            raise ValueError("Resume checkpoint must be inside --out_dir")
        if resume_state.get("optimizer") is None:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state.get("epoch", 0))
        best_valid_nll = float(
            resume_state.get("metadata", {}).get(
                "best_valid_real_nll", resume_state.get("metadata", {}).get("valid_real_nll", float("inf"))
            )
        )
        if resume_state.get("training_state", {}).get("rng"):
            restore_rng_state(resume_state["training_state"]["rng"])
    else:
        existing = list(out_dir.glob("carla_r0_epoch*.pt"))
        if existing or metrics_path.exists():
            raise FileExistsError(
                f"Output directory already contains a R0 run; use --resume: {out_dir}"
            )
        start_epoch = 0
        best_valid_nll = float("inf")
        manifest = environment_manifest()
        manifest.update({
            "stage": "r0_clean_base", "status": "running",
            "selection_policy": "musicdet_aligned_real_validation_nll",
            "formal_protocol": "strict_real_centered_r0_v1",
            "synthetic_validation_visible": bool(args.selection_csv),
            "diagnostic_only": (
                ["clean_validation_eer", "clean_validation_threshold_metrics"]
                if args.selection_csv else []
            ),
            "resolved_config": cfg,
            "data_sha256": data_sha256,
            "embedding_cache_sha256": embedding_cache_sha256,
        })
        write_json_atomic(out_dir / "run_manifest.json", manifest)
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 10))
    for epoch in range(start_epoch + 1, epochs + 1):
        train_nll = run_epoch(
            model, encoders, train_loader, optimizer, device, True,
            float(training.get("max_grad_norm", 50.0)),
        )
        valid_nll = run_epoch(
            model, encoders, valid_loader, optimizer, device, False,
            float(training.get("max_grad_norm", 50.0)),
        )
        # This optional metric never controls ``improved``. Strict real-centered
        # launchers omit the loader entirely so synthetic generators are not
        # visible until the R0/R1 winner has been frozen.
        metrics = evaluate_detection(model, encoders, selection_loader, device) if selection_loader else {}
        improved = valid_nll < best_valid_nll
        if improved:
            best_valid_nll = valid_nll
        print(
            f"epoch={epoch} train_nll={train_nll:.5f} valid_nll={valid_nll:.5f}"
            + (f" clean_val_eer={metrics['eer']:.6f}" if metrics else "")
        )
        metadata = {
            "selection_csv": str(Path(args.selection_csv).resolve()) if args.selection_csv else None,
            "selection_metrics": metrics, "valid_real_nll": valid_nll,
            "best_valid_real_nll": best_valid_nll,
            "selection_policy": "musicdet_aligned_real_validation_nll",
            "seed": seed, "optimizer": {"lr": training["lr"],
            "betas": list(training.get("betas", (0.9, 0.98))),
            "weight_decay": training.get("weight_decay", 1e-4)},
            "data_sha256": data_sha256,
            "embedding_cache_sha256": embedding_cache_sha256,
        }
        log_row = {
            "timestamp_utc": utc_now(), "epoch": epoch, "train_nll": train_nll,
            "valid_real_nll": valid_nll, "best_valid_real_nll": best_valid_nll,
            "is_best": improved, "clean_validation_metrics_diagnostic": metrics,
            "runtime": telemetry_snapshot(telemetry, device),
        }
        append_jsonl(metrics_path, log_row)
        training_state = {"rng": capture_rng_state()}
        save_carla_checkpoint(
            str(out_dir / f"carla_r0_epoch{epoch}.pt"),
            model, cfg, stage="base", epoch=epoch, metadata=metadata,
            optimizer=optimizer, training_state=training_state,
        )
        if improved:
            save_carla_checkpoint(
                str(out_dir / "carla_r0_best.pt"),
                model, cfg, stage="base", epoch=epoch, metadata=metadata,
            )
        prune_epoch_checkpoints(out_dir, "carla_r0_epoch*.pt", keep=1)
    manifest_path = out_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "stage": "r0_clean_base", "status": "complete",
        "completed_utc": utc_now(), "epochs_completed": max(start_epoch, epochs),
        "best_valid_real_nll": best_valid_nll,
        "selection_policy": "musicdet_aligned_real_validation_nll",
        "runtime": telemetry_snapshot(telemetry, device),
        "checkpoint_policy": "best_slim_plus_latest_resume",
        "resolved_config": cfg,
        "data_sha256": data_sha256,
        "completion_environment": environment_manifest(),
    })
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train CARLA R1 with real-only RandAugment.

R1 updates the shared fusion and density flow using clean and randomly
corrupted real audio. The model topology is the same as R0.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from carla.data.corruptions import AudioRandAugment
from carla.data.dataset import make_pair_loader, make_randaug_loader
from carla.data.embedding_cache import validate_stage1_randaugment_cache_metadata
from carla.models.loader import load_base_v1, load_carla_checkpoint, save_carla_checkpoint
from carla.models.lora import checkpoint_sha256
from carla.utils.config import load_config, resolve_device
from carla.utils.run_io import (
    append_jsonl, capture_rng_state, environment_manifest, public_path, restore_rng_state,
    prune_epoch_checkpoints, sha256_file, start_telemetry, telemetry_snapshot,
    utc_now, write_json_atomic,
)
from carla.utils.sofia_bridge import build_encoders, encoder_device


@torch.no_grad()
def encode_waveform_pair(
    encoders, clean: torch.Tensor, corrupt: torch.Tensor, sample_rate: int, device
):
    """Encode clean+corrupt in one call per frozen branch, then split."""
    clean_count = int(clean.shape[0])
    pair = torch.cat((clean, corrupt), dim=0)
    clean_output = {}
    corrupt_output = {}
    for name, encoder in encoders.items():
        value = encoder(
            pair.to(encoder_device(encoder), non_blocking=True),
            sample_rate=int(sample_rate),
        ).to(device)
        clean_output[name] = value[:clean_count]
        corrupt_output[name] = value[clean_count:]
    return clean_output, corrupt_output


def run_epoch(
    model, encoders, loader, optimizer, device, aug_weight: float, train: bool,
    max_grad_norm: float, *, phase: str, epoch: int, log_every: int,
):
    model.train(train)
    total = samples = 0
    previous_batch_end = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        data_seconds = time.perf_counter() - previous_batch_end
        compute_started = time.perf_counter()
        if "clean" in batch:
            clean_embeddings = {name: value.to(device, non_blocking=True) for name, value in batch["clean"].items()}
            corrupt_embeddings = {name: value.to(device, non_blocking=True) for name, value in batch["corrupt"].items()}
        else:
            clean_embeddings, corrupt_embeddings = encode_waveform_pair(
                encoders, batch["clean_waveform"], batch["corrupt_waveform"],
                int(batch["sample_rate"]), device,
            )
        labels = torch.zeros(next(iter(clean_embeddings.values())).shape[0], dtype=torch.long, device=device)
        with torch.set_grad_enabled(train):
            clean_fused = model.base_fuse(clean_embeddings)
            corrupt_fused = model.base_fuse(corrupt_embeddings)
            _, clean_nll = model.flow(clean_fused, labels)
            _, corrupt_nll = model.flow(corrupt_fused, labels)
            loss = clean_nll.mean() + float(aug_weight) * corrupt_nll.mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.fusion.parameters()) + list(model.flow.parameters()), max_grad_norm
                )
                optimizer.step()
        count = int(labels.shape[0])
        total += float(loss.detach().item()) * count
        samples += count
        compute_seconds = time.perf_counter() - compute_started
        if batch_index % max(int(log_every), 1) == 0:
            operations = sorted(
                operation.get("operation", "")
                for metadata in batch.get("metadata", [])
                for operation in metadata.get("operations", [])
                if operation.get("applied", False)
            )
            print(
                f"epoch={epoch} phase={phase} batch={batch_index}/{len(loader)} "
                f"samples={samples} running_loss={total / samples:.6f} "
                f"data_seconds={data_seconds:.3f} compute_seconds={compute_seconds:.3f} "
                f"operations={','.join(operations)}",
                flush=True,
            )
        previous_batch_end = time.perf_counter()
    if samples == 0:
        raise ValueError("RandAugment loader produced zero samples")
    return total / samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True, help="CARLA R0/R1 checkpoint")
    parser.add_argument("--train-csv", required=True, help="Real-only training CSV")
    parser.add_argument("--val-csv", required=True, help="Real-only validation CSV")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--aug-weight", type=float, default=None)
    parser.add_argument("--crop-mode", choices=["random", "start", "center"], default=None)
    parser.add_argument("--segment-seconds", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--train-pair-cache-root", default=None)
    parser.add_argument("--validation-pair-cache", default=None)
    parser.add_argument("--out-dir", default="runs/r1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_seed = int(cfg.get("seed", 42) if args.seed is None else args.seed)
    cfg["seed"] = run_seed
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
    device = resolve_device(args.device or cfg.get("device"))
    telemetry = start_telemetry(device)
    cfg["data"] = dict(cfg.get("data", {}))
    cfg["data"].update({"train_csv": str(Path(args.train_csv).resolve()),
                        "val_csv": str(Path(args.val_csv).resolve())})
    cfg["robustness"] = dict(cfg.get("robustness", {}))
    cfg["robustness"]["randaugment"] = dict(cfg["robustness"].get("randaugment", {}))
    ra_cfg = cfg["robustness"]["randaugment"]
    if args.epochs is not None: ra_cfg["base_epochs"] = args.epochs
    if args.batch_size is not None: ra_cfg["batch_size"] = args.batch_size
    if args.validation_batch_size is not None:
        ra_cfg["validation_batch_size"] = args.validation_batch_size
    if args.num_workers is not None: ra_cfg["num_workers"] = args.num_workers
    if args.aug_weight is not None: ra_cfg["base_aug_weight"] = args.aug_weight
    if args.crop_mode is not None: ra_cfg["crop_mode"] = args.crop_mode
    if args.segment_seconds is not None: ra_cfg["max_batch_duration_seconds"] = args.segment_seconds
    if bool(args.train_pair_cache_root) != bool(args.validation_pair_cache):
        raise ValueError("Both pair-cache arguments must be supplied together")
    if args.train_pair_cache_root:
        ra_cfg["train_pair_cache_root"] = str(Path(args.train_pair_cache_root).resolve())
        ra_cfg["validation_pair_cache"] = str(Path(args.validation_pair_cache).resolve())
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_state = torch.load(args.resume, map_location="cpu") if args.resume else None
    if args.resume:
        model, checkpoint_cfg = load_carla_checkpoint(args.resume, device)
        if checkpoint_cfg != cfg or resume_state.get("stage") != "base_randaug":
            raise ValueError("Resume checkpoint config/stage mismatch")
        if Path(args.resume).resolve().parent != output_dir:
            raise ValueError("Resume checkpoint must be inside --out-dir")
        if resume_state.get("metadata", {}).get("base_checkpoint_sha256") != checkpoint_sha256(args.base):
            raise ValueError("Resume checkpoint base identity mismatch")
    else:
        model = load_base_v1(args.base, cfg, device, str(cfg.get("model_version", "r1")))
    model.fusion.train()
    model.flow.train()
    encoders = None if args.train_pair_cache_root else build_encoders(
        cfg["audio_branches"], device, cfg.get("encoder_compatibility")
    )

    data_cfg = cfg.get("data", {})
    crop_mode = args.crop_mode or ra_cfg.get(
        "crop_mode", data_cfg.get("crop_mode", "random")
    )
    input_seconds = (
        args.segment_seconds
        if args.segment_seconds is not None
        else ra_cfg.get(
            "max_batch_duration_seconds",
            data_cfg.get("max_batch_duration_seconds", 60.0),
        )
    )
    augment = AudioRandAugment(
        num_layers=int(ra_cfg.get("num_layers", 1)),
        magnitude=(float(ra_cfg["magnitude"]) if ra_cfg.get("magnitude") is not None else None),
        apply_probability=float(ra_cfg.get("apply_probability", 0.9)),
        family_balanced=bool(ra_cfg.get("family_balanced", False)),
        allow_length_change=bool(ra_cfg.get("allow_length_change", True)),
        seed=run_seed,
    )

    resources_cfg = cfg.get("robustness", {}).get("resources", {})
    noise_paths = sorted(str(path) for path in Path(str(resources_cfg.get("demand_dir", ""))).rglob("*.wav"))
    rir_paths = sorted(str(path) for path in Path(str(resources_cfg.get("rir_dir", ""))).rglob("*.wav"))
    static_resources = {}

    def make_loader(csv_path: str, train: bool):
        return make_randaug_loader(
            csv_path,
            augment,
            target_sample_rate=int(
                ra_cfg.get("sample_rate", data_cfg.get("base_sample_rate", 44100))
            ),
            target_seconds=float(input_seconds),
            seed=run_seed,
            batch_size=int(
                (args.batch_size or ra_cfg.get("batch_size", 8))
                if train else
                (args.validation_batch_size or ra_cfg.get(
                    "validation_batch_size", args.batch_size or ra_cfg.get("batch_size", 8)
                ))
            ),
            # Neural codec models must stay in the parent process; spawned
            # workers would each allocate a duplicate model/GPU context.
            num_workers=int(
                ra_cfg.get("num_workers", 12)
                if train else ra_cfg.get("validation_num_workers", 4)
            ),
            train=train,
            require_real=True,
            auto_convert_mp3=bool(data_cfg.get("auto_convert_mp3", True)),
            audio_cache_dir=data_cfg.get("audio_cache_dir"),
            crop_mode=str(crop_mode if train else ra_cfg.get("eval_crop_mode", "center")),
            normalize_clean=bool(data_cfg.get("normalize", True)),
            noise_paths=noise_paths, rir_paths=rir_paths,
            static_resources=static_resources,
            apply_to_model_window=bool(ra_cfg.get("apply_to_model_window", True)),
        )

    train_loader = None if args.train_pair_cache_root else make_loader(args.train_csv, True)
    if args.train_pair_cache_root:
        validation_signature = validate_stage1_randaugment_cache_metadata(
            args.validation_pair_cache, csv_path=args.val_csv, epoch=0,
            seed=run_seed, split="validation",
        )
        val_loader = make_pair_loader(
            args.validation_pair_cache,
            int(args.validation_batch_size or ra_cfg.get("validation_batch_size", 32)),
            int(ra_cfg.get("validation_num_workers", 4)), train=False,
            expected_signature=validation_signature,
        )
    else:
        val_loader = make_loader(args.val_csv, False)
    parameters = list(model.fusion.parameters()) + list(model.flow.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(ra_cfg.get("base_lr", 5e-4)),
        betas=tuple(ra_cfg.get("betas", cfg.get("training", {}).get("betas", (0.9, 0.98)))),
        weight_decay=float(ra_cfg.get("weight_decay", 1e-4)),
    )
    aug_weight = float(
        args.aug_weight
        if args.aug_weight is not None
        else ra_cfg.get("base_aug_weight", 1.0)
    )
    epochs = int(args.epochs if args.epochs is not None else ra_cfg.get("base_epochs", 10))
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume:
        if resume_state.get("optimizer") is None:
            raise ValueError("Resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume_state["optimizer"])
        start_epoch = int(resume_state.get("epoch", 0))
        best = float(resume_state.get("metadata", {}).get("best_validation_loss", float("inf")))
        if resume_state.get("training_state", {}).get("rng"):
            restore_rng_state(resume_state["training_state"]["rng"])
    else:
        if metrics_path.exists() or list(output_dir.glob("carla_r1_epoch*.pt")):
            raise FileExistsError(f"Existing R1 RandAugment run; use --resume: {output_dir}")
        start_epoch, best = 0, float("inf")
        manifest = environment_manifest()
        manifest.update({
            "stage": "r1_randaugment", "status": "running",
            "base_checkpoint": public_path(args.base),
            "base_checkpoint_sha256": checkpoint_sha256(args.base),
            "train_csv_sha256": sha256_file(args.train_csv),
            "validation_csv_sha256": sha256_file(args.val_csv),
            "selection_policy": "fixed_real_validation_randaugment_nll",
            "formal_protocol": "strict_real_centered_stage1_v1",
            "synthetic_validation_visible": False,
            "resolved_config": cfg,
        })
        write_json_atomic(output_dir / "run_manifest.json", manifest)

    max_grad_norm = float(ra_cfg.get("max_grad_norm", cfg.get("training", {}).get("max_grad_norm", 50.0)))
    log_every = int(ra_cfg.get("progress_interval_batches", 10))

    for epoch in range(start_epoch + 1, epochs + 1):
        if args.train_pair_cache_root:
            train_cache = Path(args.train_pair_cache_root) / f"epoch_{epoch:03d}.pt"
            train_signature = validate_stage1_randaugment_cache_metadata(
                train_cache, csv_path=args.train_csv, epoch=epoch,
                seed=run_seed, split="train",
            )
            epoch_train_loader = make_pair_loader(
                str(train_cache), int(args.batch_size or ra_cfg.get("batch_size", 16)),
                int(ra_cfg.get("num_workers", 12)), train=True,
                shuffle_seed=run_seed * 1000 + epoch,
                expected_signature=train_signature,
            )
        else:
            epoch_train_loader = train_loader
            if hasattr(epoch_train_loader.dataset, "set_epoch"):
                epoch_train_loader.dataset.set_epoch(epoch)
        train_loss = run_epoch(
            model, encoders, epoch_train_loader, optimizer, device, aug_weight, True,
            max_grad_norm, phase="train", epoch=epoch, log_every=log_every,
        )
        val_loss = run_epoch(
            model, encoders, val_loader, optimizer, device, aug_weight, False,
            max_grad_norm, phase="validation", epoch=epoch, log_every=log_every,
        )
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        improved = val_loss < best
        if improved:
            best = val_loss
        metadata = {
            "seed": run_seed, "selection_split": public_path(args.val_csv),
            "train_loss": train_loss, "validation_loss": val_loss,
            "best_validation_loss": best, "selection_policy": "fixed_real_validation_randaugment_nll",
            "base_checkpoint_sha256": checkpoint_sha256(args.base),
        }
        append_jsonl(metrics_path, {"timestamp_utc": utc_now(), "epoch": epoch,
                     "train_loss": train_loss, "validation_loss": val_loss,
                     "best_validation_loss": best, "is_best": improved,
                     "runtime": telemetry_snapshot(telemetry, device)})
        training_state = {"rng": capture_rng_state()}
        save_carla_checkpoint(
            str(output_dir / f"carla_r1_epoch{epoch}.pt"),
            model,
            cfg,
            stage="base_randaug",
            epoch=epoch,
            metadata=metadata, optimizer=optimizer, training_state=training_state,
        )
        if improved:
            save_carla_checkpoint(
                str(output_dir / "carla_r1_best.pt"),
                model,
                cfg,
                stage="base_randaug",
                epoch=epoch,
                metadata=metadata,
            )
        prune_epoch_checkpoints(output_dir, "carla_r1_epoch*.pt", keep=1)
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"stage": "r1_randaugment", "status": "complete",
                     "completed_utc": utc_now(), "epochs_completed": max(start_epoch, epochs),
                     "best_validation_loss": best, "resolved_config": cfg,
                     "base_checkpoint_sha256": checkpoint_sha256(args.base),
                     "runtime": telemetry_snapshot(telemetry, device),
                     "checkpoint_policy": "best_slim_plus_latest_resume",
                     "completion_environment": environment_manifest()})
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()

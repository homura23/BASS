"""Train Stage-2 LoRA candidates from support only; never read validation/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .finetune import (
    audit_csv, cache_embeddings, encode, model_scores, seed_everything,
    support_score_normalization, weighted_nll,
)
from ..data.corruptions import AudioRandAugment, default_operation_specs
from ..data.dataset import make_audio_loader, make_randaug_loader
from ..models.loader import load_carla_checkpoint
from ..models.lora import checkpoint_sha256, inject_lora, lora_state_dict
from ..utils.config import load_config, resolve_device
from ..utils.run_io import (
    public_path, sha256_file, start_telemetry, telemetry_snapshot, torch_save_atomic,
    write_json_atomic,
)
from ..utils.sofia_bridge import build_encoders, encoder_device, set_requires_grad
from ..ablation.training_contract import validate_training_invocation
from .strict import load_frozen_recipe, recipe_identity, validate_target_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--save-steps",
        default=None,
        help="Optional comma-separated exact steps to save, in addition to max-steps.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help=(
            "Fixed model-input window for online RandAugment support training. "
            "When omitted, the RandAugment max_batch_duration_seconds config is used."
        ),
    )
    parser.add_argument(
        "--cache-support-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute frozen encoder embeddings once and reuse them during "
            "LoRA optimization. This removes repeated audio encoding."
        ),
    )
    parser.add_argument(
        "--support-crop-mode",
        choices=("start", "center"),
        default="start",
        help="Crop mode for the cached support features.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--lora-targets",
        default=None,
        help="Optional comma-separated LoRA targets, e.g. experts,projectors.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--support-augmentation", choices=("clean", "randaugment"), default="clean",
        help="Use only clean support or clean plus online views of the same labeled tracks.",
    )
    parser.add_argument("--randaugment-num-layers", type=int, default=1)
    parser.add_argument("--randaugment-apply-probability", type=float, default=0.9)
    parser.add_argument(
        "--robustness-config",
        default=str(Path(__file__).resolve().parents[2] / "config/robustness/robustness_config.yaml"),
    )
    parser.add_argument("--strict-recipe", default=None, help="Frozen development-only recipe")
    parser.add_argument("--arm", default=None, help="R0/R1 arm id required by --strict-recipe")
    parser.add_argument("--shots", type=int, default=None, help="N required by --strict-recipe")
    parser.add_argument("--ablation-manifest", default=None, help="R1-only ablation plan; support-only training contract")
    parser.add_argument("--ablation-cell", default=None, help="R1-only ablation cell id")
    parser.add_argument("--ablation-variant", default=None, help="R1-only ablation variant")
    args = parser.parse_args()
    if args.ablation_manifest and args.strict_recipe:
        raise ValueError("--ablation-manifest cannot be combined with --strict-recipe")
    if args.ablation_manifest and (not args.ablation_cell or not args.ablation_variant):
        raise ValueError("--ablation-manifest requires --ablation-cell and --ablation-variant")
    if args.max_steps <= 0 or args.save_every <= 0:
        raise ValueError("max-steps and save-every must be positive")
    if args.support_augmentation == "randaugment" and args.cache_support_embeddings:
        raise ValueError("RandAugment requires --no-cache-support-embeddings")
    save_steps = set()
    if args.save_steps:
        try:
            save_steps = {int(value) for value in args.save_steps.split(",") if value}
        except ValueError as exc:
            raise ValueError("--save-steps must be comma-separated integers") from exc
        if not save_steps or min(save_steps) <= 0:
            raise ValueError("--save-steps must contain positive integers")
        if max(save_steps) > args.max_steps:
            raise ValueError("--save-steps cannot exceed --max-steps")
    output_dir = Path(args.out_dir)
    complete_path = output_dir / "training_complete.json"
    if complete_path.is_file():
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if args.strict_recipe:
            frozen = load_frozen_recipe(args.strict_recipe)
            expected_recipe = recipe_identity(args.strict_recipe)
            if (
                completed.get("training_protocol")
                != "strict_target_support_only_development_frozen_recipe"
                or completed.get("adaptation_recipe") != expected_recipe
                or completed.get("arm") != args.arm
                or int(completed.get("shots", -1)) != int(args.shots or -1)
                or completed.get("generator") != args.generator
                or completed.get("config", {}).get("support_augmentation", "clean")
                    != frozen.get("support_augmentation", "clean")
                or args.generator not in frozen["target_generators"]
            ):
                raise ValueError("Existing completion marker does not match strict target run")
        if args.ablation_manifest:
            ablation = completed.get("ablation", {})
            if (completed.get("ablation_protocol") != "r1_only_support_only"
                    or ablation.get("cell_id") != args.ablation_cell
                    or ablation.get("variant") != args.ablation_variant
                    or ablation.get("generator") != args.generator):
                raise ValueError("Existing completion marker does not match R1-only ablation run")
        print(json.dumps({"status": "complete", "out": str(complete_path)}))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("candidate_step*.pt"))
    if existing:
        raise FileExistsError(
            f"Partial candidate run exists without completion marker: {output_dir}"
        )

    r1_only_contract = None
    seed_everything(args.seed)
    support_rows = audit_csv(args.support, generator=args.generator)
    if args.ablation_manifest:
        if len(support_rows) != 100 or sum(int(row["label"]) == 0 for row in support_rows) != 50 or sum(int(row["label"]) == 1 for row in support_rows) != 50:
            raise ValueError("R1-only support CSV must contain exactly 50 real and 50 fake rows")
    device = resolve_device(args.device)
    telemetry = start_telemetry(device)
    model, cfg = load_carla_checkpoint(args.base, device)
    lora_targets = tuple(cfg["fewshot"]["lora"]["targets"])
    if args.lora_targets:
        lora_targets = tuple(
            target.strip() for target in args.lora_targets.split(",") if target.strip()
        )
        if not lora_targets or any(
            target not in {"experts", "projectors"} for target in lora_targets
        ):
            raise ValueError("--lora-targets must contain experts and/or projectors")
    strict_recipe = None
    if args.ablation_manifest:
        r1_only_contract = validate_training_invocation(
            Path(args.ablation_manifest), args.ablation_cell, args.ablation_variant,
            base=Path(args.base), generator=args.generator, rank=args.rank,
            alpha=args.alpha, targets=list(lora_targets), lr=args.lr,
            weight_decay=args.weight_decay, steps=args.max_steps,
        )
    if args.strict_recipe:
        if args.arm is None or args.shots is None:
            raise ValueError("--strict-recipe requires --arm and --shots")
        strict_recipe = load_frozen_recipe(args.strict_recipe)
        validate_target_run(
            strict_recipe, generator=args.generator, arm=args.arm, shots=args.shots,
            rank=args.rank, alpha=args.alpha, targets=list(lora_targets), lr=args.lr,
            weight_decay=args.weight_decay, steps=args.max_steps, support_rows=support_rows,
            support_augmentation=args.support_augmentation,
            randaugment_num_layers=args.randaugment_num_layers,
            randaugment_apply_probability=args.randaugment_apply_probability,
        )
    model.eval()
    model.freeze_base()
    for parameter in model.robust_fusion.robustness_parameters():
        parameter.requires_grad_(False)
    lora_cfg = inject_lora(
        model.fusion, targets=lora_targets,
        rank=args.rank, alpha=args.alpha,
        dropout=float(cfg["fewshot"]["lora"].get("dropout", 0.0)),
    )
    set_requires_grad(model.fusion, False)
    for name, parameter in model.fusion.named_parameters():
        if "lora" in name:
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("LoRA injection produced no trainable parameters")

    encoders = build_encoders(
        cfg["audio_branches"], device, cfg.get("encoder_compatibility")
    )
    # Worker count is for the one-time audio feature cache only. Keep it
    # proportional to the number of batches; dozens of workers for N5/N10
    # merely create processes that compete for the same tiny workload.
    cache_workers = min(
        max(int(args.num_workers), 0),
        max(len(support_rows) // max(args.batch_size, 1), 0),
    )
    feature_cfg = dict(cfg)
    feature_cfg["data"] = dict(cfg["data"])
    feature_cfg["data"]["eval_crop_mode"] = args.support_crop_mode
    support_eval_loader = make_audio_loader(
        args.support, feature_cfg, train=False, batch_size=args.batch_size,
        num_workers=cache_workers, drop_last=False,
    )
    support_embeddings, support_labels = cache_embeddings(
        encoders, support_eval_loader, device
    )
    branch_names = tuple(support_embeddings)
    print(
        f"support_rows={support_labels.numel()} batch_size={args.batch_size} "
        f"cache_workers={cache_workers} "
        f"support_crop_mode={args.support_crop_mode} "
        f"training_features={'cached_frozen_encoder' if args.cache_support_embeddings else 'online_audio'}",
        flush=True,
    )
    randaugment_window_seconds = None
    if args.support_augmentation == "randaugment":
        robustness_cfg = load_config(args.robustness_config)["robustness"]
        ra_cfg = robustness_cfg["randaugment"]
        randaugment_window_seconds = (
            float(args.segment_seconds)
            if args.segment_seconds is not None
            else float(
                ra_cfg.get(
                    "max_batch_duration_seconds",
                    cfg["data"].get("max_batch_duration_seconds", 60.0),
                )
            )
        )
        if randaugment_window_seconds <= 0:
            raise ValueError(
                "RandAugment support training requires a positive model window; "
                "set --segment-seconds or robustness.randaugment.max_batch_duration_seconds."
            )
        resource_cfg = robustness_cfg.get("resources", {})
        noise_root = Path(str(resource_cfg.get("demand_dir", "")))
        rir_root = Path(str(resource_cfg.get("rir_dir", "")))
        noise_paths = sorted(str(path) for path in noise_root.rglob("*.wav")) if noise_root.is_dir() else []
        rir_paths = sorted(str(path) for path in rir_root.rglob("*.wav")) if rir_root.is_dir() else []
        augment = AudioRandAugment(
            num_layers=args.randaugment_num_layers,
            apply_probability=args.randaugment_apply_probability,
            specs=default_operation_specs(),
            family_balanced=bool(ra_cfg.get("family_balanced", False)),
            allow_length_change=bool(ra_cfg.get("allow_length_change", True)),
        )
        train_loader = make_randaug_loader(
            args.support, augment,
            target_sample_rate=int(cfg["data"].get("base_sample_rate", 44100)),
            target_seconds=randaugment_window_seconds,
            seed=args.seed, batch_size=min(args.batch_size, len(support_rows)),
            num_workers=args.num_workers, train=True, require_real=False,
            auto_convert_mp3=bool(cfg["data"].get("auto_convert_mp3", True)),
            audio_cache_dir=cfg["data"].get("audio_cache_dir"),
            crop_mode=args.support_crop_mode,
            normalize_clean=bool(cfg["data"].get("normalize", True)),
            noise_paths=noise_paths, rir_paths=rir_paths,
            apply_to_model_window=True,
        )
        print(
            f"randaugment_model_window_seconds={randaugment_window_seconds:g}",
            flush=True,
        )
    elif args.cache_support_embeddings:
        # All encoders are frozen. Reusing the support evaluation features
        # avoids running MERT/MuQ/Wave2Vec/FxPP on every optimization step.
        # The selected Stage 0 base was trained with the 60s start crop.
        # Keep this feature contract deterministic while avoiding repeated
        # encoder execution during LoRA optimization.
        train_loader = DataLoader(
            TensorDataset(
                *(support_embeddings[name] for name in branch_names),
                support_labels,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_loader = make_audio_loader(
            args.support, cfg, train=True, batch_size=args.batch_size,
            num_workers=args.num_workers, drop_last=False,
        )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    candidates: list[dict] = []
    step = 0
    training_cycle = 0
    while step < args.max_steps:
        if args.support_augmentation == "randaugment":
            train_loader.dataset.set_epoch(training_cycle)
        produced_batch = False
        for batch in train_loader:
            produced_batch = True
            if args.support_augmentation == "randaugment":
                base_labels = batch["label"].to(device)
                sample_rate = int(batch["sample_rate"])
                clean_embeddings = {
                    name: encoder(
                        batch["clean_waveform"].to(encoder_device(encoder), non_blocking=True),
                        sample_rate=sample_rate,
                    ).to(device)
                    for name, encoder in encoders.items()
                }
                corrupt_embeddings = {
                    name: encoder(
                        batch["corrupt_waveform"].to(encoder_device(encoder), non_blocking=True),
                        sample_rate=sample_rate,
                    ).to(device)
                    for name, encoder in encoders.items()
                }
                labels = torch.cat((base_labels, base_labels), dim=0)
                embeddings = {
                    name: torch.cat((clean_embeddings[name], corrupt_embeddings[name]), dim=0)
                    for name in clean_embeddings
                }
            elif args.cache_support_embeddings:
                labels = batch[-1].to(device, non_blocking=True)
                embeddings = {
                    name: value.to(device, non_blocking=True)
                    for name, value in zip(branch_names, batch[:-1])
                }
            else:
                labels = batch["label"].to(device)
                embeddings = encode(encoders, batch, device)
            _, nll = model(embeddings, labels)
            loss = weighted_nll(nll, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if (
                step % args.save_every == 0
                or step in save_steps
                or step == args.max_steps
            ):
                normalization = support_score_normalization(
                    model_scores(
                        model, support_embeddings, support_labels, device
                    ),
                    support_labels,
                )
                candidate = output_dir / f"candidate_step{step:06d}.pt"
                payload = {
                    "project": "CARLA",
                    "artifact_kind": "stage2_support_only_lora_candidate",
                    "training_protocol": "support_only_no_validation_or_test",
                    "generator": args.generator,
                    "base": public_path(args.base),
                    "base_sha256": checkpoint_sha256(args.base),
                    "support": public_path(args.support),
                    "support_sha256": checkpoint_sha256(args.support),
                    "robust_variant": model.robust_fusion.variant,
                    "lora_cfg": lora_cfg,
                    "lora": {
                        name: value.detach().cpu().clone()
                        for name, value in lora_state_dict(model.fusion).items()
                    },
                    "seed": args.seed, "step": step,
                    "score_normalization": normalization,
                    "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
                    "last_train_loss": float(loss.item()),
                    "training_features": (
                        "support_eval_frozen_encoder_embeddings"
                        if args.cache_support_embeddings else
                        ("online_randaugment_audio" if args.support_augmentation == "randaugment"
                         else "online_audio_encoding")
                    ),
                    "support_crop_mode": args.support_crop_mode,
                    "support_augmentation": args.support_augmentation,
                    "randaugment": ({
                        "num_layers": args.randaugment_num_layers,
                        "apply_probability": args.randaugment_apply_probability,
                        "operation_pool": "paper_sofia_core_plus_pitch_compression_opus_v3",
                        "unique_labeled_tracks": len(support_rows),
                        "model_window_seconds": randaugment_window_seconds,
                    } if args.support_augmentation == "randaugment" else None),
                }
                if r1_only_contract is not None:
                    payload.update({
                        "ablation": r1_only_contract,
                        "ablation_protocol": "r1_only_support_only",
                        "target_validation_used_for_selection": False,
                        "target_test_used_for_selection": False,
                    })
                if strict_recipe is not None:
                    payload.update({
                        "artifact_kind": "stage2_strict_support_only_lora",
                        "training_protocol": "strict_target_support_only_development_frozen_recipe",
                        "arm": args.arm, "shots": args.shots,
                        "fixed_training_steps": args.max_steps,
                        "adaptation_recipe": recipe_identity(args.strict_recipe),
                        "support_track_ids": [row.get("track_id") or row.get("full_path") for row in support_rows],
                        "support_source_hashes": [row.get("source_sha256", "") for row in support_rows],
                    })
                torch_save_atomic(candidate, payload)
                candidates.append({
                    "step": step, "path": str(candidate.resolve()),
                    "sha256": sha256_file(candidate),
                    "train_loss": float(loss.item()),
                })
                print(
                    f"step={step}/{args.max_steps} loss={loss.item():.6f} "
                    f"saved={candidate}", flush=True,
                )
            if step >= args.max_steps:
                break
        if not produced_batch:
            raise ValueError("Support loader produced zero batches")
        training_cycle += 1
    write_json_atomic(complete_path, {
        "status": "complete", "training_protocol": "support_only",
        "base": public_path(args.base),
        "base_sha256": checkpoint_sha256(args.base),
        "support": public_path(args.support),
        "support_sha256": checkpoint_sha256(args.support),
        "generator": args.generator, "seed": args.seed,
        "max_steps": args.max_steps, "save_every": args.save_every,
        "save_steps": sorted(save_steps),
        **({"ablation": r1_only_contract, "ablation_protocol": "r1_only_support_only", "target_validation_used_for_selection": False, "target_test_used_for_selection": False} if r1_only_contract is not None else {}),
        "config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "cache_support_embeddings": args.cache_support_embeddings,
            "support_crop_mode": args.support_crop_mode,
            "support_window_seconds": randaugment_window_seconds,
            "lora_targets": list(lora_targets),
            "rank": args.rank,
            "alpha": args.alpha,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "support_augmentation": args.support_augmentation,
            "randaugment_num_layers": args.randaugment_num_layers,
            "randaugment_apply_probability": args.randaugment_apply_probability,
        },
        "candidates": candidates,
        "runtime": telemetry_snapshot(telemetry, device),
        **({
            "training_protocol": "strict_target_support_only_development_frozen_recipe",
            "arm": args.arm, "shots": args.shots,
            "adaptation_recipe": recipe_identity(args.strict_recipe),
        } if strict_recipe is not None else {}),
    })
    print(f"saved={complete_path} candidates={len(candidates)}")


if __name__ == "__main__":
    main()

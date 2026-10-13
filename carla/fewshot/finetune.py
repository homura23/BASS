"""Stage 2: validation-selected generator LoRA on a frozen CARLA checkpoint."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import make_audio_loader
from ..data.embedding_cache import (
    embedding_cache_artifact_sha256, make_embedding_cache_loader,
)
from ..eval.metrics import binary_metrics
from ..models.loader import load_carla_checkpoint
from ..models.lora import checkpoint_sha256, inject_lora, lora_state_dict
from ..utils.config import resolve_device
from ..utils.run_io import public_path, start_telemetry, telemetry_snapshot, torch_save_atomic
from ..utils.sofia_bridge import build_encoders, encoder_device, set_requires_grad


def weighted_nll(nll: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    count = labels.shape[0]
    real = (labels == 0).sum().clamp_min(1)
    fake = (labels == 1).sum().clamp_min(1)
    weight = torch.where(labels == 0, count / (2.0 * real.float()), count / (2.0 * fake.float()))
    return (weight * nll).sum() / weight.sum()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def audit_csv(path: str, *, generator: str | None = None) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(f"CSV must contain both real=0 and fake=1: {path}")
    # The benchmark's formal sample identity is track_id (full_path fallback).
    # Content-hash aliases are diagnostic-only under the documented Jamendo policy.
    identities = [row.get("track_id") or row.get("full_path") or row.get("path") for row in rows]
    if any(not value for value in identities) or len(identities) != len(set(identities)):
        raise ValueError(f"CSV contains empty or duplicate sample identities: {path}")
    if generator:
        fake_sources = {row.get("source", "") for row in rows if int(row["label"]) == 1}
        if fake_sources != {generator}:
            raise ValueError(f"Support fake source mismatch: expected {generator!r}, got {sorted(fake_sources)}")
    return rows


@torch.no_grad()
def encode(encoders, batch, device):
    if "embeddings" in batch:
        return {name: value.to(device, non_blocking=True) for name, value in batch["embeddings"].items()}
    return {
        name: encoder(batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
                      sample_rate=encoder.target_sr).to(device)
        for name, encoder in encoders.items()
    }


@torch.no_grad()
def cache_embeddings(encoders, loader, device):
    values: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    for batch in loader:
        embeddings = encode(encoders, batch, device)
        for name, value in embeddings.items():
            values.setdefault(name, []).append(value.detach().cpu())
        labels.append(batch["label"].cpu())
    if not labels:
        raise ValueError("Validation loader produced zero samples")
    return {name: torch.cat(parts) for name, parts in values.items()}, torch.cat(labels)


@torch.no_grad()
def model_scores(model, embeddings, labels, device, batch_size: int = 256) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for start in range(0, labels.numel(), batch_size):
        batch = {name: value[start:start + batch_size].to(device) for name, value in embeddings.items()}
        scores.append(model.score(batch).detach().cpu())
    return torch.cat(scores)


@torch.no_grad()
def validation_metrics(model, embeddings, labels, device, batch_size: int = 256):
    scores = model_scores(model, embeddings, labels, device, batch_size)
    return binary_metrics(scores.tolist(), labels.tolist())


def support_score_normalization(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float | str]:
    """Fit one LoRA's NLL scale from support only (never validation/test)."""
    real = scores[labels == 0]
    if real.numel() == 0:
        raise ValueError("Support normalization requires real examples")
    mean = real.mean()
    std = scores.std(unbiased=True)
    if not torch.isfinite(mean) or not torch.isfinite(std) or float(std) <= 1e-8:
        raise ValueError(f"Invalid support NLL normalization mean={mean} std={std}")
    return {
        "method": "support_real_mean_all_std_zscore",
        "mean": float(mean),
        "std": float(std),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="CARLA robust checkpoint")
    parser.add_argument("--support", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-embedding-cache", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    support_rows = audit_csv(args.support, generator=args.generator)
    validation_rows = audit_csv(args.validation, generator=args.generator)
    identity = lambda row: row.get("track_id") or row.get("full_path") or row.get("path")
    overlap = {identity(row) for row in support_rows} & {identity(row) for row in validation_rows}
    if overlap:
        raise ValueError(f"Support/validation identity overlap: {len(overlap)}")
    device = resolve_device(args.device)
    telemetry = start_telemetry(device)
    model, cfg = load_carla_checkpoint(args.base, device)
    model.eval()
    model.freeze_base()
    for parameter in model.robust_fusion.robustness_parameters():
        parameter.requires_grad_(False)
    lora_cfg = inject_lora(
        model.fusion, targets=tuple(cfg["fewshot"]["lora"]["targets"]),
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

    encoders = build_encoders(cfg["audio_branches"], device, cfg.get("encoder_compatibility"))
    train_loader = make_audio_loader(args.support, cfg, train=True,
                                     batch_size=args.batch_size, num_workers=args.num_workers,
                                     drop_last=False)
    val_loader = (
        make_embedding_cache_loader(
            args.validation_embedding_cache, csv_path=args.validation, cfg=cfg,
            crop_mode=str(cfg.get("data", {}).get("eval_crop_mode", "center")),
            train=False, batch_size=args.batch_size, num_workers=args.num_workers,
            drop_last=False,
        ) if args.validation_embedding_cache else
        make_audio_loader(args.validation, cfg, train=False,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          drop_last=False)
    )
    val_embeddings, val_labels = cache_embeddings(encoders, val_loader, device)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_key = (float("inf"), float("inf"))
    best_step = 0
    best_metrics: dict[str, float] = {}
    best_state = {name: value.detach().cpu().clone() for name, value in lora_state_dict(model.fusion).items()}
    stale = 0
    step = 0
    validation_history: list[dict[str, float | int]] = []
    while step < args.max_steps and stale < args.patience:
        for batch in train_loader:
            labels = batch["label"].to(device)
            embeddings = encode(encoders, batch, device)
            _, nll = model(embeddings, labels)
            loss = weighted_nll(nll, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.eval_every == 0 or step == args.max_steps:
                metrics = validation_metrics(model, val_embeddings, val_labels, device)
                validation_history.append({"step": step, "train_loss": float(loss.item()), **metrics})
                if metrics["eer"] is None:
                    raise ValueError("LoRA validation selection requires both real and synthetic rows for EER")
                key = (float(metrics["eer"]), -float(metrics["macro_f1"]))
                eer_text = f"{float(metrics['eer']):.6f}"
                print(f"step={step}/{args.max_steps} loss={loss.item():.6f} val_eer={eer_text} val_macro_f1={metrics['macro_f1']:.6f}")
                if key[0] < best_key[0] - args.min_delta or (abs(key[0] - best_key[0]) <= args.min_delta and key[1] < best_key[1]):
                    best_key, best_step, best_metrics, stale = key, step, metrics, 0
                    best_state = {name: value.detach().cpu().clone() for name, value in lora_state_dict(model.fusion).items()}
                else:
                    stale += 1
            if step >= args.max_steps or stale >= args.patience:
                break
        if len(train_loader) == 0:
            raise ValueError("Support loader produced zero batches")

    full_state = model.fusion.state_dict()
    full_state.update(best_state)
    model.fusion.load_state_dict(full_state, strict=True)
    support_eval_loader = make_audio_loader(
        args.support, cfg, train=False, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=False,
    )
    support_embeddings, support_labels = cache_embeddings(encoders, support_eval_loader, device)
    normalization = support_score_normalization(
        model_scores(model, support_embeddings, support_labels, device), support_labels
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite completed LoRA checkpoint: {out}")
    payload = {
        "project": "CARLA", "generator": args.generator,
        "base": public_path(args.base), "base_sha256": checkpoint_sha256(args.base),
        "support": public_path(args.support), "support_sha256": checkpoint_sha256(args.support),
        "validation": public_path(args.validation), "validation_sha256": checkpoint_sha256(args.validation),
        "validation_embedding_cache": public_path(args.validation_embedding_cache) if args.validation_embedding_cache else None,
        "validation_embedding_cache_sha256": embedding_cache_artifact_sha256(args.validation_embedding_cache) if args.validation_embedding_cache else None,
        "robust_variant": model.robust_fusion.variant, "lora_cfg": lora_cfg,
        "lora": best_state, "seed": args.seed, "best_step": best_step,
        "validation_metrics": best_metrics, "frozen_threshold": best_metrics.get("threshold"),
        "validation_history": validation_history,
        "score_normalization": normalization,
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "runtime": telemetry_snapshot(telemetry, device),
    }
    torch_save_atomic(out, payload)
    print(f"saved={out} best_step={best_step} val_eer={best_metrics.get('eer')}")


if __name__ == "__main__":
    main()

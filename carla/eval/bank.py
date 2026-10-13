"""Cumulative non-oracle LoRA bank evaluation with support-prototype routing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.dataset import make_audio_loader
from ..data.embedding_cache import (
    build_csv_embedding_cache, load_csv_embedding_cache, save_embedding_cache,
)
from ..models.loader import load_carla_checkpoint
from ..models.lora import load_lora_checkpoint, remove_lora
from ..utils.config import resolve_device
from ..utils.run_io import (
    atomic_output_path, bank_fingerprint_from_hashes, public_path, publicize_row, sha256_file,
    torch_save_atomic,
)
from ..utils.sofia_bridge import build_encoders, encoder_device


def read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


@torch.no_grad()
def encode_csv(encoders, cfg, path: str, device, batch_size: int, num_workers: int,
               cache_path: str | None = None, crop_mode: str | None = None):
    crop_mode = crop_mode or str(cfg.get("data", {}).get("eval_crop_mode", "center"))
    eval_cfg = dict(cfg)
    eval_cfg["data"] = dict(cfg.get("data", {}))
    eval_cfg["data"]["eval_crop_mode"] = crop_mode
    if cache_path and Path(cache_path).is_file():
        payload = load_csv_embedding_cache(
            cache_path, csv_path=path, cfg=eval_cfg, crop_mode=crop_mode,
        )
        return dict(payload["embeddings"]), payload["labels"].to(torch.long)
    if encoders is None:
        raise ValueError(f"Embedding cache is missing and encoders were not built: {cache_path}")
    if cache_path:
        payload = build_csv_embedding_cache(
            encoders, path, eval_cfg, device, crop_mode=crop_mode,
            batch_size=batch_size, num_workers=num_workers,
        )
        save_embedding_cache(cache_path, payload)
        return dict(payload["embeddings"]), payload["labels"].to(torch.long)
    loader = make_audio_loader(path, eval_cfg, train=False, batch_size=batch_size,
                               num_workers=num_workers, drop_last=False)
    values: dict[str, list[torch.Tensor]] = {}
    labels: list[torch.Tensor] = []
    for batch in loader:
        labels.append(batch["label"].cpu())
        for name, encoder in encoders.items():
            value = encoder(batch["audio"][name].to(encoder_device(encoder), non_blocking=True),
                            sample_rate=encoder.target_sr).to(device)
            values.setdefault(name, []).append(value.detach().cpu())
    if not labels:
        raise ValueError(f"No encodable rows: {path}")
    return {name: torch.cat(parts) for name, parts in values.items()}, torch.cat(labels)


@torch.no_grad()
def canonical_embeddings(model, embeddings, device, batch_size: int):
    output = []
    count = next(iter(embeddings.values())).shape[0]
    for start in range(0, count, batch_size):
        batch = {name: value[start:start + batch_size].to(device) for name, value in embeddings.items()}
        output.append(model.canonicalize(batch).cpu())
    return torch.cat(output)


@torch.no_grad()
def adapter_scores(model, embeddings, device, batch_size: int):
    output = []
    count = next(iter(embeddings.values())).shape[0]
    for start in range(0, count, batch_size):
        batch = {name: value[start:start + batch_size].to(device) for name, value in embeddings.items()}
        output.append(model.score(batch).cpu())
    return torch.cat(output)


def normalize_adapter_scores(scores: torch.Tensor, state: dict, path: str) -> torch.Tensor:
    calibration = state.get("score_normalization")
    if not calibration:
        raise ValueError(f"LoRA has no support-only score normalization: {path}")
    if calibration.get("method") != "support_real_mean_all_std_zscore":
        raise ValueError(f"Unknown score normalization in {path}: {calibration}")
    mean = float(calibration["mean"])
    std = float(calibration["std"])
    if not torch.isfinite(torch.tensor([mean, std])).all() or std <= 1e-8:
        raise ValueError(f"Invalid score normalization in {path}: {calibration}")
    return (scores - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--lora", action="append", required=True, help="Ordered cumulative bank member")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--normalize-nll", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-embedding-cache", default=None)
    parser.add_argument(
        "--crop-mode",
        choices=("start", "center"),
        default=None,
        help="Override the checkpoint eval crop mode for this evaluation.",
    )
    parser.add_argument("--support-embedding-cache-root", default=None)
    parser.add_argument("--router-components-cache", default=None,
                        help="Reusable eval routes, prototypes, and per-LoRA scores")
    parser.add_argument("--router-config", default=None,
                        help="Validation-selected JSON containing top_k and temperature")
    args = parser.parse_args()
    if args.router_config:
        router = json.loads(Path(args.router_config).read_text(encoding="utf-8"))
        args.top_k = int(router["top_k"])
        args.temperature = float(router["temperature"])
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    eval_rows = read_rows(args.eval_csv)
    base_sha256 = sha256_file(args.base)
    lora_sha256 = [sha256_file(path) for path in args.lora]
    deployed_fingerprint = bank_fingerprint_from_hashes(base_sha256, lora_sha256)
    component_spec = {
        "bank_fingerprint": deployed_fingerprint,
        "eval_csv_sha256": sha256_file(args.eval_csv),
        "normalize_nll": bool(args.normalize_nll),
    }
    # Preserve the legacy component-cache contract when no crop override is
    # requested; explicit start/center overrides are part of new signatures.
    if args.crop_mode is not None:
        component_spec["crop_mode"] = args.crop_mode
    component_path = (
        Path(args.router_components_cache).resolve()
        if args.router_components_cache else None
    )
    components = None
    if component_path and component_path.is_file():
        candidate = torch.load(component_path, map_location="cpu")
        if (candidate.get("project") != "CARLA"
                or candidate.get("kind") != "lora_bank_router_components"
                or candidate.get("spec") != component_spec):
            raise ValueError(f"Router component cache signature mismatch: {component_path}")
        tensors = {
            name: candidate.get(name) for name in (
                "eval_labels", "eval_route", "score_matrix", "prototype_matrix"
            )
        }
        if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
            raise ValueError(f"Router component cache has missing tensors: {component_path}")
        if any(not torch.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"Router component cache has non-finite tensors: {component_path}")
        generators_value = list(candidate.get("generators", []))
        rows_count = len(eval_rows)
        if (
            tensors["eval_labels"].shape[0] != rows_count
            or tensors["eval_route"].shape[0] != rows_count
            or tensors["score_matrix"].shape != (rows_count, len(args.lora))
            or tensors["prototype_matrix"].shape[0] != len(args.lora)
            or len(generators_value) != len(args.lora)
        ):
            raise ValueError(f"Router component cache shape mismatch: {component_path}")
        components = candidate
    if components is not None:
        eval_labels = components["eval_labels"]
        eval_route = components["eval_route"]
        score_matrix = components["score_matrix"]
        prototype_matrix = components["prototype_matrix"]
        generators = list(components["generators"])
    else:
        device = resolve_device(args.device)
        model, cfg = load_carla_checkpoint(args.base, device)
        crop_mode = args.crop_mode or str(
            cfg.get("data", {}).get("eval_crop_mode", "center")
        )
        model.eval()
        states = [torch.load(path, map_location="cpu") for path in args.lora]
        support_cache_paths: list[str | None] = []
        for state in states:
            if args.support_embedding_cache_root:
                digest = hashlib.sha256(str(state.get("support", "")).encode()).hexdigest()[:16]
                support_cache_paths.append(str(
                    Path(args.support_embedding_cache_root) / f"support_{digest}_{crop_mode}.pt"
                ))
            else:
                support_cache_paths.append(None)
        required_caches = [args.eval_embedding_cache, *support_cache_paths]
        cache_miss = any(path is None or not Path(path).is_file() for path in required_caches)
        encoders = (
            build_encoders(cfg["audio_branches"], device, cfg.get("encoder_compatibility"))
            if cache_miss else None
        )
        eval_embeddings, eval_labels = encode_csv(
            encoders, cfg, args.eval_csv, device, args.batch_size, args.num_workers,
            cache_path=args.eval_embedding_cache, crop_mode=crop_mode,
        )
        remove_lora(model.fusion)
        eval_route = F.normalize(
            canonical_embeddings(model, eval_embeddings, device, args.batch_size), dim=-1
        ).cpu()
        prototypes = []
        generators = []
        for path, state, support_cache in zip(args.lora, states, support_cache_paths):
            if not state.get("support"):
                raise ValueError(f"LoRA has no support provenance: {path}")
            support_embeddings, support_labels = encode_csv(
                encoders, cfg, str(state["support"]), device, args.batch_size, args.num_workers,
                cache_path=support_cache, crop_mode=crop_mode,
            )
            remove_lora(model.fusion)
            support_route = canonical_embeddings(model, support_embeddings, device, args.batch_size)
            fake = support_labels == 1
            if not bool(fake.any()):
                raise ValueError(f"LoRA support has no fake rows: {path}")
            prototypes.append(F.normalize(support_route[fake].mean(dim=0), dim=0))
            generators.append(str(state["generator"]))
        all_scores = []
        for path, state in zip(args.lora, states):
            load_lora_checkpoint(model, path, base_checkpoint=args.base, strict=True)
            model.eval()
            scores = adapter_scores(model, eval_embeddings, device, args.batch_size)
            if args.normalize_nll:
                scores = normalize_adapter_scores(scores, state, path)
            all_scores.append(scores)
        score_matrix = torch.stack(all_scores, dim=1).cpu()
        prototype_matrix = torch.stack(prototypes, dim=0).cpu()
        eval_labels = eval_labels.cpu()
        if component_path:
            torch_save_atomic(component_path, {
                "project": "CARLA",
                "kind": "lora_bank_router_components",
                "spec": component_spec,
                "eval_labels": eval_labels,
                "eval_route": eval_route,
                "score_matrix": score_matrix,
                "prototype_matrix": prototype_matrix,
                "generators": generators,
            })
    similarities = eval_route @ prototype_matrix.T
    top_k = min(max(args.top_k, 1), len(args.lora))
    top_values, top_indices = similarities.topk(top_k, dim=1)
    weights = torch.softmax(top_values / args.temperature, dim=1)
    selected_scores = score_matrix.gather(1, top_indices)
    bank_scores = (selected_scores * weights).sum(dim=1)

    if len(eval_rows) != len(bank_scores):
        raise ValueError("CSV row count does not match encoded score count")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(destination) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            for row, label, score, indexes, route_weights in zip(
                eval_rows, eval_labels, bank_scores, top_indices, weights
            ):
                route = [
                    {"generator": generators[int(index)], "weight": float(weight)}
                    for index, weight in zip(indexes.tolist(), route_weights.tolist())
                ]
                value = {
                    **publicize_row(row), "track_id": row.get("track_id", ""), "label": int(label),
                    "score": float(score), "route_top1": route[0]["generator"],
                    "route": route, "bank_size": len(args.lora),
                    "nll_normalized": bool(args.normalize_nll),
                    "operation_id": row.get("operation_id") or "clean",
                    "generator_id": row.get("source") or row.get("generator") or "",
                    "bank_fingerprint": deployed_fingerprint,
                    "base_checkpoint": public_path(args.base),
                    "base_checkpoint_sha256": base_sha256,
                    "top_k": top_k, "temperature": float(args.temperature), "status": "ok",
                }
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"samples={len(bank_scores)} bank_size={len(args.lora)} out={destination}")


if __name__ == "__main__":
    main()

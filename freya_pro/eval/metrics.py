"""Model-independent common-corruption and adversarial metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_FORMAL_GENERATORS = (
    "suno:v4", "suno:v4.5", "suno:v4.5+", "suno:v4.5-all",
    "suno:v5", "suno:v5.5", "mureka:v7.5", "mureka:v7.5-all",
    "mureka:v7.6", "mureka:v8", "mureka:v9",
)


def _as_arrays(
    scores: Sequence[float],
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("scores and labels must be equal-length vectors")
    if len(scores) == 0:
        raise ValueError("Cannot score an empty set")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or Inf")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be 0=real or 1=fake")
    return scores, labels


def balanced_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    scores, labels = _as_arrays(scores, labels)
    candidates = np.unique(scores)
    best = (float("inf"), float(candidates[0]))
    for threshold in candidates:
        predicted_real = scores >= threshold
        real = labels == 0
        fake = labels == 1
        fpr = float((~predicted_real[real]).mean()) if real.any() else 0.0
        fnr = float(predicted_real[fake].mean()) if fake.any() else 0.0
        value = abs(fpr - fnr)
        if value < best[0]:
            best = (value, float(threshold))
    return best[1]


def threshold_at_target_fpr(
    scores: Sequence[float], target_fpr: float,
) -> tuple[float, float]:
    """Calibrate a real-only operating point with a deterministic FPR cap.

    FREYA scores increase toward real. A track is called synthetic when
    ``score < threshold``. The threshold is selected from real validation
    scores so the empirical false-positive rate is no larger than
    ``target_fpr``; the second return value is the achieved validation FPR.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("real-only calibration scores must be a finite non-empty vector")
    target_fpr = float(target_fpr)
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be in [0, 1]")
    ordered = np.sort(values)
    allowed = min(int(np.floor(target_fpr * len(ordered))), len(ordered))
    if allowed == 0:
        threshold = float(ordered[0])
    else:
        threshold = float(np.nextafter(ordered[allowed - 1], np.inf))
    achieved = float(np.mean(values < threshold))
    return threshold, achieved


def eer(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Approximate EER; return N/A for a single-class evaluation group."""

    scores, labels = _as_arrays(scores, labels)
    real = labels == 0
    fake = labels == 1
    if not real.any() or not fake.any():
        return None
    best = (float("inf"), float("inf"))
    for threshold in np.unique(scores):
        predicted_real = scores >= threshold
        fpr = float((~predicted_real[real]).mean()) if real.any() else 0.0
        fnr = float(predicted_real[fake].mean()) if fake.any() else 0.0
        candidate = (abs(fpr - fnr), 0.5 * (fpr + fnr))
        if candidate < best:
            best = candidate
    return best[1]


def auroc_ai(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """AUROC with synthetic (label 1) as the positive class.

    FREYA scores increase toward real, so the ranking score for this metric is
    ``-score``.  A single-class group has no defined AUROC.
    """
    scores, labels = _as_arrays(scores, labels)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if not positive_count or not negative_count:
        return None
    # Count P(score_synthetic < score_real) with tie handling in one sorted
    # pass. The pairwise formulation is O(N^2) memory and is unsafe for the
    # full benchmark test corpus.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    negative_seen = 0
    favorable = 0.0
    index = 0
    while index < len(sorted_scores):
        end = index + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group = sorted_labels[index:end]
        group_positive = int((group == 1).sum())
        group_negative = int((group == 0).sum())
        favorable += group_positive * (negative_seen + 0.5 * group_negative)
        negative_seen += group_negative
        index = end
    return float(favorable / (positive_count * negative_count))


def binary_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float | None = None,
) -> dict[str, float | None]:
    """Metrics for FREYA scores where larger score means more real.

    ``label=1`` is synthetic. ``f1``/``f1_ai`` are therefore the primary
    synthetic-positive binary F1; ``macro_f1`` is retained only as a named
    supplementary statistic for compatibility with historical artifacts.
    """

    scores, labels = _as_arrays(scores, labels)
    threshold = balanced_threshold(scores, labels) if threshold is None else float(threshold)
    predicted_real = scores >= threshold
    real = labels == 0
    fake = labels == 1
    real_recall = float(predicted_real[real].mean()) if real.any() else 0.0
    fake_recall = float((~predicted_real[fake]).mean()) if fake.any() else 0.0
    fpr = 1.0 - real_recall
    fnr = 1.0 - fake_recall
    accuracy = float((predicted_real == (labels == 0)).mean())
    precision_real_den = max(int(predicted_real.sum()), 1)
    precision_real = float((predicted_real & real).sum() / precision_real_den)
    f1_real = 2.0 * precision_real * real_recall / max(precision_real + real_recall, 1e-12)
    precision_fake_den = max(int((~predicted_real).sum()), 1)
    precision_fake = float(((~predicted_real) & fake).sum() / precision_fake_den)
    f1_fake = 2.0 * precision_fake * fake_recall / max(precision_fake + fake_recall, 1e-12)
    # Synthetic-positive confusion matrix: prediction is synthetic when
    # score<threshold. This is the paper-facing F1_AI definition.
    predicted_synthetic = ~predicted_real
    tp_ai = int((predicted_synthetic & fake).sum())
    fp_ai = int((predicted_synthetic & real).sum())
    fn_ai = int((~predicted_synthetic & fake).sum())
    f1_ai = 2.0 * tp_ai / max(2 * tp_ai + fp_ai + fn_ai, 1)
    ai_auroc = auroc_ai(scores, labels)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (real_recall + fake_recall),
        "real_specificity": real_recall,
        "synthetic_recall": fake_recall,
        "fpr_real_to_fake": fpr,
        "fnr_fake_to_real": fnr,
        "f1": f1_ai,
        "f1_ai": f1_ai,
        "auroc": ai_auroc,
        "auroc_ai": ai_auroc,
        "macro_f1": 0.5 * (f1_real + f1_fake),
        "eer": eer(scores, labels),
        "count": float(len(scores)),
    }


def grouped_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("operation_id",),
    score_key: str = "score",
    label_key: str = "label",
    threshold: float | None = None,
) -> dict[str, dict[str, float | None]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = {}
    for key, group in groups.items():
        name = "::".join(str(value) for value in key)
        output[name] = binary_metrics(
            [float(row[score_key]) for row in group],
            [int(row[label_key]) for row in group],
            threshold=threshold,
        )
    return output


def aggregate_clean_test_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    formal_generators: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report pooled clean metrics separately from equal-weight generator macro.

    The pooled population may include a designated development generator such
    as ``mureka:unknown``.  The generator macro is intentionally restricted to
    the frozen formal-generator list so it remains comparable with the 11x11
    source/target comparisons.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate an empty clean test")

    def generator_name(row: Mapping[str, Any]) -> str:
        return str(
            row.get("generator")
            or row.get("source")
            or row.get("generator_name")
            or row.get("generator_id")
            or ""
        )

    pooled = binary_metrics(
        [float(row["score"]) for row in rows],
        [int(row["label"]) for row in rows],
        threshold=threshold,
    )
    real_rows = [row for row in rows if int(row["label"]) == 0]
    fake_rows = [row for row in rows if int(row["label"]) == 1]
    if not real_rows or not fake_rows:
        raise ValueError("Clean formal test requires both real and synthetic rows")
    generators = list(formal_generators or DEFAULT_FORMAL_GENERATORS)
    by_generator: dict[str, dict[str, float | None]] = {}
    for generator in generators:
        generator_rows = [row for row in fake_rows if generator_name(row) == generator]
        if not generator_rows:
            raise ValueError(f"Clean formal test is missing generator: {generator}")
        group = real_rows + generator_rows
        by_generator[generator] = binary_metrics(
            [float(row["score"]) for row in group],
            [int(row["label"]) for row in group],
            threshold=threshold,
        )
    metric_names = (
        "accuracy", "balanced_accuracy", "real_specificity", "synthetic_recall",
        "fpr_real_to_fake", "fnr_fake_to_real", "f1_ai", "macro_f1",
        "auroc_ai", "eer",
    )
    generator_macro: dict[str, Any] = {
        "generator_count": float(len(generators)),
        "threshold": float(threshold),
    }
    for name in metric_names:
        values = [value.get(name) for value in by_generator.values()]
        values = [float(value) for value in values if value is not None]
        generator_macro[name] = float(np.mean(values)) if values else None
    observed_fake_generators = sorted({generator_name(row) for row in fake_rows})
    return {
        "pooled_all_test": pooled,
        "formal_generator_macro": generator_macro,
        "by_formal_generator": by_generator,
        "population": {
            "formal_generators": generators,
            "observed_fake_generators": observed_fake_generators,
            "nonformal_fake_generators": sorted(set(observed_fake_generators) - set(generators)),
            "real_count": len(real_rows),
            "synthetic_count": len(fake_rows),
        },
        "aggregation_spec": {
            "pooled_includes_all_test_rows": True,
            "generator_macro_equal_weights": True,
            "public_real_rows_reused_per_generator": True,
            "test_threshold_recalibrated": False,
        },
    }


def aggregate_fixed_manifest_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float | None,
    formal_generators: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate fixed-condition scores without duplicating public human rows.

    ``pooled_clean`` is the only pooled binary metric. The corrupted
    conditions are summarized by an equal-weight macro over condition-level
    metrics; concatenating all conditions would count each public human track
    repeatedly. Generator-macro EER is computed inside each condition using
    the public-human rows for each generator, then averaged across generators
    and conditions.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate an empty fixed manifest")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("operation_id", "clean"))].append(row)
    clean_key = "clean" if "clean" in grouped else sorted(grouped)[0]
    corruption_keys = [key for key in sorted(grouped) if key != clean_key]

    def rank_or_binary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scores = [float(row["score"]) for row in group]
        labels = [int(row["label"]) for row in group]
        if threshold is None:
            return {
                "eer": eer(scores, labels),
                "auroc_ai": auroc_ai(scores, labels),
                "threshold": None,
                "count": float(len(group)),
            }
        return binary_metrics(scores, labels, threshold=threshold)

    def generator_name(row: Mapping[str, Any]) -> str:
        return str(
            row.get("generator")
            or row.get("source")
            or row.get("generator_name")
            or row.get("generator_id")
            or ""
        )

    generators = list(formal_generators or DEFAULT_FORMAL_GENERATORS)
    if not generators:
        generators = sorted({generator_name(row) for row in rows if int(row["label"]) == 1})

    def with_generator_eer(group: Sequence[Mapping[str, Any]], metric: dict[str, Any]) -> dict[str, Any]:
        values = []
        for generator in generators:
            subset = [
                row for row in group
                if int(row["label"]) == 0 or generator_name(row) == generator
            ]
            if subset:
                value = eer([float(row["score"]) for row in subset], [int(row["label"]) for row in subset])
                if value is not None:
                    values.append(float(value))
        result = dict(metric)
        result["generator_macro_eer"] = float(np.mean(values)) if values else None
        result["generator_count"] = float(len(values))
        return result

    by_operation = {
        key: with_generator_eer(grouped[key], rank_or_binary(grouped[key]))
        for key in sorted(grouped)
    }

    def macro(keys: Sequence[str]) -> dict[str, Any] | None:
        if not keys:
            return None
        result: dict[str, Any] = {"condition_count": float(len(keys))}
        names = (
            "accuracy", "balanced_accuracy", "real_specificity",
            "synthetic_recall", "fpr_real_to_fake", "fnr_fake_to_real",
            "f1", "f1_ai", "auroc_ai", "macro_f1", "eer",
            "generator_macro_eer",
        )
        for name in names:
            values = [by_operation[key].get(name) for key in keys]
            values = [float(value) for value in values if value is not None]
            result[name] = float(np.mean(values)) if values else None
        result["threshold"] = threshold
        return result

    return {
        "pooled_clean": with_generator_eer(grouped[clean_key], rank_or_binary(grouped[clean_key])),
        "by_operation": by_operation,
        "condition_macro": macro([clean_key] + corruption_keys),
        "corruption_condition_macro": macro(corruption_keys),
        "aggregation_spec": {
            "score_direction": "larger_is_real",
            "binary_primary": "corruption_condition_macro",
            "condition_weighting": "equal_condition_macro",
            "fixed_condition_count": float(len(grouped)),
            "clean_condition": clean_key,
            "corruption_condition_count": float(len(corruption_keys)),
            "pooled_clean_uses_unique_rows": True,
            "public_human_policy": "reused_within_each_generator_condition_eer_only; never concatenated across conditions",
        },
    }


def attack_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    clean_score_key: str = "clean_score",
    adversarial_score_key: str = "adversarial_score",
    label_key: str = "label",
) -> dict[str, float]:
    """Report targeted fake→real ASR separately from real false alarms."""

    rows = list(rows)
    fake_rows = [row for row in rows if int(row[label_key]) == 1]
    real_rows = [row for row in rows if int(row[label_key]) == 0]
    fake_success = [
        float(row[adversarial_score_key]) >= float(threshold) for row in fake_rows
    ]
    real_false_alarm = [
        float(row[adversarial_score_key]) < float(threshold) for row in real_rows
    ]
    clean_fake_success = [
        float(row[clean_score_key]) >= float(threshold) for row in fake_rows
    ]
    return {
        "fake_to_real_asr": float(np.mean(fake_success)) if fake_success else 0.0,
        "real_to_fake_fpr": float(np.mean(real_false_alarm)) if real_false_alarm else 0.0,
        "clean_fake_to_real_rate": float(np.mean(clean_fake_success)) if clean_fake_success else 0.0,
        "quality_pass_fake_to_real_asr": float(
            np.mean(
                [
                    success
                    for success, row in zip(fake_success, fake_rows)
                    if bool(row.get("quality_pass", True))
                ]
            )
        )
        if any(bool(row.get("quality_pass", True)) for row in fake_rows)
        else 0.0,
        "count": float(len(rows)),
    }

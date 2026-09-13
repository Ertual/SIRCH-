from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from gru_factorial_common import (
    EXPECTED_TEST_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXP01_OUTPUTS,
    EXPERIMENT_DIR,
    sha256_file,
)


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
)
CELL_ORDER = (
    "lstm_original",
    "lstm_augmented",
    "gru_original",
    "gru_augmented",
)
CELL_METADATA = {
    "lstm_original": ("LSTM", "original"),
    "lstm_augmented": ("LSTM", "enrichi"),
    "gru_original": ("GRU", "original"),
    "gru_augmented": ("GRU", "enrichi"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def score_from_row(row: dict[str, str]) -> float:
    if "video_score" in row:
        return float(row["video_score"])
    candidates = [key for key in row if key.startswith("max_rolling_mean_k")]
    if len(candidates) != 1:
        raise RuntimeError("Colonne de score video absente ou ambigue.")
    return float(row[candidates[0]])


def binary_metrics(
    truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-15),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
    }


def exact_sign_flip_pvalue(values: np.ndarray) -> tuple[float, int]:
    magnitudes = [int(abs(value)) for value in values if value != 0]
    if not magnitudes:
        return 1.0, 0
    observed = abs(int(np.sum(values)))
    distribution = {0: 1}
    for magnitude in magnitudes:
        updated: dict[int, int] = {}
        for total, count in distribution.items():
            updated[total - magnitude] = updated.get(total - magnitude, 0) + count
            updated[total + magnitude] = updated.get(total + magnitude, 0) + count
        distribution = updated
    extreme = sum(count for total, count in distribution.items() if abs(total) >= observed)
    return float(extreme / (2 ** len(magnitudes))), len(magnitudes)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    cell_dirs = {
        "lstm_original": (
            EXPERIMENT_DIR.parent
            / "exp02_validation_threshold_k"
            / "outputs"
            / "final_test_locked"
        ),
        "lstm_augmented": (
            EXPERIMENT_DIR.parent
            / "exp05_augmented_evaluation"
            / "outputs"
            / "final_test_locked"
        ),
        "gru_original": EXPERIMENT_DIR / "outputs" / "final_test_locked" / "gru_original",
        "gru_augmented": EXPERIMENT_DIR / "outputs" / "final_test_locked" / "gru_augmented",
    }
    required = [EXP01_OUTPUTS / "test_manifest.csv"]
    for directory in cell_dirs.values():
        required.extend(
            [directory / "final_test_metrics.json", directory / "video_predictions.csv"]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Artefacts factoriels absents: " + ", ".join(missing))

    manifest_path = EXP01_OUTPUTS / "test_manifest.csv"
    if sha256_file(manifest_path) != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le manifeste de test a change.")
    manifest_rows = read_csv(manifest_path)
    if len(manifest_rows) != EXPECTED_TEST_COUNT:
        raise RuntimeError("Le manifeste de test doit contenir exactement 592 lignes.")

    results = {}
    rows_by_cell = {}
    for key, directory in cell_dirs.items():
        result_path = directory / "final_test_metrics.json"
        prediction_path = directory / "video_predictions.csv"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = read_csv(prediction_path)
        if result.get("status") != "final_locked_test_complete":
            raise RuntimeError(f"Evaluation non finalisee: {key}")
        if result.get("no_post_test_adjustment") is not True:
            raise RuntimeError(f"Evaluation non verrouillee: {key}")
        if result.get("test_manifest_sha256") != EXPECTED_TEST_MANIFEST_SHA256:
            raise RuntimeError(f"Mauvais manifeste de test: {key}")
        if int(result["metrics"]["n"]) != EXPECTED_TEST_COUNT or len(rows) != EXPECTED_TEST_COUNT:
            raise RuntimeError(f"Effectif inattendu: {key}")
        results[key] = result
        rows_by_cell[key] = rows

    ordered_hashes = []
    for index, manifest in enumerate(manifest_rows):
        expected_hash = manifest["sha256"]
        expected_label = int(manifest["label"])
        for key in CELL_ORDER:
            row = rows_by_cell[key][index]
            if row["video_sha256"] != expected_hash or int(row["label"]) != expected_label:
                raise RuntimeError(f"Divergence d'appariement pour {key}, ligne {index + 1}.")
        ordered_hashes.append(expected_hash)

    truth = np.asarray([int(row["label"]) for row in manifest_rows], dtype=np.int8)
    predictions = {
        key: np.asarray([int(row["prediction"]) for row in rows_by_cell[key]], dtype=np.int8)
        for key in CELL_ORDER
    }
    scores = {
        key: np.asarray([score_from_row(row) for row in rows_by_cell[key]], dtype=float)
        for key in CELL_ORDER
    }
    point = {
        key: binary_metrics(truth, predictions[key], scores[key]) for key in CELL_ORDER
    }

    def effects(metric: str, values: dict[str, dict[str, float]]) -> dict[str, float]:
        lo = values["lstm_original"][metric]
        la = values["lstm_augmented"][metric]
        go = values["gru_original"][metric]
        ga = values["gru_augmented"][metric]
        return {
            "enrichment_within_lstm": la - lo,
            "enrichment_within_gru": ga - go,
            "architecture_gru_minus_lstm_original": go - lo,
            "architecture_gru_minus_lstm_augmented": ga - la,
            "main_enrichment_average": ((la - lo) + (ga - go)) / 2,
            "main_architecture_gru_minus_lstm_average": ((go - lo) + (ga - la)) / 2,
            "interaction": (ga - go) - (la - lo),
        }

    point_effects = {metric: effects(metric, point) for metric in METRIC_NAMES}
    effect_names = tuple(point_effects["accuracy"])
    bootstrap = {
        metric: {
            effect: np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
            for effect in effect_names
        }
        for metric in METRIC_NAMES
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, EXPECTED_TEST_COUNT, size=EXPECTED_TEST_COUNT)
        sampled_truth = truth[indices]
        sampled = {
            key: binary_metrics(
                sampled_truth, predictions[key][indices], scores[key][indices]
            )
            for key in CELL_ORDER
        }
        for metric in METRIC_NAMES:
            sampled_effects = effects(metric, sampled)
            for effect in effect_names:
                bootstrap[metric][effect][iteration] = sampled_effects[effect]
        if (iteration + 1) % 1000 == 0:
            print(f"Bootstrap factoriel apparie: {iteration + 1}/{BOOTSTRAP_ITERATIONS}", flush=True)

    intervals = {}
    interval_rows = []
    for metric in METRIC_NAMES:
        intervals[metric] = {}
        for effect in effect_names:
            lower, upper = percentile_interval(bootstrap[metric][effect])
            estimate = point_effects[metric][effect]
            intervals[metric][effect] = {
                "estimate": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
            interval_rows.append(
                {
                    "metric": metric,
                    "effect": effect,
                    "estimate": f"{estimate:.10f}",
                    "ci95_lower": f"{lower:.10f}",
                    "ci95_upper": f"{upper:.10f}",
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "seed": BOOTSTRAP_SEED,
                }
            )

    correctness = {
        key: (predictions[key] == truth).astype(np.int8) for key in CELL_ORDER
    }
    per_video_accuracy_interaction = (
        correctness["gru_augmented"]
        - correctness["gru_original"]
        - correctness["lstm_augmented"]
        + correctness["lstm_original"]
    )
    sign_flip_p, nonzero_contrasts = exact_sign_flip_pvalue(
        per_video_accuracy_interaction
    )
    accuracy_interaction = intervals["accuracy"]["interaction"]
    ci_excludes_zero = bool(
        accuracy_interaction["ci95_lower"] > 0
        or accuracy_interaction["ci95_upper"] < 0
    )
    significant = bool(sign_flip_p < 0.05 and ci_excludes_zero)

    output_dir = EXPERIMENT_DIR / "outputs" / "factorial_2x2"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for key in CELL_ORDER:
        architecture, enrichment = CELL_METADATA[key]
        metrics = results[key]["metrics"]
        table_rows.append(
            {
                "cell": key,
                "architecture": architecture,
                "enrichment": enrichment,
                "theta": results[key]["theta"],
                "k": results[key]["k"],
                "n": metrics["n"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                **{metric: f"{point[key][metric]:.10f}" for metric in METRIC_NAMES},
            }
        )
    with (output_dir / "factorial_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    with (output_dir / "interaction_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(interval_rows)

    payload = {
        "status": "factorial_2x2_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "design": "architecture (LSTM/GRU) x enrichment (original/augmented)",
        "same_test_row_sequence_verified_by_sha256": True,
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "n": EXPECTED_TEST_COUNT,
        "cells": {
            key: {
                "architecture": CELL_METADATA[key][0],
                "enrichment": CELL_METADATA[key][1],
                "theta": results[key]["theta"],
                "k": results[key]["k"],
                "metrics": point[key],
                "predictions_sha256": sha256_file(cell_dirs[key] / "video_predictions.csv"),
                "result_sha256": sha256_file(cell_dirs[key] / "final_test_metrics.json"),
            }
            for key in CELL_ORDER
        },
        "paired_bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "effect_intervals": intervals,
        },
        "primary_interaction_test": {
            "scale": "accuracy probability difference-in-differences",
            "formula": "(GRU_augmented-GRU_original)-(LSTM_augmented-LSTM_original)",
            **accuracy_interaction,
            "exact_sign_flip_two_sided_p_value": sign_flip_p,
            "nonzero_per_video_interaction_contrasts": nonzero_contrasts,
            "bootstrap_ci_excludes_zero": ci_excludes_zero,
            "significant_at_0_05": significant,
        },
    }
    (output_dir / "factorial_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    display_metrics = ("accuracy", "balanced_accuracy", "f1", "recall", "specificity")
    estimates = [intervals[name]["interaction"]["estimate"] for name in display_metrics]
    lower = [intervals[name]["interaction"]["ci95_lower"] for name in display_metrics]
    upper = [intervals[name]["interaction"]["ci95_upper"] for name in display_metrics]
    errors = np.asarray(
        [
            [estimate - lo for estimate, lo in zip(estimates, lower)],
            [hi - estimate for estimate, hi in zip(estimates, upper)],
        ]
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    positions = np.arange(len(display_metrics))
    axis.errorbar(positions, estimates, yerr=errors, fmt="o", color="#24506f", capsize=5)
    axis.axhline(0, color="#a8453b", linewidth=1.2, linestyle="--")
    axis.set_xticks(positions, labels=[name.replace("_", "\n") for name in display_metrics])
    axis.set_ylabel("Interaction: delta enrichissement GRU - delta LSTM")
    axis.set_title("Interaction architecture x enrichissement (IC bootstrap apparies 95 %)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "interaction_ci95.png", bbox_inches="tight")
    plt.close(figure)

    conclusion = (
        "Interaction statistiquement detectee a 5 %."
        if significant
        else "Aucune interaction statistiquement detectee a 5 %."
    )
    summary = [
        "# Factoriel SIRCH 2x2 sur test propre",
        "",
        "Les quatre configurations ont leur propre theta/K choisi sur validation et sont comparees sur les memes 592 videos de test.",
        "",
        "| Architecture | Enrichissement | theta | K | Accuracy | Precision | Rappel | Specificite | F1 | ROC-AUC | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in CELL_ORDER:
        architecture, enrichment = CELL_METADATA[key]
        metrics = results[key]["metrics"]
        summary.append(
            f"| {architecture} | {enrichment} | {results[key]['theta']:.2f} | {results[key]['k']} | "
            f"{point[key]['accuracy']:.4f} | {point[key]['precision']:.4f} | "
            f"{point[key]['recall']:.4f} | {point[key]['specificity']:.4f} | "
            f"{point[key]['f1']:.4f} | {point[key]['roc_auc']:.4f} | "
            f"{metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']} |"
        )
    summary.extend(
        [
            "",
            "## Interaction primaire sur l'accuracy",
            "",
            f"- Effet enrichissement avec LSTM : {point_effects['accuracy']['enrichment_within_lstm']:+.4f}",
            f"- Effet enrichissement avec GRU : {point_effects['accuracy']['enrichment_within_gru']:+.4f}",
            f"- Difference des differences : {accuracy_interaction['estimate']:+.4f} "
            f"[IC 95 % {accuracy_interaction['ci95_lower']:+.4f}; {accuracy_interaction['ci95_upper']:+.4f}]",
            f"- Test exact par permutation de signes appariee : p={sign_flip_p:.6g}",
            f"- Conclusion : {conclusion}",
            "",
            "L'interaction est definie sur l'echelle additive des probabilites de classification correcte.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

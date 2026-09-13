from __future__ import annotations

import csv
import collections
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = (
    PROJECT_ROOT
    / "revision"
    / "exp02_validation_threshold_k"
    / "outputs"
    / "final_test_locked"
)
AUGMENTED_DIR = (
    PROJECT_ROOT
    / "revision"
    / "exp05_augmented_evaluation"
    / "outputs"
    / "final_test_locked"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EXPECTED_TEST_COUNT = 592
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


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def binary_metrics(
    truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    truth = truth.astype(np.int8, copy=False)
    predictions = predictions.astype(np.int8, copy=False)
    tp = int(np.sum((truth == 1) & (predictions == 1)))
    tn = int(np.sum((truth == 0) & (predictions == 0)))
    fp = int(np.sum((truth == 0) & (predictions == 1)))
    fn = int(np.sum((truth == 1) & (predictions == 0)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "precision": precision,
        "recall": sensitivity,
        "specificity": specificity,
        "f1": 2 * precision * sensitivity / max(precision + sensitivity, 1e-15),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
    }


def score_from_row(row: dict[str, str]) -> float:
    if "video_score" in row:
        return float(row["video_score"])
    candidates = [key for key in row if key.startswith("max_rolling_mean_k")]
    if len(candidates) != 1:
        raise RuntimeError("Colonne de score video originale introuvable ou ambigue.")
    return float(row[candidates[0]])


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def main() -> None:
    original_metrics_path = ORIGINAL_DIR / "final_test_metrics.json"
    augmented_metrics_path = AUGMENTED_DIR / "final_test_metrics.json"
    original_predictions_path = ORIGINAL_DIR / "video_predictions.csv"
    augmented_predictions_path = AUGMENTED_DIR / "video_predictions.csv"
    test_manifest_path = (
        PROJECT_ROOT
        / "revision"
        / "exp01_manifests_sha256"
        / "outputs"
        / "test_manifest.csv"
    )
    required = (
        original_metrics_path,
        augmented_metrics_path,
        original_predictions_path,
        augmented_predictions_path,
        test_manifest_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Les deux evaluations finales verrouillees doivent exister avant la comparaison: "
            + ", ".join(missing)
        )

    original_result = json.loads(original_metrics_path.read_text(encoding="utf-8"))
    augmented_result = json.loads(augmented_metrics_path.read_text(encoding="utf-8"))
    for name, result in (("original", original_result), ("augmented", augmented_result)):
        if result.get("status") != "final_locked_test_complete":
            raise RuntimeError(f"Evaluation {name} non finalisee.")
        if result.get("no_post_test_adjustment") is not True:
            raise RuntimeError(f"Evaluation {name} non conforme au verrouillage.")
        if int(result["metrics"]["n"]) != EXPECTED_TEST_COUNT:
            raise RuntimeError(f"Evaluation {name}: nombre de videos inattendu.")

    original_rows = read_csv(original_predictions_path)
    augmented_rows = read_csv(augmented_predictions_path)
    manifest_rows = read_csv(test_manifest_path)
    if (
        len(original_rows) != EXPECTED_TEST_COUNT
        or len(augmented_rows) != EXPECTED_TEST_COUNT
        or len(manifest_rows) != EXPECTED_TEST_COUNT
    ):
        raise RuntimeError("Les fichiers de predictions doivent contenir exactement 592 videos.")
    ordered_hashes: list[str] = []
    for index, (manifest, original, augmented) in enumerate(
        zip(manifest_rows, original_rows, augmented_rows), start=1
    ):
        expected_hash = manifest["sha256"]
        if original["video_sha256"] != expected_hash or augmented["video_sha256"] != expected_hash:
            raise RuntimeError(f"Sequence de hashes divergente a la ligne {index}.")
        expected_label = int(manifest["label"])
        if int(original["label"]) != expected_label or int(augmented["label"]) != expected_label:
            raise RuntimeError(f"Etiquettes divergentes a la ligne {index}.")
        ordered_hashes.append(expected_hash)

    hash_counts = collections.Counter(ordered_hashes)
    duplicate_groups = {key: count for key, count in hash_counts.items() if count > 1}
    truth = np.asarray([int(row["label"]) for row in manifest_rows])
    original_predictions = np.asarray([int(row["prediction"]) for row in original_rows])
    augmented_predictions = np.asarray([int(row["prediction"]) for row in augmented_rows])
    original_scores = np.asarray([score_from_row(row) for row in original_rows])
    augmented_scores = np.asarray([score_from_row(row) for row in augmented_rows])

    original_correct = original_predictions == truth
    augmented_correct = augmented_predictions == truth
    both_correct = int(np.sum(original_correct & augmented_correct))
    original_only = int(np.sum(original_correct & ~augmented_correct))
    augmented_only = int(np.sum(~original_correct & augmented_correct))
    both_wrong = int(np.sum(~original_correct & ~augmented_correct))
    mcnemar_p = exact_mcnemar_pvalue(original_only, augmented_only)

    point_original = binary_metrics(truth, original_predictions, original_scores)
    point_augmented = binary_metrics(truth, augmented_predictions, augmented_scores)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {
        model: {metric: np.empty(BOOTSTRAP_ITERATIONS) for metric in METRIC_NAMES}
        for model in ("original", "augmented", "delta")
    }
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, EXPECTED_TEST_COUNT, size=EXPECTED_TEST_COUNT)
        sampled_truth = truth[indices]
        original_sample = binary_metrics(
            sampled_truth, original_predictions[indices], original_scores[indices]
        )
        augmented_sample = binary_metrics(
            sampled_truth, augmented_predictions[indices], augmented_scores[indices]
        )
        for metric in METRIC_NAMES:
            bootstrap["original"][metric][iteration] = original_sample[metric]
            bootstrap["augmented"][metric][iteration] = augmented_sample[metric]
            bootstrap["delta"][metric][iteration] = (
                augmented_sample[metric] - original_sample[metric]
            )
        if (iteration + 1) % 1000 == 0:
            print(f"Bootstrap apparie: {iteration + 1}/{BOOTSTRAP_ITERATIONS}", flush=True)

    interval_rows: list[dict[str, object]] = []
    interval_data: dict[str, dict[str, dict[str, float]]] = {
        "original": {},
        "augmented": {},
        "delta_augmented_minus_original": {},
    }
    for metric in METRIC_NAMES:
        for model, point in (
            ("original", point_original[metric]),
            ("augmented", point_augmented[metric]),
            ("delta", point_augmented[metric] - point_original[metric]),
        ):
            lower, upper = percentile_interval(bootstrap[model][metric])
            output_model = "delta_augmented_minus_original" if model == "delta" else model
            interval_data[output_model][metric] = {
                "estimate": float(point),
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
            interval_rows.append(
                {
                    "metric": metric,
                    "estimate_type": output_model,
                    "estimate": f"{point:.10f}",
                    "ci95_lower": f"{lower:.10f}",
                    "ci95_upper": f"{upper:.10f}",
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "seed": BOOTSTRAP_SEED,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "bootstrap_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(interval_rows)
    with (OUTPUT_DIR / "mcnemar_contingency.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", "augmented_correct", "augmented_wrong"])
        writer.writerow(["original_correct", both_correct, original_only])
        writer.writerow(["original_wrong", augmented_only, both_wrong])

    result = {
        "status": "paired_locked_test_comparison_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "same_test_row_sequence_verified_by_sha256": True,
        "n": EXPECTED_TEST_COUNT,
        "unique_content_sha256": len(hash_counts),
        "within_test_duplicate_sha256_groups": len(duplicate_groups),
        "within_test_duplicate_rows_beyond_first": sum(
            count - 1 for count in duplicate_groups.values()
        ),
        "within_test_duplicate_sha256": duplicate_groups,
        "inputs": {
            "original_predictions_sha256": sha256_file(original_predictions_path),
            "augmented_predictions_sha256": sha256_file(augmented_predictions_path),
            "original_result_sha256": sha256_file(original_metrics_path),
            "augmented_result_sha256": sha256_file(augmented_metrics_path),
        },
        "locked_configurations": {
            "original": {
                "theta": original_result["theta"],
                "k": original_result["k"],
                "model_sha256": original_result["model_sha256"],
            },
            "augmented": {
                "theta": augmented_result["theta"],
                "k": augmented_result["k"],
                "weights_sha256": augmented_result["weights_sha256"],
                "source_best_epoch": augmented_result["source_best_epoch"],
            },
        },
        "mcnemar_exact": {
            "both_correct": both_correct,
            "original_correct_augmented_wrong_b": original_only,
            "original_wrong_augmented_correct_c": augmented_only,
            "both_wrong": both_wrong,
            "discordant_pairs": original_only + augmented_only,
            "two_sided_exact_p_value": mcnemar_p,
            "significant_at_0_05": bool(mcnemar_p < 0.05),
        },
        "bootstrap": {
            "method": "paired percentile bootstrap",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "intervals": interval_data,
        },
    }
    (OUTPUT_DIR / "paired_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    display_metrics = ("accuracy", "f1", "recall", "specificity", "roc_auc", "pr_auc")
    estimates = [interval_data["delta_augmented_minus_original"][name]["estimate"] for name in display_metrics]
    lower = [interval_data["delta_augmented_minus_original"][name]["ci95_lower"] for name in display_metrics]
    upper = [interval_data["delta_augmented_minus_original"][name]["ci95_upper"] for name in display_metrics]
    errors = np.asarray(
        [[estimate - lo for estimate, lo in zip(estimates, lower)], [hi - estimate for estimate, hi in zip(estimates, upper)]]
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    positions = np.arange(len(display_metrics))
    axis.errorbar(positions, estimates, yerr=errors, fmt="o", color="#1f4b7a", capsize=5)
    axis.axhline(0, color="#b5473c", linewidth=1.2, linestyle="--")
    axis.set_xticks(positions, labels=[name.replace("_", "\n") for name in display_metrics])
    axis.set_ylabel("Difference enrichi - original")
    axis.set_title("Intervalles de confiance bootstrap apparies a 95 %")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "bootstrap_delta_ci95.png", bbox_inches="tight")
    plt.close(figure)

    significant = "oui" if mcnemar_p < 0.05 else "non"
    summary = [
        "# Comparaison finale appariee : original contre enrichi",
        "",
        "Les deux configurations ont ete verrouillees sur validation puis appliquees une seule fois aux memes 592 videos de test.",
        "La sequence des 592 lignes a ete verifiee position par position par SHA-256.",
        f"Le test contient {len(hash_counts)} contenus SHA-256 uniques et {len(duplicate_groups)} groupe de doublon interne, sans fuite entre splits.",
        "",
        "## McNemar exact",
        "",
        f"- Corrects par les deux : {both_correct}",
        f"- Original correct, enrichi faux (b) : {original_only}",
        f"- Original faux, enrichi correct (c) : {augmented_only}",
        f"- Faux pour les deux : {both_wrong}",
        f"- p bilaterale exacte : {mcnemar_p:.6g}",
        f"- Difference significative a 5 % : {significant}",
        "",
        "## Estimations et IC 95 % apparies",
        "",
        "| Metrique | Original | Enrichi | Delta enrichi-original [IC 95 %] |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRIC_NAMES:
        delta = interval_data["delta_augmented_minus_original"][metric]
        summary.append(
            f"| {metric} | {point_original[metric]:.4f} | {point_augmented[metric]:.4f} | "
            f"{delta['estimate']:+.4f} [{delta['ci95_lower']:+.4f}; {delta['ci95_upper']:+.4f}] |"
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

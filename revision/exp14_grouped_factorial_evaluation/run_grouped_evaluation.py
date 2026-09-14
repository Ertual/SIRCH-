from __future__ import annotations

import collections
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_curve

from grouped_evaluation_common import (
    EXP11_OUTPUTS,
    EXP13_EXTERNAL,
    EXPECTED_MANIFEST_HASHES,
    EXPECTED_TEST_COUNT,
    EXPECTED_VALIDATION_COUNT,
    K_VALUES,
    METRIC_NAMES,
    MODEL_KEYS,
    MODEL_METADATA,
    N_FRAMES,
    PROJECT_ROOT,
    STRIDE_FRAMES,
    THRESHOLDS,
    binary_metrics,
    evaluate_grid,
    load_grouped_manifest,
    load_hard_negative_manifest,
    load_locked_model_heads,
    records_for_model,
    rolling_means,
    score_partition,
    selection_key,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
EXTERNAL_CACHE_ROOT = Path(
    os.environ.get(
        "SIRCH_EXP14_CACHE_DIR",
        r"C:\SIRCH_ENV\models\revision\exp14_grouped_factorial_evaluation",
    )
)
EXP13_REPO_OUTPUTS = PROJECT_ROOT / "revision" / "exp13_grouped_retraining" / "outputs"
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42


def save_selection_curve(grid: list[dict[str, Any]], selected: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.3), dpi=180)
    colors = ["#315d8a", "#2f8f6b", "#c68a16", "#ba4d57", "#6d5b87"]
    for color, k_value in zip(colors, K_VALUES):
        subset = [row for row in grid if int(row["k"]) == k_value]
        axis.plot(
            [float(row["threshold"]) for row in subset],
            [float(row["f1"]) for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=color,
            label=f"K={k_value}",
        )
    axis.scatter(
        [float(selected["threshold"])],
        [float(selected["f1"])],
        s=72,
        marker="*",
        color="#111111",
        label="Selected",
        zorder=5,
    )
    axis.set_xlabel("Threshold theta")
    axis.set_ylabel("Validation F1")
    axis.set_xticks(THRESHOLDS[::2])
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_training_curve(log_path: Path, title: str, path: Path) -> None:
    rows = []
    with log_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {log_path}")
    epochs = np.asarray([int(float(row["epoch"])) + 1 for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=180)
    axes[0].plot(epochs, [float(row["loss"]) for row in rows], color="#315d8a", label="Train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], color="#ba4d57", label="Validation")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [float(row["accuracy"]) for row in rows], color="#315d8a", label="Train")
    axes[1].plot(epochs, [float(row["val_accuracy"]) for row in rows], color="#ba4d57", label="Validation")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def copy_if_changed(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(source) == sha256_file(destination):
        return
    shutil.copy2(source, destination)


def collect_training_outputs() -> None:
    pipeline_path = EXP13_EXTERNAL / "pipeline_status.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("status") != "complete":
        raise RuntimeError("Cannot collect exp13 before all four trainings complete.")
    EXP13_REPO_OUTPUTS.mkdir(parents=True, exist_ok=True)
    root_files = (
        "pipeline_status.json",
        "pipeline_stdout.log",
        "pipeline_stderr.log",
        "feature_cache_status.json",
        "feature_cache_manifest.json",
        "uniform_frame_feature_index.csv",
    )
    for name in root_files:
        source = EXP13_EXTERNAL / name
        if source.exists():
            copy_if_changed(source, EXP13_REPO_OUTPUTS / name)
    for key in MODEL_KEYS:
        source_root = EXP13_EXTERNAL / key
        destination_root = EXP13_REPO_OUTPUTS / key
        for source in source_root.rglob("*"):
            if source.is_file() and not source.name.endswith(".tmp"):
                copy_if_changed(source, destination_root / source.relative_to(source_root))
        log_path = next(destination_root.glob("training_log_*.csv"))
        save_training_curve(
            log_path,
            key.replace("_", " ").title(),
            destination_root / "training_curves.png",
        )


def validation_select_and_lock(
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    lock_path = OUTPUT_ROOT / "locked_protocol.json"
    if lock_path.exists():
        protocol = json.loads(lock_path.read_text(encoding="utf-8"))
        if protocol.get("status") != "locked_before_test":
            raise RuntimeError("Existing grouped protocol lock has an invalid status.")
        for key in MODEL_KEYS:
            selected_path = OUTPUT_ROOT / "validation_selection" / key / "selected_config.json"
            if sha256_file(selected_path) != protocol["selected_configurations"][key]["selected_config_sha256"]:
                raise RuntimeError(f"Selected validation configuration changed after lock: {key}")
        print("Validation configurations already locked; test remains governed by the lock.", flush=True)
        return protocol

    rows, manifest_hash = load_grouped_manifest("validation")
    if manifest_hash != EXPECTED_MANIFEST_HASHES["validation"]:
        raise RuntimeError("Validation hash mismatch before selection.")
    cache_dir = EXTERNAL_CACHE_ROOT / "validation_cache"
    records = score_partition(
        "validation",
        rows,
        backbone,
        heads,
        model_metadata,
        cache_dir,
    )
    selections: dict[str, Any] = {}
    for key in MODEL_KEYS:
        model_records = records_for_model(records, key)
        model_output = OUTPUT_ROOT / "validation_selection" / key
        model_output.mkdir(parents=True, exist_ok=True)
        stream_rows = []
        for record in model_records:
            for score_index, score in enumerate(record["scores"]):
                stream_rows.append(
                    {
                        "dataset": record["dataset"],
                        "relative_path": record["relative_path"],
                        "video_sha256": record["video_sha256"],
                        "label": record["label"],
                        "score_index": score_index,
                        "end_frame_index": N_FRAMES - 1 + score_index * STRIDE_FRAMES,
                        "score": f"{float(score):.10f}",
                        "original_frame_count": record["original_frame_count"],
                        "padded_frames": record["padded_frames"],
                    }
                )
        write_csv(model_output / "validation_score_streams.csv", stream_rows)
        grid = evaluate_grid(model_records)
        selected = max(grid, key=selection_key)
        for grid_row in grid:
            grid_row["selected"] = int(grid_row is selected)
        write_csv(model_output / "selection_grid.csv", grid)
        selected_payload = {
            "status": "selected_on_grouped_validation_only",
            "created_utc": utc_now(),
            "experiment": key,
            "architecture": MODEL_METADATA[key]["architecture"],
            "enrichment": MODEL_METADATA[key]["enrichment"],
            "validation_manifest_path": str(EXP11_OUTPUTS / "validation_manifest_grouped.csv"),
            "validation_manifest_sha256": manifest_hash,
            "n_validation": EXPECTED_VALIDATION_COUNT,
            "test_manifest_read": False,
            "selection_rule": (
                "max F1, then balanced accuracy, recall, minimum FPR, minimum K, "
                "threshold closest to 0.50"
            ),
            "candidate_k": list(K_VALUES),
            "candidate_thresholds": list(THRESHOLDS),
            "model": {key2: value for key2, value in model_metadata[key].items()},
            "selected": selected,
            "videos_with_padding": sum(int(record["padded_frames"]) > 0 for record in model_records),
        }
        selected_path = model_output / "selected_config.json"
        write_json(selected_path, selected_payload)
        save_selection_curve(grid, selected, model_output / "theta_selection_curve.png")
        summary = [
            f"# Validation-only theta/K selection: {key}",
            "",
            f"Selected on the sealed {EXPECTED_VALIDATION_COUNT}-video grouped validation split.",
            "The test manifest was not read.",
            "",
            f"- theta: {float(selected['threshold']):.2f}",
            f"- K: {int(selected['k'])}",
            f"- F1: {float(selected['f1']):.4f}",
            f"- balanced accuracy: {float(selected['balanced_accuracy']):.4f}",
            f"- precision: {float(selected['precision']):.4f}",
            f"- recall: {float(selected['recall']):.4f}",
            f"- confusion matrix: TN={selected['tn']}, FP={selected['fp']}, FN={selected['fn']}, TP={selected['tp']}",
        ]
        (model_output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        selections[key] = {
            "theta": float(selected["threshold"]),
            "k": int(selected["k"]),
            "selected_config_path": str(selected_path),
            "selected_config_sha256": sha256_file(selected_path),
        }

    protocol = {
        "status": "locked_before_test",
        "locked_utc": utc_now(),
        "validation_manifest_sha256": manifest_hash,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "test_manifest_not_read_before_lock": True,
        "no_post_test_adjustment": True,
        "models": {
            key: {key2: value for key2, value in model_metadata[key].items()} for key in MODEL_KEYS
        },
        "selected_configurations": selections,
    }
    write_json(lock_path, protocol)
    print(f"LOCKED before test: {lock_path}", flush=True)
    return protocol


def save_confusion_matrix(metrics: dict[str, Any], title: str, path: Path) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_yticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def save_roc_pr_curves(
    truth: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    output_dir: Path,
) -> None:
    fpr, tpr, roc_thresholds = roc_curve(truth, scores)
    precision, recall, pr_thresholds = precision_recall_curve(truth, scores)
    roc_rows = [
        {"false_positive_rate": f"{x:.10f}", "true_positive_rate": f"{y:.10f}", "threshold": f"{z:.10f}"}
        for x, y, z in zip(fpr, tpr, roc_thresholds)
    ]
    pr_rows = [
        {
            "recall": f"{x:.10f}",
            "precision": f"{y:.10f}",
            "threshold": "" if index == len(pr_thresholds) else f"{pr_thresholds[index]:.10f}",
        }
        for index, (x, y) in enumerate(zip(recall, precision))
    ]
    write_csv(output_dir / "roc_curve.csv", roc_rows)
    write_csv(output_dir / "precision_recall_curve.csv", pr_rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), dpi=180)
    axes[0].plot(fpr, tpr, color="#315d8a", label=f"AUC={float(metrics['roc_auc']):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("ROC")
    axes[1].plot(recall, precision, color="#2f8f6b", label=f"AP={float(metrics['pr_auc']):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_pr_curves.png", bbox_inches="tight")
    plt.close(figure)


def evaluate_test_once(
    protocol: dict[str, Any],
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    output_root = OUTPUT_ROOT / "final_test_locked"
    joint_path = output_root / "joint_test_evaluation.json"
    if joint_path.exists():
        existing = json.loads(joint_path.read_text(encoding="utf-8"))
        if existing.get("status") != "final_locked_test_complete":
            raise RuntimeError("Existing joint test result is not complete.")
        print("Final grouped test already complete; refusing to evaluate it again.", flush=True)
        return existing

    lock_path = OUTPUT_ROOT / "locked_protocol.json"
    if protocol.get("status") != "locked_before_test" or protocol.get("no_post_test_adjustment") is not True:
        raise RuntimeError("The four validation configurations are not locked.")
    protocol_hash = sha256_file(lock_path)
    for key in MODEL_KEYS:
        selected_path = OUTPUT_ROOT / "validation_selection" / key / "selected_config.json"
        if sha256_file(selected_path) != protocol["selected_configurations"][key]["selected_config_sha256"]:
            raise RuntimeError(f"Validation configuration changed after lock: {key}")

    rows, manifest_hash = load_grouped_manifest("test")
    if manifest_hash != EXPECTED_MANIFEST_HASHES["test"]:
        raise RuntimeError("Test hash mismatch after lock.")
    state_path = EXTERNAL_CACHE_ROOT / "test_logical_run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("A partial test cache belongs to another protocol lock.")
    else:
        state = {
            "status": "running",
            "logical_run_started_utc": utc_now(),
            "attempts": 0,
            "protocol_sha256": protocol_hash,
            "test_manifest_sha256": manifest_hash,
        }
    state["attempts"] = int(state["attempts"]) + 1
    state["last_attempt_started_utc"] = utc_now()
    write_json(state_path, state)
    try:
        records = score_partition(
            "test",
            rows,
            backbone,
            heads,
            model_metadata,
            EXTERNAL_CACHE_ROOT / "test_cache",
        )
    except BaseException:
        state.update({"status": "interrupted", "updated_utc": utc_now()})
        write_json(state_path, state)
        raise

    results: dict[str, Any] = {}
    combined_roc: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for key in MODEL_KEYS:
        model_records = records_for_model(records, key)
        locked = protocol["selected_configurations"][key]
        theta = float(locked["theta"])
        k_value = int(locked["k"])
        truth = np.asarray([int(record["label"]) for record in model_records], dtype=np.int8)
        video_scores = np.asarray(
            [float(np.max(rolling_means(record["scores"], k_value))) for record in model_records]
        )
        predictions = (video_scores >= theta).astype(np.int8)
        metrics = binary_metrics(truth, predictions, video_scores)
        model_output = output_root / key
        model_output.mkdir(parents=True, exist_ok=True)
        prediction_rows = []
        for record, score, prediction in zip(model_records, video_scores, predictions):
            prediction_rows.append(
                {
                    "dataset": record["dataset"],
                    "relative_path": record["relative_path"],
                    "video_sha256": record["video_sha256"],
                    "label": record["label"],
                    "video_score": f"{float(score):.10f}",
                    "k": k_value,
                    "threshold": f"{theta:.2f}",
                    "prediction": int(prediction),
                    "correct": int(int(record["label"]) == int(prediction)),
                    "sequence_scores": len(record["scores"]),
                    "original_frame_count": record["original_frame_count"],
                    "padded_frames": record["padded_frames"],
                }
            )
        predictions_path = model_output / "video_predictions.csv"
        write_csv(predictions_path, prediction_rows)
        save_confusion_matrix(
            metrics,
            f"{key.replace('_', ' ')} (theta={theta:.2f}, K={k_value})",
            model_output / "confusion_matrix.png",
        )
        save_roc_pr_curves(truth, video_scores, metrics, key.replace("_", " ").title(), model_output)
        fpr, tpr, _ = roc_curve(truth, video_scores)
        combined_roc[key] = (fpr, tpr, float(metrics["roc_auc"]))
        result = {
            "status": "final_locked_test_complete",
            "logical_run_started_utc": state["logical_run_started_utc"],
            "completed_utc": utc_now(),
            "no_post_test_adjustment": True,
            "experiment": key,
            "selected_config_sha256": locked["selected_config_sha256"],
            "protocol_sha256": protocol_hash,
            "test_manifest_path": str(EXP11_OUTPUTS / "test_manifest_grouped.csv"),
            "test_manifest_sha256": manifest_hash,
            "theta": theta,
            "k": k_value,
            "n_frames": N_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "model": {key2: value for key2, value in model_metadata[key].items()},
            "metrics": metrics,
            "video_predictions_sha256": sha256_file(predictions_path),
        }
        write_json(model_output / "final_test_metrics.json", result)
        results[key] = result

    colors = ["#315d8a", "#2f8f6b", "#c68a16", "#ba4d57"]
    figure, axis = plt.subplots(figsize=(7.2, 6.1), dpi=180)
    for color, key in zip(colors, MODEL_KEYS):
        fpr, tpr, auc = combined_roc[key]
        axis.plot(fpr, tpr, color=color, linewidth=1.7, label=f"{key.replace('_grouped', '')} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_root / "combined_roc_curves.png", bbox_inches="tight")
    plt.close(figure)

    joint = {
        "status": "final_locked_test_complete",
        "logical_run_started_utc": state["logical_run_started_utc"],
        "completed_utc": utc_now(),
        "attempts_with_cache_resume": state["attempts"],
        "test_evaluated_as_one_locked_logical_run": True,
        "no_post_test_adjustment": True,
        "protocol_sha256": protocol_hash,
        "test_manifest_sha256": manifest_hash,
        "n_test": EXPECTED_TEST_COUNT,
        "results": results,
    }
    write_json(joint_path, joint)
    state.update({"status": "complete", "completed_utc": utc_now(), "joint_result": str(joint_path)})
    write_json(state_path, state)
    return joint


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(discordant, i) for i in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def load_test_arrays() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifest_rows, _ = load_grouped_manifest("test")
    truth = np.asarray([int(row["label"]) for row in manifest_rows], dtype=np.int8)
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for key in MODEL_KEYS:
        rows = []
        path = OUTPUT_ROOT / "final_test_locked" / key / "video_predictions.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != EXPECTED_TEST_COUNT:
            raise RuntimeError(f"Unexpected test prediction count: {key}")
        for index, (manifest, row) in enumerate(zip(manifest_rows, rows), start=1):
            if row["video_sha256"] != manifest["sha256"] or int(row["label"]) != int(manifest["label"]):
                raise RuntimeError(f"Test pairing mismatch for {key}, row {index}")
        predictions[key] = np.asarray([int(row["prediction"]) for row in rows], dtype=np.int8)
        scores[key] = np.asarray([float(row["video_score"]) for row in rows], dtype=float)
    return truth, predictions, scores


def paired_model_comparison(
    architecture: str,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    key_original = f"{architecture.lower()}_original_grouped"
    key_augmented = f"{architecture.lower()}_augmented_grouped"
    output_dir = OUTPUT_ROOT / "paired_comparisons" / architecture.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_correct = predictions[key_original] == truth
    augmented_correct = predictions[key_augmented] == truth
    both_correct = int(np.sum(original_correct & augmented_correct))
    original_only = int(np.sum(original_correct & ~augmented_correct))
    augmented_only = int(np.sum(~original_correct & augmented_correct))
    both_wrong = int(np.sum(~original_correct & ~augmented_correct))
    p_value = exact_mcnemar_pvalue(original_only, augmented_only)
    point = {
        "original": binary_metrics(truth, predictions[key_original], scores[key_original]),
        "augmented": binary_metrics(truth, predictions[key_augmented], scores[key_augmented]),
    }
    bootstrap = {
        estimate: {metric: np.empty(BOOTSTRAP_ITERATIONS) for metric in METRIC_NAMES}
        for estimate in ("original", "augmented", "delta")
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if architecture == "LSTM" else 1))
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(truth), size=len(truth))
        sampled_truth = truth[indices]
        original = binary_metrics(
            sampled_truth, predictions[key_original][indices], scores[key_original][indices]
        )
        augmented = binary_metrics(
            sampled_truth, predictions[key_augmented][indices], scores[key_augmented][indices]
        )
        for metric in METRIC_NAMES:
            bootstrap["original"][metric][iteration] = float(original[metric])
            bootstrap["augmented"][metric][iteration] = float(augmented[metric])
            bootstrap["delta"][metric][iteration] = float(augmented[metric]) - float(original[metric])
    interval_rows = []
    intervals: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        lower, upper = percentile_interval(bootstrap["delta"][metric])
        estimate = float(point["augmented"][metric]) - float(point["original"][metric])
        intervals[metric] = {"estimate": estimate, "ci95_lower": lower, "ci95_upper": upper}
        interval_rows.append(
            {
                "metric": metric,
                "delta_augmented_minus_original": f"{estimate:.10f}",
                "ci95_lower": f"{lower:.10f}",
                "ci95_upper": f"{upper:.10f}",
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "seed": BOOTSTRAP_SEED + (0 if architecture == "LSTM" else 1),
            }
        )
    write_csv(output_dir / "bootstrap_delta_intervals.csv", interval_rows)
    write_csv(
        output_dir / "mcnemar_contingency.csv",
        [
            {
                "both_correct": both_correct,
                "original_correct_augmented_wrong_b": original_only,
                "original_wrong_augmented_correct_c": augmented_only,
                "both_wrong": both_wrong,
                "discordant_pairs": original_only + augmented_only,
                "two_sided_exact_p_value": f"{p_value:.10f}",
            }
        ],
    )
    payload = {
        "status": "paired_locked_test_comparison_complete",
        "completed_utc": utc_now(),
        "architecture": architecture,
        "n": len(truth),
        "same_test_rows_verified_by_sha256": True,
        "mcnemar_exact": {
            "both_correct": both_correct,
            "original_correct_augmented_wrong_b": original_only,
            "original_wrong_augmented_correct_c": augmented_only,
            "both_wrong": both_wrong,
            "discordant_pairs": original_only + augmented_only,
            "two_sided_exact_p_value": p_value,
            "significant_at_0_05": bool(p_value < 0.05),
        },
        "point_metrics": point,
        "paired_bootstrap": {
            "method": "paired percentile bootstrap",
            "iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": 0.95,
            "delta_augmented_minus_original": intervals,
        },
    }
    write_json(output_dir / "paired_comparison.json", payload)
    display = ("accuracy", "balanced_accuracy", "f1", "recall", "specificity", "roc_auc")
    estimates = [intervals[metric]["estimate"] for metric in display]
    lower = [intervals[metric]["ci95_lower"] for metric in display]
    upper = [intervals[metric]["ci95_upper"] for metric in display]
    errors = np.asarray(
        [
            [estimate - lo for estimate, lo in zip(estimates, lower)],
            [hi - estimate for estimate, hi in zip(estimates, upper)],
        ]
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    positions = np.arange(len(display))
    axis.errorbar(positions, estimates, yerr=errors, fmt="o", color="#315d8a", capsize=5)
    axis.axhline(0, color="#ba4d57", linestyle="--", linewidth=1.2)
    axis.set_xticks(positions, labels=[metric.replace("_", "\n") for metric in display])
    axis.set_ylabel("Enriched minus original")
    axis.set_title(f"{architecture}: paired bootstrap 95% intervals")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "bootstrap_delta_ci95.png", bbox_inches="tight")
    plt.close(figure)
    summary = [
        f"# {architecture}: original versus enriched",
        "",
        f"McNemar exact: b={original_only}, c={augmented_only}, p={p_value:.6g}.",
        "",
        "| Metric | Original | Enriched | Delta [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRIC_NAMES:
        interval = intervals[metric]
        summary.append(
            f"| {metric} | {float(point['original'][metric]):.4f} | {float(point['augmented'][metric]):.4f} | "
            f"{interval['estimate']:+.4f} [{interval['ci95_lower']:+.4f}; {interval['ci95_upper']:+.4f}] |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return payload


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


def factorial_analysis(
    protocol: dict[str, Any],
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / "factorial_2x2"
    output_dir.mkdir(parents=True, exist_ok=True)
    point = {key: binary_metrics(truth, predictions[key], scores[key]) for key in MODEL_KEYS}

    def effects(metric: str, values: dict[str, dict[str, Any]]) -> dict[str, float]:
        lo = float(values["lstm_original_grouped"][metric])
        la = float(values["lstm_augmented_grouped"][metric])
        go = float(values["gru_original_grouped"][metric])
        ga = float(values["gru_augmented_grouped"][metric])
        return {
            "enrichment_within_lstm": la - lo,
            "enrichment_within_gru": ga - go,
            "architecture_gru_minus_lstm_original": go - lo,
            "architecture_gru_minus_lstm_enriched": ga - la,
            "main_enrichment_average": ((la - lo) + (ga - go)) / 2,
            "main_architecture_average": ((go - lo) + (ga - la)) / 2,
            "interaction": (ga - go) - (la - lo),
        }

    point_effects = {metric: effects(metric, point) for metric in METRIC_NAMES}
    effect_names = tuple(point_effects["accuracy"])
    bootstrap = {
        metric: {effect: np.empty(BOOTSTRAP_ITERATIONS) for effect in effect_names}
        for metric in METRIC_NAMES
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(truth), size=len(truth))
        sampled_truth = truth[indices]
        sampled = {
            key: binary_metrics(sampled_truth, predictions[key][indices], scores[key][indices])
            for key in MODEL_KEYS
        }
        for metric in METRIC_NAMES:
            sampled_effects = effects(metric, sampled)
            for effect in effect_names:
                bootstrap[metric][effect][iteration] = sampled_effects[effect]
    intervals: dict[str, Any] = {}
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
                    "seed": BOOTSTRAP_SEED + 2,
                }
            )
    write_csv(output_dir / "factorial_effects_bootstrap.csv", interval_rows)
    table_rows = []
    for key in MODEL_KEYS:
        locked = protocol["selected_configurations"][key]
        table_rows.append(
            {
                "cell": key,
                "architecture": MODEL_METADATA[key]["architecture"],
                "enrichment": MODEL_METADATA[key]["enrichment"],
                "theta": locked["theta"],
                "k": locked["k"],
                **{metric: f"{float(point[key][metric]):.10f}" for metric in ("n", "tn", "fp", "fn", "tp", *METRIC_NAMES)},
            }
        )
    write_csv(output_dir / "factorial_table.csv", table_rows)
    correctness = {key: (predictions[key] == truth).astype(np.int8) for key in MODEL_KEYS}
    per_video_interaction = (
        correctness["gru_augmented_grouped"]
        - correctness["gru_original_grouped"]
        - correctness["lstm_augmented_grouped"]
        + correctness["lstm_original_grouped"]
    )
    sign_flip_p, nonzero = exact_sign_flip_pvalue(per_video_interaction)
    primary = intervals["accuracy"]["interaction"]
    ci_excludes_zero = primary["ci95_lower"] > 0 or primary["ci95_upper"] < 0
    significant = bool(sign_flip_p < 0.05 and ci_excludes_zero)
    payload = {
        "status": "factorial_2x2_complete",
        "completed_utc": utc_now(),
        "design": "architecture (LSTM/GRU) x enrichment (original/enriched)",
        "same_597_test_rows_verified_by_sha256": True,
        "test_manifest_sha256": EXPECTED_MANIFEST_HASHES["test"],
        "cells": point,
        "paired_bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED + 2,
            "effect_intervals": intervals,
        },
        "primary_interaction_test": {
            "scale": "accuracy probability difference-in-differences",
            "formula": "(GRU_enriched-GRU_original)-(LSTM_enriched-LSTM_original)",
            **primary,
            "exact_sign_flip_two_sided_p_value": sign_flip_p,
            "nonzero_per_video_interaction_contrasts": nonzero,
            "bootstrap_ci_excludes_zero": bool(ci_excludes_zero),
            "significant_at_0_05": significant,
        },
    }
    write_json(output_dir / "factorial_results.json", payload)

    metric = "accuracy"
    values = {
        "LSTM": [float(point["lstm_original_grouped"][metric]), float(point["lstm_augmented_grouped"][metric])],
        "GRU": [float(point["gru_original_grouped"][metric]), float(point["gru_augmented_grouped"][metric])],
    }
    figure, axis = plt.subplots(figsize=(7.6, 5.2), dpi=180)
    x = np.arange(2)
    axis.plot(x, values["LSTM"], marker="o", linewidth=2, color="#315d8a", label="LSTM")
    axis.plot(x, values["GRU"], marker="o", linewidth=2, color="#ba4d57", label="GRU")
    axis.set_xticks(x, labels=["Original", "Enriched"])
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(max(0, min(*values["LSTM"], *values["GRU"]) - 0.08), min(1, max(*values["LSTM"], *values["GRU"]) + 0.08))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Architecture by enrichment interaction")
    figure.tight_layout()
    figure.savefig(output_dir / "factorial_interaction.png", bbox_inches="tight")
    plt.close(figure)

    display = ("accuracy", "balanced_accuracy", "f1", "recall", "specificity")
    estimates = [intervals[name]["interaction"]["estimate"] for name in display]
    lower = [intervals[name]["interaction"]["ci95_lower"] for name in display]
    upper = [intervals[name]["interaction"]["ci95_upper"] for name in display]
    errors = np.asarray(
        [
            [estimate - lo for estimate, lo in zip(estimates, lower)],
            [hi - estimate for estimate, hi in zip(estimates, upper)],
        ]
    )
    figure, axis = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    positions = np.arange(len(display))
    axis.errorbar(positions, estimates, yerr=errors, fmt="o", color="#315d8a", capsize=5)
    axis.axhline(0, color="#ba4d57", linestyle="--", linewidth=1.2)
    axis.set_xticks(positions, labels=[name.replace("_", "\n") for name in display])
    axis.set_ylabel("Difference-in-differences")
    axis.set_title("Interaction with paired bootstrap 95% intervals")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "interaction_ci95.png", bbox_inches="tight")
    plt.close(figure)

    conclusion = "Interaction detected at 5%." if significant else "No interaction detected at 5%."
    summary = [
        "# Grouped 2x2 factorial",
        "",
        "| Architecture | Enrichment | theta | K | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in MODEL_KEYS:
        locked = protocol["selected_configurations"][key]
        metrics = point[key]
        summary.append(
            f"| {MODEL_METADATA[key]['architecture']} | {MODEL_METADATA[key]['enrichment']} | "
            f"{float(locked['theta']):.2f} | {int(locked['k'])} | {float(metrics['accuracy']):.4f} | "
            f"{float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | "
            f"{float(metrics['specificity']):.4f} | {float(metrics['f1']):.4f} | "
            f"{float(metrics['roc_auc']):.4f} | {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']} |"
        )
    summary.extend(
        [
            "",
            "## Accuracy interaction",
            "",
            f"- LSTM enrichment effect: {point_effects['accuracy']['enrichment_within_lstm']:+.4f}",
            f"- GRU enrichment effect: {point_effects['accuracy']['enrichment_within_gru']:+.4f}",
            f"- Difference-in-differences: {primary['estimate']:+.4f} "
            f"[95% CI {primary['ci95_lower']:+.4f}; {primary['ci95_upper']:+.4f}]",
            f"- Exact paired sign-flip p-value: {sign_flip_p:.6g}",
            f"- Conclusion: {conclusion}",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return payload


def evaluate_hard_negative(
    protocol: dict[str, Any],
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    rows, manifest_hash = load_hard_negative_manifest()
    records = score_partition(
        "hard_negative",
        rows,
        backbone,
        heads,
        model_metadata,
        EXTERNAL_CACHE_ROOT / "hard_negative_cache",
        hard_negative=True,
    )
    prediction_rows = []
    summary_rows = []
    for key in MODEL_KEYS:
        model_records = records_for_model(records, key)
        locked = protocol["selected_configurations"][key]
        theta = float(locked["theta"])
        k_value = int(locked["k"])
        for record in model_records:
            score = float(np.max(rolling_means(record["scores"], k_value)))
            prediction_rows.append(
                {
                    "model": key,
                    "architecture": MODEL_METADATA[key]["architecture"],
                    "enrichment": MODEL_METADATA[key]["enrichment"],
                    "theta": f"{theta:.2f}",
                    "k": k_value,
                    "category": record["category"],
                    "relative_path": record["relative_path"],
                    "video_sha256": record["video_sha256"],
                    "video_score": f"{score:.10f}",
                    "prediction": int(score >= theta),
                    "false_positive": int(score >= theta),
                }
            )
        model_rows = [row for row in prediction_rows if row["model"] == key]
        for category in ("sport", "danse", "calme", "overall"):
            category_rows = model_rows if category == "overall" else [row for row in model_rows if row["category"] == category]
            false_positives = sum(int(row["false_positive"]) for row in category_rows)
            summary_rows.append(
                {
                    "model": key,
                    "architecture": MODEL_METADATA[key]["architecture"],
                    "enrichment": MODEL_METADATA[key]["enrichment"],
                    "theta": f"{theta:.2f}",
                    "k": k_value,
                    "category": category,
                    "n_videos": len(category_rows),
                    "false_positives": false_positives,
                    "false_positive_rate": f"{false_positives / len(category_rows):.10f}",
                }
            )
    output_dir = OUTPUT_ROOT / "hard_negative_locked"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hard_negative_predictions.csv", prediction_rows)
    write_csv(output_dir / "false_positive_by_category.csv", summary_rows)
    categories = ("sport", "danse", "calme", "overall")
    colors = ["#315d8a", "#2f8f6b", "#c68a16", "#ba4d57"]
    x = np.arange(len(categories))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9.4, 5.4), dpi=180)
    for index, (key, color) in enumerate(zip(MODEL_KEYS, colors)):
        values = [
            100 * float(next(row["false_positive_rate"] for row in summary_rows if row["model"] == key and row["category"] == category))
            for category in categories
        ]
        bars = axis.bar(x + (index - 1.5) * width, values, width, color=color, label=key.replace("_grouped", ""))
        axis.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=2)
    axis.set_xticks(x, labels=["Sport (15)", "Dance (10)", "Calm (5)", "Overall (30)"])
    axis.set_ylabel("False-positive rate (%)")
    axis.set_ylim(0, 108)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "hard_negative_false_positive_rates.png", bbox_inches="tight")
    plt.close(figure)
    payload = {
        "status": "locked_hard_negative_evaluation_complete",
        "completed_utc": utc_now(),
        "hard_negative_manifest_sha256": manifest_hash,
        "hard_negative_train_leakage_audit": "PASS; zero affected videos",
        "n_videos": 30,
        "no_retuning": True,
        "results": summary_rows,
    }
    write_json(output_dir / "result.json", payload)
    summary = [
        "# Locked grouped models on hard negatives",
        "",
        "No theta/K was selected or adjusted on these 30 non-violent videos.",
        "",
        "| Model | Category | False positives | Rate |",
        "|---|---|---:|---:|",
    ]
    for row in summary_rows:
        summary.append(
            f"| {row['model']} | {row['category']} | {row['false_positives']}/{row['n_videos']} | "
            f"{100 * float(row['false_positive_rate']):.2f}% |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return payload


def make_artifact_manifests() -> None:
    for root in (EXP13_REPO_OUTPUTS, OUTPUT_ROOT):
        files = [path for path in root.rglob("*") if path.is_file() and path.name != "artifact_manifest.json"]
        artifacts = [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        ]
        write_json(
            root / "artifact_manifest.json",
            {"status": "complete", "created_utc": utc_now(), "root": str(root), "artifacts": artifacts},
        )
    figures = []
    for root in (EXP13_REPO_OUTPUTS, OUTPUT_ROOT):
        for path in root.rglob("*.png"):
            figures.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        OUTPUT_ROOT / "figure_manifest.json",
        {"status": "complete", "created_utc": utc_now(), "figures": sorted(figures, key=lambda row: row["path"])},
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    pipeline_status_path = OUTPUT_ROOT / "evaluation_pipeline_status.json"
    status = {"status": "running", "stage": "verify_training", "pid": os.getpid(), "started_utc": utc_now()}
    write_json(pipeline_status_path, status)
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("A GPU is visible although this evaluation protocol is CPU-only.")
    try:
        backbone, heads, model_metadata = load_locked_model_heads()
        owners = model_metadata.pop("_owners")
        _ = owners
        status.update({"stage": "collect_training_outputs", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        collect_training_outputs()
        status.update({"stage": "validation_selection_and_lock", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        protocol = validation_select_and_lock(backbone, heads, model_metadata)
        status.update({"stage": "single_locked_test", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        test_result = evaluate_test_once(protocol, backbone, heads, model_metadata)
        status.update({"stage": "paired_statistics", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        truth, predictions, scores = load_test_arrays()
        lstm_comparison = paired_model_comparison("LSTM", truth, predictions, scores)
        gru_comparison = paired_model_comparison("GRU", truth, predictions, scores)
        status.update({"stage": "factorial_interaction", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        factorial = factorial_analysis(protocol, truth, predictions, scores)
        status.update({"stage": "hard_negative", "updated_utc": utc_now()})
        write_json(pipeline_status_path, status)
        hard_negative = evaluate_hard_negative(protocol, backbone, heads, model_metadata)
        result = {
            "status": "complete",
            "completed_utc": utc_now(),
            "validation_locked_before_test": True,
            "test_evaluated_once": True,
            "test_result": str(OUTPUT_ROOT / "final_test_locked" / "joint_test_evaluation.json"),
            "lstm_mcnemar_p": lstm_comparison["mcnemar_exact"]["two_sided_exact_p_value"],
            "gru_mcnemar_p": gru_comparison["mcnemar_exact"]["two_sided_exact_p_value"],
            "accuracy_interaction": factorial["primary_interaction_test"],
            "hard_negative_result": str(OUTPUT_ROOT / "hard_negative_locked" / "result.json"),
            "joint_test_status": test_result["status"],
            "hard_negative_status": hard_negative["status"],
        }
        write_json(OUTPUT_ROOT / "result.json", result)
        make_artifact_manifests()
        status.update({"status": "complete", "stage": "complete", "completed_utc": utc_now()})
        write_json(pipeline_status_path, status)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        status.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": utc_now(),
            }
        )
        write_json(pipeline_status_path, status)
        raise


if __name__ == "__main__":
    main()

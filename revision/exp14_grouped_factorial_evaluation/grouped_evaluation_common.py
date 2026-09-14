from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.applications.efficientnet import preprocess_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP11_OUTPUTS = PROJECT_ROOT / "revision" / "exp11_grouped_scene_splits" / "outputs"
EXP13_EXTERNAL = Path(
    os.environ.get(
        "SIRCH_EXP13_MODEL_DIR",
        r"C:\SIRCH_ENV\models\revision\exp13_grouped_retraining",
    )
)
DATASETS_ROOT = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
N_FRAMES = 20
STRIDE_FRAMES = 5
K_VALUES = (1, 3, 5, 7, 10)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(30, 91, 5))
IMG_SIZE = 224
FRAME_BATCH_SIZE = 32
EXPECTED_VALIDATION_COUNT = 597
EXPECTED_TEST_COUNT = 597
EXPECTED_MANIFEST_HASHES = {
    "validation": "d4b3f4a41c4d118a99abd70e88943f7886970695353369259ec3ea506e3964e5",
    "test": "a37aa9cba85d28ac315c1ea813c902330a3e99525042c13012023914e0a9887a",
    "hard_negative": "4b9c6c135ec3869e7b109f1aa5a695b3a544368b07ada753919ca226e9c76dba",
}
MODEL_KEYS = (
    "lstm_original_grouped",
    "lstm_augmented_grouped",
    "gru_original_grouped",
    "gru_augmented_grouped",
)
MODEL_METADATA = {
    "lstm_original_grouped": {"architecture": "LSTM", "enrichment": "original"},
    "lstm_augmented_grouped": {"architecture": "LSTM", "enrichment": "enriched"},
    "gru_original_grouped": {"architecture": "GRU", "enrichment": "original"},
    "gru_augmented_grouped": {"architecture": "GRU", "enrichment": "enriched"},
}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_grouped_manifest(partition: str) -> tuple[list[dict[str, str]], str]:
    if partition not in {"validation", "test"}:
        raise ValueError(partition)
    path = EXP11_OUTPUTS / f"{partition}_manifest_grouped.csv"
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_HASHES[partition]:
        raise RuntimeError(f"Grouped {partition} manifest hash changed: {actual_hash}")
    rows = read_csv(path)
    expected_count = EXPECTED_VALIDATION_COUNT if partition == "validation" else EXPECTED_TEST_COUNT
    if len(rows) != expected_count:
        raise RuntimeError(f"Unexpected grouped {partition} count: {len(rows)}")
    if any(row["split"] != partition for row in rows):
        raise RuntimeError(f"A row is outside grouped {partition}.")
    counts = [sum(int(row["label"]) == value for row in rows) for value in (0, 1)]
    if counts != [300, 297]:
        raise RuntimeError(f"Unexpected grouped {partition} class counts: {counts}")
    return rows, actual_hash


def load_hard_negative_manifest() -> tuple[list[dict[str, str]], str]:
    path = (
        PROJECT_ROOT
        / "revision"
        / "exp01_manifests_sha256"
        / "outputs"
        / "hard_negative_v2_manifest.csv"
    )
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_MANIFEST_HASHES["hard_negative"]:
        raise RuntimeError(f"Hard-negative manifest hash changed: {actual_hash}")
    rows = read_csv(path)
    if len(rows) != 30 or any(row["split"] != "hard_negative_v2" for row in rows):
        raise RuntimeError("The sealed hard-negative corpus must contain exactly 30 rows.")
    categories = {name: sum(row["category"] == name for row in rows) for name in ("sport", "danse", "calme")}
    if categories != {"sport": 15, "danse": 10, "calme": 5}:
        raise RuntimeError(f"Unexpected hard-negative categories: {categories}")
    audit_path = (
        PROJECT_ROOT
        / "revision"
        / "exp12_hard_negative_leakage_audit"
        / "outputs"
        / "result.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("affected_hard_negative_video_count") != 0:
        raise RuntimeError("Hard-negative versus grouped-train leakage audit is not clean.")
    return rows, actual_hash


def resolve_video_path(row: dict[str, str], hard_negative: bool = False) -> Path:
    if hard_negative:
        return PROJECT_ROOT / "datasets" / row["relative_path"]
    if row["dataset"] == "RLVS":
        return DATASETS_ROOT / "RLVS" / "Real Life Violence Dataset" / row["relative_path"]
    if row["dataset"] == "RWF-2000":
        return DATASETS_ROOT / "RWF-2000" / row["relative_path"]
    raise RuntimeError(f"Unexpected dataset: {row['dataset']}")


def backbones_equal(reference: tf.keras.Model, candidate: tf.keras.Model) -> bool:
    left = reference.get_weights()
    right = candidate.get_weights()
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


def load_locked_model_heads() -> tuple[tf.keras.Model, dict[str, tf.keras.Model], dict[str, Any]]:
    pipeline_path = EXP13_EXTERNAL / "pipeline_status.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("status") != "complete":
        raise RuntimeError(f"Grouped retraining is not complete: {pipeline.get('status')}")

    exp13_code = PROJECT_ROOT / "revision" / "exp13_grouped_retraining"
    sys.path.insert(0, str(exp13_code))
    from train_grouped_factorial import build_full_and_head

    reference_backbone = None
    heads: dict[str, tf.keras.Model] = {}
    owners: list[tf.keras.Model] = []
    metadata: dict[str, Any] = {}
    for key in MODEL_KEYS:
        model_dir = EXP13_EXTERNAL / key
        result_path = model_dir / "training_result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError(f"Incomplete grouped model: {key}")
        architecture = str(MODEL_METADATA[key]["architecture"])
        full, backbone, head = build_full_and_head(architecture)
        weights_path = Path(str(result["best_weights_path"]))
        if sha256_file(weights_path) != result["best_full_weights_sha256"]:
            raise RuntimeError(f"Best grouped weights changed: {key}")
        full.load_weights(weights_path)
        if reference_backbone is None:
            reference_backbone = backbone
        elif not backbones_equal(reference_backbone, backbone):
            raise RuntimeError(f"Frozen EfficientNet backbone differs for {key}")
        equivalence_path = model_dir / "feature_training_equivalence.json"
        equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
        if equivalence.get("status") != "PASS":
            raise RuntimeError(f"Feature/full model equivalence did not pass: {key}")
        heads[key] = head
        owners.append(full)
        metadata[key] = {
            **MODEL_METADATA[key],
            "best_epoch": int(result["best_epoch"]),
            "best_val_loss": float(result["best_val_loss"]),
            "weights_path": str(weights_path),
            "weights_sha256": result["best_full_weights_sha256"],
            "model_path": result["model_path"],
            "model_sha256": result["full_model_sha256"],
            "training_result_sha256": sha256_file(result_path),
            "feature_training_equivalence_sha256": sha256_file(equivalence_path),
        }
    if reference_backbone is None:
        raise RuntimeError("No grouped model was loaded.")
    metadata["_owners"] = owners
    return reference_backbone, heads, metadata


def video_features(path: Path, backbone: tf.keras.Model) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable video: {path}")
    feature_batches: list[np.ndarray] = []
    frame_batch: list[np.ndarray] = []
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
        frame_batch.append(preprocess_input(resized))
        frame_count += 1
        if len(frame_batch) == FRAME_BATCH_SIZE:
            feature_batches.append(backbone(np.stack(frame_batch), training=False).numpy())
            frame_batch.clear()
    capture.release()
    if frame_batch:
        feature_batches.append(backbone(np.stack(frame_batch), training=False).numpy())
    if not feature_batches:
        raise RuntimeError(f"No decoded frame: {path}")
    return np.concatenate(feature_batches, axis=0), frame_count


def score_video(
    row: dict[str, str],
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
    model_hashes: dict[str, str],
    cache_dir: Path,
    hard_negative: bool,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{row['sha256']}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("video_sha256") == row["sha256"]
            and cached.get("model_sha256") == model_hashes
            and cached.get("n_frames") == N_FRAMES
            and cached.get("stride_frames") == STRIDE_FRAMES
            and all(key in cached.get("scores", {}) for key in MODEL_KEYS)
        ):
            return cached

    path = resolve_video_path(row, hard_negative=hard_negative)
    if not path.exists():
        raise FileNotFoundError(path)
    actual_hash = sha256_file(path)
    if actual_hash != row["sha256"]:
        raise RuntimeError(f"Video changed since sealing: {path}: {actual_hash}")
    features, frame_count = video_features(path, backbone)
    required = N_FRAMES + (max(K_VALUES) - 1) * STRIDE_FRAMES
    padded_frames = max(0, required - len(features))
    if padded_frames:
        features = np.concatenate(
            [features, np.repeat(features[-1:], padded_frames, axis=0)], axis=0
        )
    windows = np.stack(
        [
            features[end - N_FRAMES + 1 : end + 1]
            for end in range(N_FRAMES - 1, len(features), STRIDE_FRAMES)
        ]
    )
    scores = {
        key: [float(value) for value in head.predict(windows, batch_size=64, verbose=0).reshape(-1)]
        for key, head in heads.items()
    }
    payload = {
        "dataset": row.get("dataset", "hard_negative_v2"),
        "relative_path": row["relative_path"],
        "category": row.get("category", ""),
        "video_sha256": row["sha256"],
        "label": int(row["label"]),
        "model_sha256": model_hashes,
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "original_frame_count": frame_count,
        "padded_frames": padded_frames,
        "scores": scores,
    }
    write_json(cache_path, payload)
    return payload


def score_partition(
    name: str,
    rows: list[dict[str, str]],
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
    model_metadata: dict[str, Any],
    cache_dir: Path,
    hard_negative: bool = False,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = cache_dir.parent / f"{name}_scoring_status.json"
    model_hashes = {key: str(model_metadata[key]["weights_sha256"]) for key in MODEL_KEYS}
    records = []
    try:
        for index, row in enumerate(rows, start=1):
            records.append(
                score_video(row, backbone, heads, model_hashes, cache_dir, hard_negative)
            )
            write_json(
                status_path,
                {
                    "status": "running",
                    "partition": name,
                    "pid": os.getpid(),
                    "completed": index,
                    "total": len(rows),
                    "current_video": row["relative_path"],
                    "updated_utc": utc_now(),
                },
            )
            if index % 10 == 0 or index == len(rows):
                print(f"{name}: {index}/{len(rows)}", flush=True)
    except BaseException as error:
        write_json(
            status_path,
            {
                "status": "error",
                "partition": name,
                "completed": len(records),
                "total": len(rows),
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": utc_now(),
            },
        )
        raise
    write_json(
        status_path,
        {
            "status": "complete",
            "partition": name,
            "completed": len(rows),
            "total": len(rows),
            "completed_utc": utc_now(),
        },
    )
    return records


def records_for_model(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {
            "dataset": record["dataset"],
            "relative_path": record["relative_path"],
            "category": record.get("category", ""),
            "video_sha256": record["video_sha256"],
            "label": record["label"],
            "original_frame_count": record["original_frame_count"],
            "padded_frames": record["padded_frames"],
            "scores": record["scores"][key],
        }
        for record in records
    ]


def rolling_means(scores: list[float], k_value: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.convolve(values, np.ones(k_value) / k_value, mode="valid")


def evaluate_grid(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    truth = np.asarray([int(record["label"]) for record in records], dtype=int)
    results = []
    for k_value in K_VALUES:
        maxima = np.asarray(
            [float(np.max(rolling_means(record["scores"], k_value))) for record in records]
        )
        for threshold in THRESHOLDS:
            predictions = (maxima >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
            results.append(
                {
                    "k": k_value,
                    "threshold": threshold,
                    "n_validation": len(truth),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "accuracy": float(accuracy_score(truth, predictions)),
                    "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
                    "precision": float(precision_score(truth, predictions, zero_division=0)),
                    "recall": float(recall_score(truth, predictions, zero_division=0)),
                    "f1": float(f1_score(truth, predictions, zero_division=0)),
                    "false_positive_rate": float(fp / max(tn + fp, 1)),
                    "false_negative_rate": float(fn / max(tp + fn, 1)),
                }
            )
    return results


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["f1"]),
        float(row["balanced_accuracy"]),
        float(row["recall"]),
        -float(row["false_positive_rate"]),
        -float(row["k"]),
        -abs(float(row["threshold"]) - 0.50),
    )


def binary_metrics(
    truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "n": int(len(truth)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tp + tn) / max(len(truth), 1)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-15)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
    }

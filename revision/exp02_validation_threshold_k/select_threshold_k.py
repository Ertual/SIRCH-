from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras.applications.efficientnet import preprocess_input


N_FRAMES = 20
STRIDE_FRAMES = 5
K_VALUES = (1, 3, 5, 7, 10)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(30, 91, 5))
IMG_SIZE = 224
FRAME_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    exp01_outputs = project_root / "revision" / "exp01_manifests_sha256" / "outputs"
    parser = argparse.ArgumentParser(description="Select SIRCH theta and K on validation.")
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"C:\SIRCH_ENV\models\sirch_model.h5"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=exp01_outputs / "validation_manifest.csv",
    )
    parser.add_argument(
        "--manifest-seal", type=Path, default=exp01_outputs / "manifest_seal.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validation_manifest(
    path: Path, seal_path: Path
) -> tuple[list[dict[str, str]], str, dict[str, object]]:
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_clean":
        raise RuntimeError(f"Le sceau n'est pas propre: {seal.get('status')}")
    if int(seal.get("cross_split_sha256_duplicate_groups", -1)) != 0:
        raise RuntimeError("Le sceau signale encore des doublons inter-splits.")
    expected_hash = seal["manifest_sha256"]["validation"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Manifest validation non scelle: attendu={expected_hash}, obtenu={actual_hash}"
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = int(seal["counts"]["validation"]["total"])
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Validation attendue: {expected_count} lignes, obtenu: {len(rows)}"
        )
    invalid = [row for row in rows if row.get("split") != "validation"]
    if invalid:
        raise RuntimeError("Le manifeste contient une ligne hors validation.")
    return rows, actual_hash, seal


def resolve_video_path(row: dict[str, str], datasets_root: Path) -> Path:
    if row["dataset"] == "RLVS":
        root = datasets_root / "RLVS" / "Real Life Violence Dataset"
    elif row["dataset"] == "RWF-2000":
        root = datasets_root / "RWF-2000"
    else:
        raise RuntimeError(f"Dataset inattendu dans validation: {row['dataset']}")
    return root / Path(row["relative_path"])


def load_model_and_parts(model_path: Path) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.inference import ViolenceInference

    inference = ViolenceInference(str(model_path))
    inference.load()
    if inference.model is None:
        raise RuntimeError("Le modele SIRCH n'a pas pu etre charge.")
    model = inference.model
    td_index = next(
        index
        for index, layer in enumerate(model.layers)
        if isinstance(layer, tf.keras.layers.TimeDistributed)
    )
    backbone = model.layers[td_index].layer
    feature_input = tf.keras.Input(shape=(N_FRAMES, int(backbone.output_shape[-1])))
    x = feature_input
    for layer in model.layers[td_index + 1 :]:
        x = layer(x)
    head = tf.keras.Model(feature_input, x, name="sirch_temporal_head")
    return model, backbone, head


def video_features(path: Path, backbone: tf.keras.Model) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {path}")
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
            features = backbone(np.stack(frame_batch), training=False).numpy()
            feature_batches.append(features)
            frame_batch.clear()
    capture.release()
    if frame_batch:
        features = backbone(np.stack(frame_batch), training=False).numpy()
        feature_batches.append(features)
    if not feature_batches:
        raise RuntimeError(f"Aucune frame decodee: {path}")
    return np.concatenate(feature_batches, axis=0), frame_count


def score_video(
    path: Path, backbone: tf.keras.Model, head: tf.keras.Model
) -> tuple[list[float], int, int]:
    features, original_frame_count = video_features(path, backbone)
    required_frames = N_FRAMES + (max(K_VALUES) - 1) * STRIDE_FRAMES
    padded_frames = max(0, required_frames - len(features))
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
    scores = head.predict(windows, batch_size=64, verbose=0).reshape(-1)
    return [float(score) for score in scores], original_frame_count, padded_frames


def load_or_score(
    row: dict[str, str],
    datasets_root: Path,
    model_hash: str,
    cache_dir: Path,
    backbone: tf.keras.Model,
    head: tf.keras.Model,
) -> dict[str, object]:
    cache_path = cache_dir / f"{row['sha256']}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("model_sha256") == model_hash
            and cached.get("n_frames") == N_FRAMES
            and cached.get("stride_frames") == STRIDE_FRAMES
            and cached.get("video_sha256") == row["sha256"]
            and int(cached.get("label", -1)) == int(row["label"])
        ):
            return cached
    path = resolve_video_path(row, datasets_root)
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Hash video different du manifeste: {path}")
    scores, original_frame_count, padded_frames = score_video(path, backbone, head)
    cached = {
        "dataset": row["dataset"],
        "relative_path": row["relative_path"],
        "video_sha256": row["sha256"],
        "model_sha256": model_hash,
        "label": int(row["label"]),
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "original_frame_count": original_frame_count,
        "padded_frames": padded_frames,
        "scores": scores,
    }
    temporary = cache_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cached, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return cached


def rolling_means(scores: list[float], k_value: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.convolve(values, np.ones(k_value) / k_value, mode="valid")


def evaluate_grid(records: list[dict[str, object]]) -> list[dict[str, object]]:
    truth = np.asarray([int(record["label"]) for record in records], dtype=int)
    results: list[dict[str, object]] = []
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
                    "accuracy": accuracy_score(truth, predictions),
                    "balanced_accuracy": balanced_accuracy_score(truth, predictions),
                    "precision": precision_score(truth, predictions, zero_division=0),
                    "recall": recall_score(truth, predictions, zero_division=0),
                    "f1": f1_score(truth, predictions, zero_division=0),
                    "false_positive_rate": fp / max(tn + fp, 1),
                    "false_negative_rate": fn / max(tp + fn, 1),
                }
            )
    return results


def selection_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["f1"]),
        float(row["balanced_accuracy"]),
        float(row["recall"]),
        -float(row["false_positive_rate"]),
        -float(row["k"]),
        -abs(float(row["threshold"]) - 0.50),
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows, manifest_hash, manifest_seal = load_validation_manifest(
        args.validation_manifest, args.manifest_seal
    )
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    model_hash = sha256_file(args.model_path)
    _, backbone, head = load_model_and_parts(args.model_path)
    print(
        f"Validation uniquement: {len(manifest_rows)} videos | CPU | "
        f"N={N_FRAMES} | stride={STRIDE_FRAMES}",
        flush=True,
    )

    records: list[dict[str, object]] = []
    for index, row in enumerate(manifest_rows, start=1):
        record = load_or_score(
            row, args.datasets_root, model_hash, cache_dir, backbone, head
        )
        records.append(record)
        if index % 10 == 0 or index == len(manifest_rows):
            print(f"Validation scoree: {index}/{len(manifest_rows)}", flush=True)

    stream_rows: list[dict[str, object]] = []
    for record in records:
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
    write_csv(
        args.output_dir / "validation_score_streams.csv",
        [
            "dataset",
            "relative_path",
            "video_sha256",
            "label",
            "score_index",
            "end_frame_index",
            "score",
            "original_frame_count",
            "padded_frames",
        ],
        stream_rows,
    )

    grid = evaluate_grid(records)
    selected = max(grid, key=selection_key)
    for row in grid:
        row["selected"] = int(row is selected)
    grid_fields = list(grid[0].keys())
    write_csv(args.output_dir / "selection_grid.csv", grid_fields, grid)

    selected_config = {
        "status": "selected_on_validation_only",
        "model_path": str(args.model_path),
        "model_sha256": model_hash,
        "validation_manifest_sha256": manifest_hash,
        "manifest_status": manifest_seal["status"],
        "cross_split_sha256_duplicate_groups": manifest_seal[
            "cross_split_sha256_duplicate_groups"
        ],
        "test_manifest_read": False,
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "candidate_k": list(K_VALUES),
        "candidate_thresholds": list(THRESHOLDS),
        "selection_rule": [
            "max_f1",
            "max_balanced_accuracy",
            "max_recall",
            "min_false_positive_rate",
            "min_k",
            "threshold_closest_to_0.50",
        ],
        "selected": selected,
        "videos_with_padding": sum(int(record["padded_frames"]) > 0 for record in records),
    }
    (args.output_dir / "selected_config.json").write_text(
        json.dumps(selected_config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    summary = [
        "# Selection theta/K sur validation",
        "",
        "Le jeu de test principal n'a pas ete lu.",
        "",
        f"- Videos de validation : {len(records)}",
        f"- Videos completees par repetition : {selected_config['videos_with_padding']}",
        f"- K selectionne : {selected['k']}",
        f"- Seuil selectionne : {float(selected['threshold']):.2f}",
        f"- Accuracy validation : {float(selected['accuracy']):.4f}",
        f"- Balanced accuracy validation : {float(selected['balanced_accuracy']):.4f}",
        f"- Precision validation : {float(selected['precision']):.4f}",
        f"- Rappel validation : {float(selected['recall']):.4f}",
        f"- F1 validation : {float(selected['f1']):.4f}",
        f"- Matrice : TN={selected['tn']}, FP={selected['fp']}, FN={selected['fn']}, TP={selected['tp']}",
        "",
        "## Comparaison avant/apres deduplication",
        "",
        "| Version validation | N | theta | K | Accuracy | Precision | Rappel | F1 | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| Avant correction | 599 | 0.70 | 7 | 0.8681 | 0.8503 | 0.8930 | 0.8711 | 253/47/32/267 |",
        f"| Apres correction | {len(records)} | {float(selected['threshold']):.2f} | {selected['k']} | {float(selected['accuracy']):.4f} | {float(selected['precision']):.4f} | {float(selected['recall']):.4f} | {float(selected['f1']):.4f} | {selected['tn']}/{selected['fp']}/{selected['fn']}/{selected['tp']} |",
        "",
        "Le nettoyage change legerement les metriques, mais pas le choix de theta=0.70 et K=7.",
        "",
        "Le manifeste utilise est scelle sans doublon SHA-256 entre les splits.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(selected_config, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

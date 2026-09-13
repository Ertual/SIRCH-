from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from gru_factorial_common import (
    AUGMENTED_WEIGHTS_PATH,
    AUGMENTED_WEIGHTS_SHA256,
    EXPECTED_TEST_COUNT,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXP01_OUTPUTS,
    EXPERIMENT_DIR,
    MODEL_KEYS,
    N_FRAMES,
    ORIGINAL_MODEL_PATH,
    ORIGINAL_MODEL_SHA256,
    SOURCE_BEST_EPOCH,
    STRIDE_FRAMES,
    load_or_score_pair,
    load_shared_backbone_and_heads,
    records_for_model,
    sha256_file,
)
from select_threshold_k import rolling_means


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_metrics(
    truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "false_negative_rate": float(fn / max(tp + fn, 1)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
    }


def save_confusion_matrix(
    metrics: dict[str, float | int], title: str, path: Path
) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_yticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Verite terrain")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_root = EXPERIMENT_DIR / "outputs" / "final_test_locked"
    output_root.mkdir(parents=True, exist_ok=True)
    joint_result_path = output_root / "joint_test_evaluation.json"
    if joint_result_path.exists():
        raise RuntimeError(
            "Les evaluations finales GRU sont deja publiees; une seconde execution est interdite."
        )

    protocol_path = EXPERIMENT_DIR / "locked_test_protocols.json"
    if not protocol_path.exists():
        raise RuntimeError("Les configurations doivent etre verrouillees avant le test.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_final_test":
        raise RuntimeError("Statut du protocole verrouille invalide.")
    if protocol.get("no_post_test_adjustment") is not True:
        raise RuntimeError("Le protocole n'interdit pas les ajustements post-test.")

    selected = {}
    for key in MODEL_KEYS:
        config_path = EXPERIMENT_DIR / "outputs" / key / "selected_config.json"
        locked = protocol["selected_configurations"][key]
        if sha256_file(config_path) != locked["selected_config_sha256"]:
            raise RuntimeError(f"La configuration {key} a change apres verrouillage.")
        selected[key] = json.loads(config_path.read_text(encoding="utf-8"))
        chosen = selected[key]["selected"]
        if (
            float(chosen["threshold"]) != float(locked["theta"])
            or int(chosen["k"]) != int(locked["k"])
        ):
            raise RuntimeError(f"Theta/K divergent pour {key}.")

    if sha256_file(ORIGINAL_MODEL_PATH) != ORIGINAL_MODEL_SHA256:
        raise RuntimeError("Le modele GRU original a change.")
    if sha256_file(AUGMENTED_WEIGHTS_PATH) != AUGMENTED_WEIGHTS_SHA256:
        raise RuntimeError("Les meilleurs poids GRU enrichis de l'epoch 5 ont change.")
    augmented_meta = protocol["models"]["gru_augmented"]
    if (
        int(augmented_meta["source_best_epoch"]) != SOURCE_BEST_EPOCH
        or augmented_meta.get("epoch_11_checkpoint_used") is not False
    ):
        raise RuntimeError("Le protocole enrichi ne verrouille pas exclusivement l'epoch 5.")

    manifest_path = EXP01_OUTPUTS / "test_manifest.csv"
    if sha256_file(manifest_path) != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le manifeste de test ne correspond pas au verrou.")
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"Test attendu: {EXPECTED_TEST_COUNT}, obtenu: {len(rows)}")
    if any(row.get("split") != "test" for row in rows):
        raise RuntimeError("Le manifeste contient une ligne hors test.")
    labels = [int(row["label"]) for row in rows]
    if labels.count(1) != 297 or labels.count(0) != 295:
        raise RuntimeError("La repartition des classes du test a change.")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Un GPU est visible alors que le protocole exige CPU seul.")

    state_path = output_root / "evaluation_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in {"running", "interrupted"}:
            raise RuntimeError(f"Etat de reprise inattendu: {state.get('status')}")
    else:
        state = {
            "status": "running",
            "logical_run_started_utc": utc_now(),
            "attempts": 0,
            "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
            "protocol_sha256": sha256_file(protocol_path),
        }
    state["attempts"] = int(state["attempts"]) + 1
    state["last_attempt_started_utc"] = utc_now()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    cache_dir = output_root / "shared_test_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone, heads = load_shared_backbone_and_heads()
    print(
        "TEST FINAL UNIQUE GRU: 592 videos | configurations validation verrouillees | "
        "poids enrichis epoch 5",
        flush=True,
    )
    datasets_root = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
    paired_records = []
    try:
        for index, row in enumerate(rows, start=1):
            paired_records.append(
                load_or_score_pair(row, datasets_root, cache_dir, backbone, heads)
            )
            if index % 10 == 0 or index == len(rows):
                print(f"Test final GRU score: {index}/{len(rows)}", flush=True)
    except BaseException:
        state["status"] = "interrupted"
        state["updated_utc"] = utc_now()
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        raise

    completed_utc = utc_now()
    results = {}
    for key in MODEL_KEYS:
        model_output = output_root / key
        model_output.mkdir(parents=True, exist_ok=True)
        records = records_for_model(paired_records, key)
        locked = protocol["selected_configurations"][key]
        theta = float(locked["theta"])
        k_value = int(locked["k"])
        truth = np.asarray([int(record["label"]) for record in records], dtype=int)
        video_scores = np.asarray(
            [float(np.max(rolling_means(record["scores"], k_value))) for record in records]
        )
        predictions = (video_scores >= theta).astype(int)
        metrics = calculate_metrics(truth, predictions, video_scores)

        prediction_rows = []
        for record, score, prediction in zip(records, video_scores, predictions):
            prediction_rows.append(
                {
                    "dataset": record["dataset"],
                    "relative_path": record["relative_path"],
                    "video_sha256": record["video_sha256"],
                    "label": record["label"],
                    "video_score": f"{score:.10f}",
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
        with predictions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(prediction_rows)

        title = (
            f"{key.replace('_', ' ')} - test verrouille "
            f"(theta={theta:.2f}; K={k_value})"
        )
        save_confusion_matrix(metrics, title, model_output / "confusion_matrix.png")
        result = {
            "status": "final_locked_test_complete",
            "logical_run_started_utc": state["logical_run_started_utc"],
            "completed_utc": completed_utc,
            "no_post_test_adjustment": True,
            "experiment": key,
            "selected_config_sha256": locked["selected_config_sha256"],
            "protocol_sha256": sha256_file(protocol_path),
            "test_manifest_path": str(manifest_path),
            "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
            "theta": theta,
            "k": k_value,
            "n_frames": N_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "videos_with_padding": sum(int(record["padded_frames"]) > 0 for record in records),
            "metrics": metrics,
        }
        if key == "gru_original":
            result.update(
                {"model_path": str(ORIGINAL_MODEL_PATH), "model_sha256": ORIGINAL_MODEL_SHA256}
            )
        else:
            result.update(
                {
                    "weights_path": str(AUGMENTED_WEIGHTS_PATH),
                    "weights_sha256": AUGMENTED_WEIGHTS_SHA256,
                    "source_best_epoch": SOURCE_BEST_EPOCH,
                    "epoch_11_checkpoint_used": False,
                }
            )
        result_path = model_output / "final_test_metrics.json"
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        results[key] = result

        summary = [
            f"# Evaluation finale unique - {key}",
            "",
            f"Configuration verrouillee sur validation : theta={theta:.2f}, K={k_value}.",
            "Aucun parametre n'a ete ajuste apres lecture du test.",
            "",
            f"- Videos : {metrics['n']} (297 violence, 295 non-violence)",
            f"- Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f} %)",
            f"- Balanced accuracy : {metrics['balanced_accuracy']:.4f}",
            f"- Precision : {metrics['precision']:.4f}",
            f"- Rappel : {metrics['recall']:.4f}",
            f"- Specificite : {metrics['specificity']:.4f}",
            f"- F1 : {metrics['f1']:.4f}",
            f"- ROC-AUC : {metrics['roc_auc']:.4f}",
            f"- PR-AUC : {metrics['pr_auc']:.4f}",
            f"- Matrice : TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
        ]
        if key == "gru_augmented":
            summary.insert(4, "Poids utilises : best de l'epoch 5; checkpoint epoch 11 exclu.")
        (model_output / "summary.md").write_text(
            "\n".join(summary) + "\n", encoding="utf-8"
        )

    joint = {
        "status": "joint_final_locked_test_complete",
        "logical_run_started_utc": state["logical_run_started_utc"],
        "completed_utc": completed_utc,
        "same_video_decode_and_backbone_pass": True,
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "n": EXPECTED_TEST_COUNT,
        "results": {
            key: {
                "theta": results[key]["theta"],
                "k": results[key]["k"],
                "metrics": results[key]["metrics"],
                "result_sha256": sha256_file(
                    output_root / key / "final_test_metrics.json"
                ),
                "predictions_sha256": sha256_file(
                    output_root / key / "video_predictions.csv"
                ),
            }
            for key in MODEL_KEYS
        },
    }
    temporary = joint_result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(joint, indent=2) + "\n", encoding="utf-8")
    temporary.replace(joint_result_path)
    state["status"] = "complete"
    state["completed_utc"] = completed_utc
    state["joint_result_sha256"] = sha256_file(joint_result_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(joint, indent=2), flush=True)


if __name__ == "__main__":
    main()

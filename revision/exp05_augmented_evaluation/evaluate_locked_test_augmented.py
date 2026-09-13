from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXP02_DIR = PROJECT_ROOT / "revision" / "exp02_validation_threshold_k"
sys.path.insert(0, str(EXP02_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

from select_threshold_k import (  # noqa: E402
    N_FRAMES,
    STRIDE_FRAMES,
    load_or_score,
    rolling_means,
    sha256_file,
)
from select_threshold_k_augmented import (  # noqa: E402
    EXPECTED_WEIGHTS_SHA256,
    WEIGHTS_PATH,
    load_augmented_model_parts,
)


EXPECTED_TEST_COUNT = 592
EXPECTED_TEST_MANIFEST_SHA256 = (
    "13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f"
)


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


def save_confusion_matrix(metrics: dict, theta: float, k_value: int, path: Path) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    image = axis.imshow(matrix, cmap="Greens")
    axis.set_xticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_yticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Verite terrain")
    axis.set_title(f"SIRCH enrichi - test verrouille (theta={theta:.2f}; K={k_value})")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_dir = EXPERIMENT_DIR / "outputs" / "final_test_locked"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_test_metrics.json"
    if final_path.exists():
        raise RuntimeError(
            "Evaluation finale enrichie deja publiee. Le protocole interdit une seconde execution."
        )

    protocol_path = EXPERIMENT_DIR / "locked_test_protocol.json"
    selected_path = EXPERIMENT_DIR / "outputs" / "selected_config.json"
    if not protocol_path.exists():
        raise RuntimeError("Le protocole doit etre verrouille avant l'ouverture du test.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "preregistered_before_final_test":
        raise RuntimeError("Statut du protocole verrouille invalide.")
    if protocol["selected_configuration"]["sha256"] != sha256_file(selected_path):
        raise RuntimeError("La configuration selectionnee a change apres verrouillage.")
    if protocol["model"]["weights_sha256"] != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("Le protocole ne verrouille pas les poids de l'epoch 6.")
    if sha256_file(WEIGHTS_PATH) != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("Les poids de l'epoch 6 sont absents ou ont change.")

    theta = float(protocol["selected_configuration"]["theta"])
    k_value = int(protocol["selected_configuration"]["k"])
    if theta != float(selected["selected"]["threshold"]) or k_value != int(
        selected["selected"]["k"]
    ):
        raise RuntimeError("Theta/K divergent entre la selection et le verrou.")

    manifest_path = (
        PROJECT_ROOT
        / "revision"
        / "exp01_manifests_sha256"
        / "outputs"
        / "test_manifest.csv"
    )
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

    state_path = output_dir / "evaluation_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") not in {"running", "interrupted"}:
            raise RuntimeError(f"Etat de reprise inattendu: {state.get('status')}")
    else:
        state = {
            "status": "running",
            "logical_run_started_utc": utc_now(),
            "attempts": 0,
            "theta": theta,
            "k": k_value,
            "n_frames": N_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        }
    state["attempts"] = int(state["attempts"]) + 1
    state["last_attempt_started_utc"] = utc_now()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, backbone, head = load_augmented_model_parts()
    print(
        f"TEST FINAL UNIQUE ENRICHI: {len(rows)} videos | theta={theta:.2f} | "
        f"K={k_value} | poids epoch 6",
        flush=True,
    )

    datasets_root = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
    records: list[dict[str, object]] = []
    try:
        for index, row in enumerate(rows, start=1):
            records.append(
                load_or_score(
                    row,
                    datasets_root,
                    EXPECTED_WEIGHTS_SHA256,
                    cache_dir,
                    backbone,
                    head,
                )
            )
            if index % 10 == 0 or index == len(rows):
                print(f"Test final enrichi score: {index}/{len(rows)}", flush=True)
    except BaseException:
        state["status"] = "interrupted"
        state["updated_utc"] = utc_now()
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        raise

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
    predictions_path = output_dir / "video_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)

    save_confusion_matrix(metrics, theta, k_value, output_dir / "confusion_matrix.png")
    result = {
        "status": "final_locked_test_complete",
        "logical_run_started_utc": state["logical_run_started_utc"],
        "completed_utc": utc_now(),
        "no_post_test_adjustment": True,
        "weights_path": str(WEIGHTS_PATH),
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "source_best_epoch": 6,
        "epoch_12_checkpoint_used": False,
        "selected_config_sha256": sha256_file(selected_path),
        "protocol_sha256": sha256_file(protocol_path),
        "test_manifest_path": str(manifest_path),
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "theta": theta,
        "k": k_value,
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "videos_with_padding": sum(
            int(record["padded_frames"]) > 0 for record in records
        ),
        "metrics": metrics,
    }
    temporary = final_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(final_path)

    summary = [
        "# Evaluation finale unique du modele LSTM enrichi",
        "",
        f"Configuration verrouillee sur validation : theta={theta:.2f}, K={k_value}, N=20, stride=5.",
        "Aucun parametre n'a ete ajuste apres lecture du test.",
        "Les poids utilises sont ceux de l'epoch 6, restaures par EarlyStopping.",
        "",
        f"- Videos : {metrics['n']} (297 violence, 295 non-violence)",
        f"- Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f} %)",
        f"- Balanced accuracy : {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy'] * 100:.2f} %)",
        f"- Precision : {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f} %)",
        f"- Rappel : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f} %)",
        f"- Specificite : {metrics['specificity']:.4f} ({metrics['specificity'] * 100:.2f} %)",
        f"- F1 : {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f} %)",
        f"- ROC-AUC : {metrics['roc_auc']:.4f}",
        f"- PR-AUC : {metrics['pr_auc']:.4f}",
        f"- Matrice : TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    state["status"] = "complete"
    state["completed_utc"] = result["completed_utc"]
    state["final_result_sha256"] = sha256_file(final_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

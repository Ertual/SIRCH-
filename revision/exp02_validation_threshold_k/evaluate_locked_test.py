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

from select_threshold_k import (
    N_FRAMES,
    STRIDE_FRAMES,
    load_model_and_parts,
    load_or_score,
    rolling_means,
    sha256_file,
)


LOCKED_THRESHOLD = 0.70
LOCKED_K = 7
EXPECTED_TEST_COUNT = 592
EXPECTED_TEST_MANIFEST_SHA256 = (
    "13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f"
)
EXPECTED_MODEL_SHA256 = (
    "9d63a86c78f042dea0d12fe1c880f07f57d13eed95abdd4773aef672982615ec"
)
EXPECTED_CONFIG_SHA256 = (
    "084c8bf4d66339833e6d77d9f76cbe4a89b6f64f0588d0d4023912f27d4aa644"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_locked_inputs(project_root: Path) -> tuple[list[dict[str, str]], dict, Path, Path]:
    exp01 = project_root / "revision" / "exp01_manifests_sha256" / "outputs"
    exp02 = project_root / "revision" / "exp02_validation_threshold_k"
    manifest_path = exp01 / "test_manifest.csv"
    seal_path = exp01 / "manifest_seal.json"
    selected_path = exp02 / "outputs" / "selected_config.json"
    protocol_path = exp02 / "locked_test_protocol.json"

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_clean":
        raise RuntimeError(f"Sceau non propre: {seal.get('status')}")
    if int(seal.get("cross_split_sha256_duplicate_groups", -1)) != 0:
        raise RuntimeError("Le sceau signale encore des doublons inter-splits.")
    if seal["manifest_sha256"]["test"] != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le hash du test dans le sceau ne correspond pas au protocole verrouille.")
    if sha256_file(manifest_path) != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le manifeste de test ne correspond pas au protocole verrouille.")
    if sha256_file(selected_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("La configuration selectionnee a change depuis son verrouillage.")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "selected_on_validation_only":
        raise RuntimeError("La configuration n'a pas ete selectionnee uniquement sur validation.")
    if selected.get("test_manifest_read") is not False:
        raise RuntimeError("La selection indique que le test a ete consulte.")
    chosen = selected["selected"]
    if int(chosen["k"]) != LOCKED_K or float(chosen["threshold"]) != LOCKED_THRESHOLD:
        raise RuntimeError("La configuration verrouillee theta=0.70, K=7 a change.")
    if int(selected["n_frames"]) != N_FRAMES or int(selected["stride_frames"]) != STRIDE_FRAMES:
        raise RuntimeError("N_FRAMES ou stride different du protocole de validation.")

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TEST_COUNT:
        raise RuntimeError(f"Test attendu: {EXPECTED_TEST_COUNT}, obtenu: {len(rows)}")
    if any(row.get("split") != "test" for row in rows):
        raise RuntimeError("Le manifeste contient une ligne hors test.")
    labels = [int(row["label"]) for row in rows]
    if labels.count(1) != 297 or labels.count(0) != 295:
        raise RuntimeError("La repartition des classes du test a change.")
    return rows, selected, manifest_path, protocol_path


def calculate_metrics(truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
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


def save_confusion_matrix(metrics: dict, path: Path) -> None:
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6.4, 5.4), dpi=180)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_yticks([0, 1], labels=["Non-violence", "Violence"])
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Verite terrain")
    axis.set_title("SIRCH - test final verrouille (theta=0,70; K=7)")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=15)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = Path(__file__).resolve().parent / "outputs" / "final_test_locked"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_test_metrics.json"
    if final_path.exists():
        raise RuntimeError(
            "Evaluation finale deja publiee. Le protocole interdit une seconde execution."
        )

    rows, selected, manifest_path, protocol_path = load_locked_inputs(project_root)
    model_path = Path(selected["model_path"])
    if not model_path.exists() or sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Le modele original verrouille est absent ou a change.")

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
            "theta": LOCKED_THRESHOLD,
            "k": LOCKED_K,
            "n_frames": N_FRAMES,
            "stride_frames": STRIDE_FRAMES,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        }
    state["attempts"] = int(state["attempts"]) + 1
    state["last_attempt_started_utc"] = utc_now()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, backbone, head = load_model_and_parts(model_path)
    print(
        f"TEST FINAL UNIQUE: {len(rows)} videos | theta={LOCKED_THRESHOLD:.2f} | "
        f"K={LOCKED_K} | N={N_FRAMES} | stride={STRIDE_FRAMES}",
        flush=True,
    )

    records: list[dict] = []
    for index, row in enumerate(rows, start=1):
        record = load_or_score(
            row, Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
            EXPECTED_MODEL_SHA256, cache_dir, backbone, head
        )
        records.append(record)
        if index % 10 == 0 or index == len(rows):
            print(f"Test final score: {index}/{len(rows)}", flush=True)

    truth = np.asarray([int(record["label"]) for record in records], dtype=int)
    video_scores = np.asarray(
        [float(np.max(rolling_means(record["scores"], LOCKED_K))) for record in records]
    )
    predictions = (video_scores >= LOCKED_THRESHOLD).astype(int)
    metrics = calculate_metrics(truth, predictions, video_scores)

    prediction_rows = []
    for record, score, prediction in zip(records, video_scores, predictions):
        prediction_rows.append(
            {
                "dataset": record["dataset"],
                "relative_path": record["relative_path"],
                "video_sha256": record["video_sha256"],
                "label": record["label"],
                "max_rolling_mean_k7": f"{score:.10f}",
                "threshold": f"{LOCKED_THRESHOLD:.2f}",
                "prediction": int(prediction),
                "correct": int(int(record["label"]) == int(prediction)),
                "sequence_scores": len(record["scores"]),
                "original_frame_count": record["original_frame_count"],
                "padded_frames": record["padded_frames"],
            }
        )
    predictions_path = output_dir / "video_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    save_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    result = {
        "status": "final_locked_test_complete",
        "logical_run_started_utc": state["logical_run_started_utc"],
        "completed_utc": utc_now(),
        "no_post_test_adjustment": True,
        "model_path": str(model_path),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "selected_config_sha256": EXPECTED_CONFIG_SHA256,
        "protocol_sha256": sha256_file(protocol_path),
        "test_manifest_path": str(manifest_path),
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "theta": LOCKED_THRESHOLD,
        "k": LOCKED_K,
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "videos_with_padding": sum(int(record["padded_frames"]) > 0 for record in records),
        "metrics": metrics,
    }
    temporary = final_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(final_path)

    summary = [
        "# Evaluation finale unique du test nettoye",
        "",
        "Configuration verrouillee sur validation : theta=0,70, K=7, N=20, stride=5.",
        "Aucun parametre n'a ete ajuste apres lecture du test.",
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
        "",
        "Ce resultat est l'unique evaluation finale prevue par le protocole verrouille.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    state["status"] = "complete"
    state["completed_utc"] = result["completed_utc"]
    state["final_result_sha256"] = sha256_file(final_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

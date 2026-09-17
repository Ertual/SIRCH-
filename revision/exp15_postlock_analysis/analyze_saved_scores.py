from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TEST = ROOT / "revision" / "exp14_grouped_factorial_evaluation" / "outputs" / "final_test_locked"
CACHE = Path(r"C:\SIRCH_ENV\models\revision\exp14_grouped_factorial_evaluation\test_cache")
MODELS = (
    "lstm_original_grouped",
    "lstm_augmented_grouped",
    "gru_original_grouped",
    "gru_augmented_grouped",
)
EXPECTED_TEST_SHA256 = "a37aa9cba85d28ac315c1ea813c902330a3e99525042c13012023914e0a9887a"
N_FRAMES = 20
STRIDE = 5
ASSUMED_FPS = 25.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rolling(scores: list[float], k: int) -> list[float]:
    return [statistics.fmean(scores[i - k + 1 : i + 1]) for i in range(k - 1, len(scores))]


def summarize_delay(rows: list[dict]) -> dict:
    detected = [float(row["delay_seconds_at_25fps"]) for row in rows if row["detected"]]
    detected.sort()
    return {
        "violent_videos": len(rows),
        "detected": len(detected),
        "missed": len(rows) - len(detected),
        "mean_seconds_detected_only": statistics.fmean(detected) if detected else None,
        "median_seconds_detected_only": statistics.median(detected) if detected else None,
        "std_seconds_detected_only": statistics.pstdev(detected) if detected else None,
        "p95_seconds_detected_only": detected[math.ceil(0.95 * len(detected)) - 1] if detected else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = {model: read_csv(TEST / model / "video_predictions.csv") for model in MODELS}
    metrics = {model: json.loads((TEST / model / "final_test_metrics.json").read_text(encoding="utf-8")) for model in MODELS}
    first = predictions[MODELS[0]]
    if len(first) != 597 or sum(int(row["label"]) for row in first) != 297:
        raise RuntimeError("Unexpected locked test composition")
    keys = [(row["dataset"], row["relative_path"], row["video_sha256"]) for row in first]
    for model in MODELS:
        rows = predictions[model]
        if [(r["dataset"], r["relative_path"], r["video_sha256"]) for r in rows] != keys:
            raise RuntimeError(f"Test row order differs for {model}")
        if metrics[model]["test_manifest_sha256"] != EXPECTED_TEST_SHA256:
            raise RuntimeError(f"Test manifest seal differs for {model}")

    pr_rows = []
    for model in MODELS:
        rows = predictions[model]
        y_true = [int(row["label"]) for row in rows]
        scores = [float(row["video_score"]) for row in rows]
        ap = average_precision_score(y_true, scores)
        archived = float(metrics[model]["metrics"]["pr_auc"])
        if abs(ap - archived) > 1e-9:
            raise RuntimeError(f"Archived PR-AUC mismatch for {model}: {ap} vs {archived}")
        pr_rows.append({
            "model": model, "test_n": len(rows), "positive_n": sum(y_true),
            "theta": metrics[model]["theta"], "k": metrics[model]["k"],
            "average_precision_pr_auc": f"{ap:.12f}",
            "archived_pr_auc": f"{archived:.12f}",
            "test_manifest_sha256": EXPECTED_TEST_SHA256,
        })
    write_csv(OUT / "pr_auc_locked_test.csv", pr_rows)

    delay_rows = []
    missing_cache = []
    for row_index, row in enumerate(first):
        if int(row["label"]) != 1:
            continue
        path = CACHE / f"{row['video_sha256']}.json"
        if not path.exists():
            missing_cache.append(str(path))
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["video_sha256"] != row["video_sha256"] or int(cached["label"]) != 1:
            raise RuntimeError(f"Invalid cached stream: {path}")
        if int(cached["n_frames"]) != N_FRAMES or int(cached["stride_frames"]) != STRIDE:
            raise RuntimeError(f"Unexpected stream geometry: {path}")
        for model in MODELS:
            expected_weight_sha = metrics[model]["model"]["weights_sha256"]
            if cached["model_sha256"][model] != expected_weight_sha:
                raise RuntimeError(f"Weight SHA mismatch for {model}: {path}")
            scores = [float(value) for value in cached["scores"][model]]
            k = int(metrics[model]["k"])
            theta = float(metrics[model]["theta"])
            means = rolling(scores, k)
            crossing = next((i + k - 1 for i, mean in enumerate(means) if mean >= theta), None)
            saved = predictions[model][row_index]
            if abs(max(means) - float(saved["video_score"])) > 1e-6:
                raise RuntimeError(f"Locked video score differs from cached stream: {model} {path}")
            if int(saved["prediction"]) != int(crossing is not None):
                raise RuntimeError(f"Locked decision differs from threshold crossing: {model} {path}")
            delay_rows.append({
                "model": model, "dataset": row["dataset"], "relative_path": row["relative_path"],
                "video_sha256": row["video_sha256"], "theta": theta, "k": k,
                "sequence_scores": len(scores), "detected": int(crossing is not None),
                "alert_score_index_zero_based": crossing if crossing is not None else "",
                "alert_frame_one_based": N_FRAMES + STRIDE * crossing if crossing is not None else "",
                "delay_seconds_at_25fps": f"{(N_FRAMES + STRIDE * crossing) / ASSUMED_FPS:.6f}" if crossing is not None else "",
                "alert_rolling_score": f"{means[crossing - k + 1]:.9f}" if crossing is not None else "",
            })
    if missing_cache:
        raise RuntimeError(f"Missing {len(missing_cache)} violent score caches, first: {missing_cache[0]}")
    write_csv(OUT / "alert_delay_by_video.csv", delay_rows)
    delay_summary = {model: summarize_delay([row for row in delay_rows if row["model"] == model]) for model in MODELS}
    if any(info["violent_videos"] != 297 for info in delay_summary.values()):
        raise RuntimeError("Not all 297 violent videos were analyzed")
    delay_payload = {
        "method": "First full K-window rolling mean at or above locked theta; stream stride 5 frames; frame count from clip start; assumed 25 fps; detected clips only in delay average.",
        "not_wall_clock_latency": True,
        "violence_onset_not_annotated": True,
        "models": delay_summary,
    }
    with (OUT / "alert_delay_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(delay_payload, handle, indent=2)
        handle.write("\n")

    error_rows = []
    errors_by_key = defaultdict(int)
    for model in MODELS:
        for row in predictions[model]:
            if int(row["correct"]) == 0:
                errors_by_key[(row["dataset"], row["relative_path"], row["video_sha256"])] += 1
    for model in MODELS:
        for row in predictions[model]:
            if int(row["correct"]) != 0:
                continue
            truth = int(row["label"])
            prediction = int(row["prediction"])
            score = float(row["video_score"])
            error_rows.append({
                "model": model, "error_type": "FP" if truth == 0 else "FN",
                "dataset": row["dataset"], "relative_path": row["relative_path"],
                "video_sha256": row["video_sha256"], "true_label": truth,
                "predicted_label": prediction, "score_violence": f"{score:.9f}",
                "confidence_predicted_class": f"{score if prediction == 1 else 1 - score:.9f}",
                "theta": row["threshold"], "k": row["k"],
                "wrong_model_count": errors_by_key[(row["dataset"], row["relative_path"], row["video_sha256"])],
            })
    write_csv(OUT / "all_locked_test_errors.csv", error_rows)
    counts = {model: {
        "fp": sum(r["model"] == model and r["error_type"] == "FP" for r in error_rows),
        "fn": sum(r["model"] == model and r["error_type"] == "FN" for r in error_rows),
    } for model in MODELS}
    print(json.dumps({"pr_auc": pr_rows, "alert_delay": delay_summary, "error_counts": counts}, indent=2))


if __name__ == "__main__":
    main()

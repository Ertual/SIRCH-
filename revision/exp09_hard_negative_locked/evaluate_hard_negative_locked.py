from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
EXP01_OUTPUTS = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
EXP02_DIR = PROJECT_ROOT / "revision" / "exp02_validation_threshold_k"
EXP05_DIR = PROJECT_ROOT / "revision" / "exp05_augmented_evaluation"
EXP08_DIR = PROJECT_ROOT / "revision" / "exp08_gru_factorial_evaluation"

for path in (EXP02_DIR, EXP05_DIR, EXP08_DIR):
    sys.path.insert(0, str(path))

from select_threshold_k import (  # noqa: E402
    K_VALUES,
    N_FRAMES,
    STRIDE_FRAMES,
    load_model_and_parts,
    rolling_means,
    sha256_file,
    video_features,
)
from select_threshold_k_augmented import (  # noqa: E402
    EXPECTED_WEIGHTS_SHA256 as LSTM_AUGMENTED_SHA256,
    WEIGHTS_PATH as LSTM_AUGMENTED_PATH,
    load_augmented_model_parts,
)
from gru_factorial_common import (  # noqa: E402
    AUGMENTED_WEIGHTS_PATH as GRU_AUGMENTED_PATH,
    AUGMENTED_WEIGHTS_SHA256 as GRU_AUGMENTED_SHA256,
    ORIGINAL_MODEL_PATH as GRU_ORIGINAL_PATH,
    ORIGINAL_MODEL_SHA256 as GRU_ORIGINAL_SHA256,
    load_shared_backbone_and_heads,
)


LSTM_ORIGINAL_PATH = Path(r"C:\SIRCH_ENV\models\sirch_model.h5")
LSTM_ORIGINAL_SHA256 = (
    "9d63a86c78f042dea0d12fe1c880f07f57d13eed95abdd4773aef672982615ec"
)
EXPECTED_MANIFEST_SHA256 = (
    "4b9c6c135ec3869e7b109f1aa5a695b3a544368b07ada753919ca226e9c76dba"
)
LOCKED_CONFIGS = {
    "lstm_original": {
        "architecture": "LSTM",
        "enrichment": "original",
        "threshold": 0.70,
        "k": 7,
        "model_path": str(LSTM_ORIGINAL_PATH),
        "model_sha256": LSTM_ORIGINAL_SHA256,
    },
    "lstm_augmented": {
        "architecture": "LSTM",
        "enrichment": "enriched",
        "threshold": 0.60,
        "k": 7,
        "model_path": str(LSTM_AUGMENTED_PATH),
        "model_sha256": LSTM_AUGMENTED_SHA256,
        "source_best_epoch": 6,
    },
    "gru_original": {
        "architecture": "GRU",
        "enrichment": "original",
        "threshold": 0.60,
        "k": 10,
        "model_path": str(GRU_ORIGINAL_PATH),
        "model_sha256": GRU_ORIGINAL_SHA256,
    },
    "gru_augmented": {
        "architecture": "GRU",
        "enrichment": "enriched",
        "threshold": 0.60,
        "k": 7,
        "model_path": str(GRU_AUGMENTED_PATH),
        "model_sha256": GRU_AUGMENTED_SHA256,
        "source_best_epoch": 5,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest() -> tuple[list[dict[str, str]], str]:
    manifest_path = EXP01_OUTPUTS / "hard_negative_v2_manifest.csv"
    seal_path = EXP01_OUTPUTS / "manifest_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = seal["manifest_sha256"]["hard_negative_v2"]
    actual = sha256_file(manifest_path)
    if expected != EXPECTED_MANIFEST_SHA256 or actual != expected:
        raise RuntimeError(
            f"Manifest hard-negative non scelle: attendu={EXPECTED_MANIFEST_SHA256}, obtenu={actual}"
        )
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30 or any(row["split"] != "hard_negative_v2" for row in rows):
        raise RuntimeError("Le corpus hard-negative doit contenir exactement 30 videos v2.")
    expected_categories = {"sport": 15, "danse": 10, "calme": 5}
    actual_categories = {
        category: sum(row["category"] == category for row in rows)
        for category in expected_categories
    }
    if actual_categories != expected_categories:
        raise RuntimeError(f"Repartition hard-negative inattendue: {actual_categories}")
    return rows, actual


def verify_model_hashes() -> None:
    for config in LOCKED_CONFIGS.values():
        path = Path(str(config["model_path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != config["model_sha256"]:
            raise RuntimeError(f"Modele modifie: {path}; obtenu={actual}")


def backbones_equal(reference: tf.keras.Model, candidate: tf.keras.Model) -> bool:
    left = reference.get_weights()
    right = candidate.get_weights()
    return len(left) == len(right) and all(
        np.array_equal(a, b) for a, b in zip(left, right)
    )


def load_shared_models() -> tuple[tf.keras.Model, dict[str, tf.keras.Model]]:
    lstm_original_model, lstm_original_backbone, lstm_original_head = (
        load_model_and_parts(LSTM_ORIGINAL_PATH)
    )
    lstm_augmented_model, lstm_augmented_backbone, lstm_augmented_head = (
        load_augmented_model_parts()
    )
    gru_backbone, gru_heads = load_shared_backbone_and_heads()

    if not backbones_equal(lstm_original_backbone, lstm_augmented_backbone):
        raise RuntimeError("Les backbones LSTM original et enrichi ne sont pas identiques.")
    if not backbones_equal(lstm_original_backbone, gru_backbone):
        raise RuntimeError("Les backbones LSTM et GRU ne sont pas identiques.")

    # Keep the owning models alive while their temporal heads are used.
    _ = (lstm_original_model, lstm_augmented_model)
    return lstm_original_backbone, {
        "lstm_original": lstm_original_head,
        "lstm_augmented": lstm_augmented_head,
        **gru_heads,
    }


def score_video(
    row: dict[str, str],
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
) -> dict[str, object]:
    cache_path = CACHE_DIR / f"{row['sha256']}.json"
    model_hashes = {
        key: str(value["model_sha256"]) for key, value in LOCKED_CONFIGS.items()
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("video_sha256") == row["sha256"]
            and cached.get("model_sha256") == model_hashes
            and cached.get("n_frames") == N_FRAMES
            and cached.get("stride_frames") == STRIDE_FRAMES
        ):
            return cached

    video_path = PROJECT_ROOT / "datasets" / Path(row["relative_path"])
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if sha256_file(video_path) != row["sha256"]:
        raise RuntimeError(f"Video modifiee depuis le scellement: {video_path}")

    features, frame_count = video_features(video_path, backbone)
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
    scores = {
        key: [float(value) for value in head.predict(windows, batch_size=64, verbose=0).reshape(-1)]
        for key, head in heads.items()
    }
    cached = {
        "relative_path": row["relative_path"],
        "category": row["category"],
        "video_sha256": row["sha256"],
        "model_sha256": model_hashes,
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "original_frame_count": frame_count,
        "padded_frames": padded_frames,
        "scores": scores,
    }
    write_json(cache_path, cached)
    return cached


def make_figure(summary_rows: list[dict[str, object]], path: Path) -> None:
    categories = ["sport", "danse", "calme", "overall"]
    model_keys = list(LOCKED_CONFIGS)
    values = {
        key: [
            next(
                float(row["false_positive_rate"])
                for row in summary_rows
                if row["model"] == key and row["category"] == category
            )
            for category in categories
        ]
        for key in model_keys
    }
    labels = {
        "lstm_original": "LSTM original",
        "lstm_augmented": "LSTM enrichi",
        "gru_original": "GRU original",
        "gru_augmented": "GRU enrichi",
    }
    colors = ["#3969ac", "#11a579", "#f2b701", "#e73f74"]
    x = np.arange(len(categories))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for index, key in enumerate(model_keys):
        bars = ax.bar(
            x + (index - 1.5) * width,
            np.asarray(values[key]) * 100,
            width,
            label=labels[key],
            color=colors[index],
        )
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_ylabel("Taux de faux positifs (%)")
    ax.set_xticks(x, ["Sport (n=15)", "Danse (n=10)", "Calme (n=5)", "Global (n=30)"])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows, manifest_hash = load_manifest()
    verify_model_hashes()
    backbone, heads = load_shared_models()

    scored: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        scored.append(score_video(row, backbone, heads))
        print(f"[{index:02d}/30] {row['relative_path']}", flush=True)

    prediction_rows: list[dict[str, object]] = []
    for record in scored:
        stem = Path(str(record["relative_path"])).stem
        subcategory = stem.split("_v2_")[0].split("_")[0]
        for model_key, config in LOCKED_CONFIGS.items():
            k_value = int(config["k"])
            threshold = float(config["threshold"])
            max_rolling_mean = float(
                np.max(rolling_means(record["scores"][model_key], k_value))
            )
            prediction_rows.append(
                {
                    "model": model_key,
                    "architecture": config["architecture"],
                    "enrichment": config["enrichment"],
                    "threshold": threshold,
                    "k": k_value,
                    "category": record["category"],
                    "subcategory": subcategory,
                    "relative_path": record["relative_path"],
                    "video_sha256": record["video_sha256"],
                    "original_frame_count": record["original_frame_count"],
                    "padded_frames": record["padded_frames"],
                    "score_windows": len(record["scores"][model_key]),
                    "max_rolling_mean": max_rolling_mean,
                    "prediction": int(max_rolling_mean >= threshold),
                    "false_positive": int(max_rolling_mean >= threshold),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for model_key, config in LOCKED_CONFIGS.items():
        model_rows = [row for row in prediction_rows if row["model"] == model_key]
        for category in ("sport", "danse", "calme", "overall"):
            category_rows = (
                model_rows
                if category == "overall"
                else [row for row in model_rows if row["category"] == category]
            )
            false_positives = sum(int(row["false_positive"]) for row in category_rows)
            summary_rows.append(
                {
                    "model": model_key,
                    "architecture": config["architecture"],
                    "enrichment": config["enrichment"],
                    "threshold": config["threshold"],
                    "k": config["k"],
                    "category": category,
                    "n_videos": len(category_rows),
                    "false_positives": false_positives,
                    "false_positive_rate": false_positives / len(category_rows),
                }
            )

    prediction_fields = list(prediction_rows[0])
    summary_fields = list(summary_rows[0])
    write_csv(OUTPUT_DIR / "hard_negative_predictions.csv", prediction_rows, prediction_fields)
    write_csv(OUTPUT_DIR / "false_positive_by_category.csv", summary_rows, summary_fields)
    make_figure(summary_rows, OUTPUT_DIR / "hard_negative_false_positive_rates.png")

    result = {
        "status": "completed_locked_hard_negative_evaluation",
        "created_utc": utc_now(),
        "hard_negative_manifest_sha256": manifest_hash,
        "n_videos": len(rows),
        "categories": {"sport": 15, "danse": 10, "calme": 5},
        "n_frames": N_FRAMES,
        "stride_frames": STRIDE_FRAMES,
        "shared_efficientnet_backbone_verified_bit_identical": True,
        "locked_configs": LOCKED_CONFIGS,
        "results": summary_rows,
    }
    write_json(OUTPUT_DIR / "result.json", result)

    readable = []
    for row in summary_rows:
        readable.append(
            f"| {row['model']} | {row['category']} | {row['false_positives']}/{row['n_videos']} | "
            f"{100 * float(row['false_positive_rate']):.2f}% |"
        )
    summary_md = "\n".join(
        [
            "# Faux positifs hard-negative - configurations verrouillees",
            "",
            "Evaluation categorielle des 30 videos non violentes du corpus v2 scelle.",
            "Aucun theta ni K n'a ete reselectionne sur ce corpus.",
            "",
            "| Modele | Categorie | Faux positifs | Taux |",
            "|---|---:|---:|---:|",
            *readable,
            "",
            "Les predictions individuelles et les scores de decision sont dans",
            "`hard_negative_predictions.csv`.",
        ]
    )
    (OUTPUT_DIR / "summary.md").write_text(summary_md + "\n", encoding="utf-8")

    figure_names = ["hard_negative_false_positive_rates.png"]
    figure_manifest = {
        "status": "complete",
        "created_utc": utc_now(),
        "figures": {
            name: {
                "sha256": sha256_file(OUTPUT_DIR / name),
                "size_bytes": (OUTPUT_DIR / name).stat().st_size,
            }
            for name in figure_names
        },
    }
    write_json(OUTPUT_DIR / "figure_manifest.json", figure_manifest)

    published = [
        "false_positive_by_category.csv",
        "hard_negative_predictions.csv",
        "hard_negative_false_positive_rates.png",
        "figure_manifest.json",
        "result.json",
        "summary.md",
    ]
    artifact_manifest = {
        "status": "complete",
        "created_utc": utc_now(),
        "inputs": {
            "hard_negative_manifest_sha256": manifest_hash,
            "models": {
                key: config["model_sha256"] for key, config in LOCKED_CONFIGS.items()
            },
        },
        "outputs": {
            name: {
                "sha256": sha256_file(OUTPUT_DIR / name),
                "size_bytes": (OUTPUT_DIR / name).stat().st_size,
            }
            for name in published
        },
        "cache": {
            path.name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(CACHE_DIR.glob("*.json"))
        },
    }
    write_json(OUTPUT_DIR / "artifact_manifest.json", artifact_manifest)
    print("Evaluation hard-negative terminee.", flush=True)


if __name__ == "__main__":
    main()

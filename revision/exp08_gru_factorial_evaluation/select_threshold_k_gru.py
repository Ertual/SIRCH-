from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

from gru_factorial_common import (
    AUGMENTED_WEIGHTS_PATH,
    AUGMENTED_WEIGHTS_SHA256,
    EXPECTED_VALIDATION_COUNT,
    EXP01_OUTPUTS,
    EXPERIMENT_DIR,
    K_VALUES,
    N_FRAMES,
    ORIGINAL_MODEL_PATH,
    ORIGINAL_MODEL_SHA256,
    SOURCE_BEST_EPOCH,
    STRIDE_FRAMES,
    load_or_score_pair,
    load_shared_backbone_and_heads,
    records_for_model,
    write_validation_artifacts,
)
from select_threshold_k import THRESHOLDS, load_validation_manifest


def main() -> None:
    outputs = EXPERIMENT_DIR / "outputs"
    model_output_dirs = {
        "gru_original": outputs / "gru_original",
        "gru_augmented": outputs / "gru_augmented",
    }
    existing = [
        path / "selected_config.json"
        for path in model_output_dirs.values()
        if (path / "selected_config.json").exists()
    ]
    if existing:
        raise RuntimeError(
            "Une selection GRU existe deja; elle ne peut pas etre recalculee implicitement: "
            + ", ".join(str(path) for path in existing)
        )

    rows, manifest_hash, seal = load_validation_manifest(
        EXP01_OUTPUTS / "validation_manifest.csv",
        EXP01_OUTPUTS / "manifest_seal.json",
    )
    if len(rows) != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            f"Validation attendue: {EXPECTED_VALIDATION_COUNT}, obtenu: {len(rows)}"
        )
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Un GPU est visible alors que le protocole exige CPU seul.")

    cache_dir = outputs / "shared_validation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone, heads = load_shared_backbone_and_heads()
    print(
        "VALIDATION UNIQUEMENT: 594 videos | GRU original + GRU enrichi epoch 5 | "
        f"N={N_FRAMES} | stride={STRIDE_FRAMES}",
        flush=True,
    )

    datasets_root = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
    paired_records = []
    for index, row in enumerate(rows, start=1):
        paired_records.append(
            load_or_score_pair(row, datasets_root, cache_dir, backbone, heads)
        )
        if index % 10 == 0 or index == len(rows):
            print(f"Validation GRU scoree: {index}/{len(rows)}", flush=True)

    common = {
        "status": "selected_on_validation_only",
        "validation_manifest_sha256": manifest_hash,
        "manifest_status": seal["status"],
        "cross_split_sha256_duplicate_groups": seal[
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
        "shared_backbone_verified_bit_identical": True,
    }
    configs = {
        "gru_original": {
            **common,
            "experiment": "gru_original_clean_evaluation",
            "model_path": str(ORIGINAL_MODEL_PATH),
            "model_sha256": ORIGINAL_MODEL_SHA256,
        },
        "gru_augmented": {
            **common,
            "experiment": "gru_augmented_clean",
            "weights_path": str(AUGMENTED_WEIGHTS_PATH),
            "weights_sha256": AUGMENTED_WEIGHTS_SHA256,
            "source_best_epoch": SOURCE_BEST_EPOCH,
            "epoch_11_checkpoint_used": False,
        },
    }
    for model_key, output_dir in model_output_dirs.items():
        write_validation_artifacts(
            output_dir,
            records_for_model(paired_records, model_key),
            configs[model_key],
        )

    selected = {
        key: json.loads(
            (path / "selected_config.json").read_text(encoding="utf-8")
        )["selected"]
        for key, path in model_output_dirs.items()
    }
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()

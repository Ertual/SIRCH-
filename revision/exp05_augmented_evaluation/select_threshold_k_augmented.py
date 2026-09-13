from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP02_DIR = PROJECT_ROOT / "revision" / "exp02_validation_threshold_k"
EXP04_DIR = PROJECT_ROOT / "revision" / "exp04_augmented_clean"
sys.path.insert(0, str(EXP02_DIR))
sys.path.insert(0, str(EXP04_DIR))

from select_threshold_k import (  # noqa: E402
    K_VALUES,
    N_FRAMES,
    STRIDE_FRAMES,
    THRESHOLDS,
    evaluate_grid,
    load_or_score,
    load_validation_manifest,
    selection_key,
    sha256_file,
    write_csv,
)
from train_lstm_augmented_clean import build_model  # noqa: E402


WEIGHTS_PATH = Path(
    r"C:\SIRCH_ENV\models\revision\exp04_augmented_clean"
    r"\best_lstm_augmented_clean.weights.h5"
)
EXPECTED_WEIGHTS_SHA256 = (
    "4ccfe5bb3838cd829fa5e4eaba0332128d14e52380509eee2d4083c321aeaa13"
)
SOURCE_BEST_EPOCH = 6
EXPECTED_VALIDATION_COUNT = 594


def load_augmented_model_parts() -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(WEIGHTS_PATH)
    actual_hash = sha256_file(WEIGHTS_PATH)
    if actual_hash != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            "Les poids augmentes ne correspondent pas a l'epoch 6 restauree: "
            f"attendu={EXPECTED_WEIGHTS_SHA256}, obtenu={actual_hash}"
        )

    model = build_model()
    model.load_weights(WEIGHTS_PATH)
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
    head = tf.keras.Model(feature_input, x, name="sirch_augmented_temporal_head")
    return model, backbone, head


def main() -> None:
    exp01_outputs = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
    validation_manifest = exp01_outputs / "validation_manifest.csv"
    manifest_seal = exp01_outputs / "manifest_seal.json"
    output_dir = Path(__file__).resolve().parent / "outputs"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "selected_config.json"
    if selected_path.exists():
        raise RuntimeError(
            "La selection validation du modele enrichi existe deja. "
            "Supprimer explicitement les sorties serait necessaire pour la recalculer."
        )

    rows, manifest_hash, seal = load_validation_manifest(
        validation_manifest, manifest_seal
    )
    if len(rows) != EXPECTED_VALIDATION_COUNT:
        raise RuntimeError(
            f"Validation attendue: {EXPECTED_VALIDATION_COUNT}, obtenu: {len(rows)}"
        )
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Un GPU est visible alors que le protocole exige CPU seul.")

    _, backbone, head = load_augmented_model_parts()
    print(
        f"VALIDATION UNIQUEMENT: {len(rows)} videos | poids epoch {SOURCE_BEST_EPOCH} | "
        f"N={N_FRAMES} | stride={STRIDE_FRAMES}",
        flush=True,
    )

    datasets_root = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
    records: list[dict[str, object]] = []
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
            print(f"Validation scoree: {index}/{len(rows)}", flush=True)

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
        output_dir / "validation_score_streams.csv",
        list(stream_rows[0]),
        stream_rows,
    )

    grid = evaluate_grid(records)
    selected = max(grid, key=selection_key)
    for row in grid:
        row["selected"] = int(row is selected)
    write_csv(output_dir / "selection_grid.csv", list(grid[0]), grid)

    selected_config = {
        "status": "selected_on_validation_only",
        "experiment": "lstm_augmented_clean",
        "weights_path": str(WEIGHTS_PATH),
        "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "source_best_epoch": SOURCE_BEST_EPOCH,
        "epoch_12_checkpoint_used": False,
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
        "selected": selected,
        "videos_with_padding": sum(
            int(record["padded_frames"]) > 0 for record in records
        ),
    }
    selected_path.write_text(
        json.dumps(selected_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    summary = [
        "# Selection theta/K du modele LSTM enrichi",
        "",
        "Le jeu de test principal n'a pas ete lu.",
        f"Les poids utilises sont ceux du meilleur epoch restaure ({SOURCE_BEST_EPOCH}).",
        f"SHA-256 des poids : `{EXPECTED_WEIGHTS_SHA256}`.",
        "",
        f"- Videos de validation : {len(records)}",
        f"- K selectionne : {selected['k']}",
        f"- Seuil selectionne : {float(selected['threshold']):.2f}",
        f"- Accuracy : {float(selected['accuracy']):.4f}",
        f"- Balanced accuracy : {float(selected['balanced_accuracy']):.4f}",
        f"- Precision : {float(selected['precision']):.4f}",
        f"- Rappel : {float(selected['recall']):.4f}",
        f"- F1 : {float(selected['f1']):.4f}",
        f"- Matrice : TN={selected['tn']}, FP={selected['fp']}, FN={selected['fn']}, TP={selected['tp']}",
    ]
    (output_dir / "summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected_config, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

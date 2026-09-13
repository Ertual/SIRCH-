from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXP01_OUTPUTS = PROJECT_ROOT / "revision" / "exp01_manifests_sha256" / "outputs"
EXP02_DIR = PROJECT_ROOT / "revision" / "exp02_validation_threshold_k"
EXP07_DIR = PROJECT_ROOT / "revision" / "exp07_gru_augmented_clean"

sys.path.insert(0, str(EXP02_DIR))
sys.path.insert(0, str(EXP07_DIR))

from select_threshold_k import (  # noqa: E402
    K_VALUES,
    N_FRAMES,
    STRIDE_FRAMES,
    evaluate_grid,
    resolve_video_path,
    selection_key,
    sha256_file,
    video_features,
    write_csv,
)
from train_gru_augmented_clean import build_model  # noqa: E402


ORIGINAL_MODEL_PATH = Path(r"C:\SIRCH_ENV\models\phase3\sirch_model_gru.h5")
ORIGINAL_MODEL_SHA256 = (
    "fc060649cc4935c59c23596ab926d19af0e189c76983ac666c9038a8d0d98eaf"
)
AUGMENTED_WEIGHTS_PATH = Path(
    r"C:\SIRCH_ENV\models\revision\exp07_gru_augmented_clean"
    r"\best_gru_augmented_clean.weights.h5"
)
AUGMENTED_WEIGHTS_SHA256 = (
    "f7d3a87697d6b6b1ca36d46cdf7c1af6a499557b3e60dccc06078ff26a15956e"
)
SOURCE_BEST_EPOCH = 5
EXPECTED_VALIDATION_COUNT = 594
EXPECTED_TEST_COUNT = 592
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "933a26378a88236beafeb51fc91b53c7abab5185aba9d97daa0b68dd2310e397"
)
EXPECTED_TEST_MANIFEST_SHA256 = (
    "13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f"
)
MODEL_KEYS = ("gru_original", "gru_augmented")


def _split_model(
    model: tf.keras.Model, head_name: str
) -> tuple[tf.keras.Model, tf.keras.Model]:
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
    return backbone, tf.keras.Model(feature_input, x, name=head_name)


def load_shared_backbone_and_heads() -> tuple[tf.keras.Model, dict[str, tf.keras.Model]]:
    expected = {
        ORIGINAL_MODEL_PATH: ORIGINAL_MODEL_SHA256,
        AUGMENTED_WEIGHTS_PATH: AUGMENTED_WEIGHTS_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Artefact modele modifie: {path}; attendu={expected_hash}, obtenu={actual_hash}"
            )

    original_model = build_model()
    original_model.load_weights(ORIGINAL_MODEL_PATH)
    original_backbone, original_head = _split_model(
        original_model, "sirch_gru_original_temporal_head"
    )

    augmented_model = build_model()
    augmented_model.load_weights(AUGMENTED_WEIGHTS_PATH)
    augmented_backbone, augmented_head = _split_model(
        augmented_model, "sirch_gru_augmented_temporal_head"
    )

    original_weights = original_backbone.get_weights()
    augmented_weights = augmented_backbone.get_weights()
    if len(original_weights) != len(augmented_weights) or any(
        not np.array_equal(original, augmented)
        for original, augmented in zip(original_weights, augmented_weights)
    ):
        raise RuntimeError(
            "Les backbones EfficientNet des deux GRU ne sont pas identiques; "
            "le partage des caracteristiques est interdit."
        )
    return original_backbone, {
        "gru_original": original_head,
        "gru_augmented": augmented_head,
    }


def score_video_pair(
    path: Path,
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
) -> tuple[dict[str, list[float]], int, int]:
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
    scores = {
        key: [float(value) for value in head.predict(windows, batch_size=64, verbose=0).reshape(-1)]
        for key, head in heads.items()
    }
    return scores, original_frame_count, padded_frames


def load_or_score_pair(
    row: dict[str, str],
    datasets_root: Path,
    cache_dir: Path,
    backbone: tf.keras.Model,
    heads: dict[str, tf.keras.Model],
) -> dict[str, object]:
    cache_path = cache_dir / f"{row['sha256']}.json"
    model_hashes = {
        "gru_original": ORIGINAL_MODEL_SHA256,
        "gru_augmented": AUGMENTED_WEIGHTS_SHA256,
    }
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("model_sha256") == model_hashes
            and cached.get("n_frames") == N_FRAMES
            and cached.get("stride_frames") == STRIDE_FRAMES
            and cached.get("video_sha256") == row["sha256"]
            and int(cached.get("label", -1)) == int(row["label"])
            and all(key in cached.get("scores", {}) for key in MODEL_KEYS)
        ):
            return cached

    path = resolve_video_path(row, datasets_root)
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Hash video different du manifeste: {path}")
    scores, original_frame_count, padded_frames = score_video_pair(
        path, backbone, heads
    )
    cached = {
        "dataset": row["dataset"],
        "relative_path": row["relative_path"],
        "video_sha256": row["sha256"],
        "model_sha256": model_hashes,
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


def records_for_model(
    paired_records: list[dict[str, object]], model_key: str
) -> list[dict[str, object]]:
    records = []
    for paired in paired_records:
        records.append(
            {
                "dataset": paired["dataset"],
                "relative_path": paired["relative_path"],
                "video_sha256": paired["video_sha256"],
                "label": paired["label"],
                "original_frame_count": paired["original_frame_count"],
                "padded_frames": paired["padded_frames"],
                "scores": paired["scores"][model_key],
            }
        )
    return records


def write_validation_artifacts(
    output_dir: Path,
    records: list[dict[str, object]],
    selected_config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    write_csv(output_dir / "validation_score_streams.csv", list(stream_rows[0]), stream_rows)

    grid = evaluate_grid(records)
    selected = max(grid, key=selection_key)
    for row in grid:
        row["selected"] = int(row is selected)
    write_csv(output_dir / "selection_grid.csv", list(grid[0]), grid)
    selected_config["selected"] = selected
    selected_config["videos_with_padding"] = sum(
        int(record["padded_frames"]) > 0 for record in records
    )
    (output_dir / "selected_config.json").write_text(
        json.dumps(selected_config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    summary = [
        f"# Selection theta/K - {selected_config['experiment']}",
        "",
        "Selection effectuee sur les 594 videos de validation uniquement.",
        "Le manifeste de test n'a pas ete lu.",
        "",
        f"- K selectionne : {selected['k']}",
        f"- Seuil selectionne : {float(selected['threshold']):.2f}",
        f"- Accuracy : {float(selected['accuracy']):.4f}",
        f"- Balanced accuracy : {float(selected['balanced_accuracy']):.4f}",
        f"- Precision : {float(selected['precision']):.4f}",
        f"- Rappel : {float(selected['recall']):.4f}",
        f"- F1 : {float(selected['f1']):.4f}",
        f"- Matrice : TN={selected['tn']}, FP={selected['fp']}, FN={selected['fn']}, TP={selected['tp']}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

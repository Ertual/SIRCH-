from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from numpy.lib.format import open_memmap
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


N_FRAMES = 20
IMG_SIZE = 224
FEATURE_DIM = 1280
RECURRENT_UNITS = 256
DROPOUT = 0.5
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42
FEATURE_VIDEO_BATCH = 4


@dataclass(frozen=True)
class Experiment:
    key: str
    architecture: str
    train_manifest: str


EXPERIMENTS = (
    Experiment("lstm_original_grouped", "LSTM", "train_manifest_grouped.csv"),
    Experiment("lstm_augmented_grouped", "LSTM", "augmented_train_manifest_grouped.csv"),
    Experiment("gru_original_grouped", "GRU", "train_manifest_grouped.csv"),
    Experiment("gru_augmented_grouped", "GRU", "augmented_train_manifest_grouped.csv"),
)

EXPECTED_MANIFESTS = {
    "train_manifest_grouped.csv": {
        "rows": 2785,
        "sha256": "55c83d2b2546c09700bfc208f5b08cc915755359dea98d58aa746c00ea9453f8",
    },
    "validation_manifest_grouped.csv": {
        "rows": 597,
        "sha256": "d4b3f4a41c4d118a99abd70e88943f7886970695353369259ec3ea506e3964e5",
    },
    "augmented_train_manifest_grouped.csv": {
        "rows": 2810,
        "sha256": "287a299b9c47808a9dda2ce51877e1f8f1b14dc00c919a3f8fdf3c446abd3b2f",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_weights(model: Model) -> str:
    digest = hashlib.sha256()
    for weight in model.get_weights():
        array = np.ascontiguousarray(weight)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    try:
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(0.05 * (2**attempt), 0.5))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def row_key(row: dict[str, str]) -> str:
    return "|".join((row["dataset"], row["relative_path"], row["sha256"]))


def resolve_path(row: dict[str, str], project_root: Path, datasets_root: Path) -> Path:
    if row["dataset"] == "RLVS":
        return datasets_root / "RLVS" / "Real Life Violence Dataset" / row["relative_path"]
    if row["dataset"] == "RWF-2000":
        return datasets_root / "RWF-2000" / row["relative_path"]
    if row["dataset"] == "historical_enrichment_v1":
        return project_root / "datasets" / row["relative_path"]
    raise RuntimeError(f"Unexpected dataset: {row['dataset']}")


def configure_runtime() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("A GPU is visible although this protocol is CPU-only.")


def load_and_verify_manifests(
    project_root: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    exp11 = project_root / "revision" / "exp11_grouped_scene_splits" / "outputs"
    integrity_path = exp11 / "integrity_checks.json"
    seal_path = exp11 / "manifest_sha256.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise RuntimeError("The grouped split integrity report is not PASS.")
    if integrity.get("scene_group_split_violations"):
        raise RuntimeError("The grouped split report contains scene-group violations.")
    if integrity.get("exact_sha256_split_violations"):
        raise RuntimeError("The grouped split report contains exact-hash violations.")

    sealed = {entry["file"]: entry for entry in seal["files"]}
    manifests: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    for name, expected in EXPECTED_MANIFESTS.items():
        path = exp11 / name
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(f"Grouped manifest hash changed: {name}: {actual_hash}")
        if sealed.get(name, {}).get("sha256") != actual_hash:
            raise RuntimeError(f"Grouped manifest seal mismatch: {name}")
        rows = read_csv(path)
        if len(rows) != expected["rows"] or int(sealed[name]["rows"]) != len(rows):
            raise RuntimeError(f"Grouped manifest row count changed: {name}: {len(rows)}")
        manifests[name] = rows
        hashes[name] = actual_hash

    train = manifests["train_manifest_grouped.csv"]
    validation = manifests["validation_manifest_grouped.csv"]
    augmented = manifests["augmented_train_manifest_grouped.csv"]
    if any(row["split"] != "train" for row in train + augmented):
        raise RuntimeError("A training manifest contains a row outside train.")
    if any(row["split"] != "validation" for row in validation):
        raise RuntimeError("The validation manifest contains a row outside validation.")
    if [sum(int(row["label"]) == value for row in train) for value in (0, 1)] != [1399, 1386]:
        raise RuntimeError("Unexpected class counts in grouped principal train.")
    if [sum(int(row["label"]) == value for row in augmented) for value in (0, 1)] != [1424, 1386]:
        raise RuntimeError("Unexpected class counts in grouped augmented train.")
    if [sum(int(row["label"]) == value for row in validation) for value in (0, 1)] != [300, 297]:
        raise RuntimeError("Unexpected class counts in grouped validation.")

    augmented_keys = {row_key(row) for row in augmented}
    missing_principal = [row_key(row) for row in train if row_key(row) not in augmented_keys]
    if missing_principal:
        raise RuntimeError("The augmented train does not contain the complete principal train.")
    if {row["sha256"] for row in augmented} & {row["sha256"] for row in validation}:
        raise RuntimeError("Exact hash overlap between augmented train and validation.")

    exp12_result = (
        project_root
        / "revision"
        / "exp12_hard_negative_leakage_audit"
        / "outputs"
        / "result.json"
    )
    hard_negative_audit = json.loads(exp12_result.read_text(encoding="utf-8"))
    if hard_negative_audit.get("status") != "PASS":
        raise RuntimeError("The hard-negative versus train leakage audit is not PASS.")
    if int(hard_negative_audit.get("probable_near_duplicate_count", -1)) != 0:
        raise RuntimeError("A probable hard-negative versus train near-duplicate remains.")
    if int(hard_negative_audit.get("possible_near_duplicate_count", -1)) != 0:
        raise RuntimeError("A possible hard-negative versus train near-duplicate remains.")
    return manifests, hashes


def load_video_frames(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")
    indices = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
    frames: list[np.ndarray] = []
    last_frame: np.ndarray | None = None
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            if last_frame is None:
                capture.release()
                raise RuntimeError(f"Unreadable first sampled frame: {video_path}")
            frame = last_frame.copy()
        last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32))
    capture.release()
    return preprocess_input(np.stack(frames))


def make_backbone() -> Model:
    backbone = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False
    if int(backbone.output_shape[-1]) != FEATURE_DIM:
        raise RuntimeError(f"Unexpected EfficientNet feature dimension: {backbone.output_shape}")
    return backbone


def precompute_features(
    project_root: Path,
    datasets_root: Path,
    output_root: Path,
    manifests: dict[str, list[dict[str, str]]],
    manifest_hashes: dict[str, str],
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    augmented = manifests["augmented_train_manifest_grouped.csv"]
    validation = manifests["validation_manifest_grouped.csv"]
    combined = [("augmented_train", row) for row in augmented] + [
        ("validation", row) for row in validation
    ]
    keys = [row_key(row) for _, row in combined]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate feature-cache key in train plus validation.")

    feature_path = output_root / "uniform_frame_features.npy"
    index_path = output_root / "uniform_frame_feature_index.csv"
    state_path = output_root / "feature_cache_status.json"
    meta_path = output_root / "feature_cache_manifest.json"
    expected_shape = (len(combined), N_FRAMES, FEATURE_DIM)

    if meta_path.exists() and feature_path.exists() and index_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            metadata.get("status") == "complete"
            and tuple(metadata.get("shape", ())) == expected_shape
            and metadata.get("manifest_sha256") == manifest_hashes
            and metadata.get("feature_file_sha256") == sha256_file(feature_path)
        ):
            index_rows = read_csv(index_path)
            if len(index_rows) != len(combined):
                raise RuntimeError("Feature cache index count does not match metadata.")
            print(f"Feature cache already complete: {feature_path}", flush=True)
            return feature_path, index_rows, metadata

    backbone = make_backbone()
    backbone_hash = sha256_weights(backbone)
    if state_path.exists() and feature_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            tuple(state.get("shape", ())) != expected_shape
            or state.get("manifest_sha256") != manifest_hashes
            or state.get("backbone_weights_sha256") != backbone_hash
        ):
            raise RuntimeError("An incompatible partial feature cache already exists.")
        completed = int(state.get("completed", 0))
        features = np.load(feature_path, mmap_mode="r+")
    else:
        completed = 0
        features = open_memmap(feature_path, mode="w+", dtype=np.float32, shape=expected_shape)
        state = {
            "status": "running",
            "created_utc": utc_now(),
            "pid": os.getpid(),
            "shape": list(expected_shape),
            "dtype": "float32",
            "completed": 0,
            "manifest_sha256": manifest_hashes,
            "backbone_weights_sha256": backbone_hash,
            "sampling": "20 uniformly spaced frames using the original training loader",
        }
        write_json(state_path, state)

    print(
        f"Frozen EfficientNet features: resume={completed}/{len(combined)} | "
        f"batch={FEATURE_VIDEO_BATCH}",
        flush=True,
    )
    try:
        for start in range(completed, len(combined), FEATURE_VIDEO_BATCH):
            stop = min(start + FEATURE_VIDEO_BATCH, len(combined))
            frame_batches: list[np.ndarray] = []
            paths: list[Path] = []
            for _, row in combined[start:stop]:
                path = resolve_path(row, project_root, datasets_root)
                if not path.exists():
                    raise FileNotFoundError(path)
                actual_hash = sha256_file(path)
                if actual_hash != row["sha256"]:
                    raise RuntimeError(f"Video hash changed: {path}: {actual_hash}")
                paths.append(path)
                frame_batches.append(load_video_frames(path))
            flat_frames = np.concatenate(frame_batches, axis=0)
            flat_features = backbone.predict(
                flat_frames, batch_size=N_FRAMES * FEATURE_VIDEO_BATCH, verbose=0
            )
            features[start:stop] = flat_features.reshape(stop - start, N_FRAMES, FEATURE_DIM)
            features.flush()
            state.update(
                {
                    "status": "running",
                    "completed": stop,
                    "current_video": str(paths[-1]),
                    "updated_utc": utc_now(),
                }
            )
            write_json(state_path, state)
            if stop % 20 == 0 or stop == len(combined):
                print(f"Frozen features: {stop}/{len(combined)}", flush=True)
    except BaseException as error:
        state.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": utc_now(),
            }
        )
        write_json(state_path, state)
        raise

    index_rows: list[dict[str, Any]] = []
    for index, (source, row) in enumerate(combined):
        index_rows.append(
            {
                "feature_index": index,
                "source_partition": source,
                "dataset": row["dataset"],
                "relative_path": row["relative_path"],
                "label": int(row["label"]),
                "video_sha256": row["sha256"],
                "row_key": row_key(row),
            }
        )
    write_csv(index_path, index_rows)
    feature_hash = sha256_file(feature_path)
    metadata = {
        "status": "complete",
        "completed_utc": utc_now(),
        "shape": list(expected_shape),
        "dtype": "float32",
        "feature_file": str(feature_path),
        "feature_file_sha256": feature_hash,
        "feature_index": str(index_path),
        "feature_index_sha256": sha256_file(index_path),
        "manifest_sha256": manifest_hashes,
        "backbone_weights_sha256": backbone_hash,
        "sampling": {
            "n_frames": N_FRAMES,
            "method": "numpy.linspace over decoded frame count",
            "image_size": [IMG_SIZE, IMG_SIZE],
            "preprocess": "tensorflow.keras.applications.efficientnet.preprocess_input",
        },
        "equivalence_basis": (
            "The EfficientNetB0 backbone is frozen in all four models; cached features are "
            "therefore the exact deterministic input of the recurrent head."
        ),
    }
    write_json(meta_path, metadata)
    state.update({"status": "complete", "completed": len(combined), "updated_utc": utc_now()})
    write_json(state_path, state)
    del features
    tf.keras.backend.clear_session()
    return feature_path, index_rows, metadata


class FeatureSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        feature_path: Path,
        feature_indices: list[int],
        labels: list[int],
        shuffle: bool,
    ) -> None:
        self.features = np.load(feature_path, mmap_mode="r")
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.shuffle = shuffle
        self.indices = np.arange(len(labels), dtype=np.int64)
        self.rng = np.random.RandomState(SEED)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / BATCH_SIZE))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        selected = self.indices[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        feature_rows = self.feature_indices[selected]
        return np.asarray(self.features[feature_rows]), self.labels[selected]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)


def build_full_and_head(architecture: str) -> tuple[Model, Model, Model]:
    backbone = make_backbone()
    tf.keras.utils.set_random_seed(SEED)
    sequence_input = layers.Input(
        shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3), name="video_frames"
    )
    td = layers.TimeDistributed(backbone, name="frozen_efficientnet")
    x = td(sequence_input)
    if architecture == "LSTM":
        recurrent = layers.LSTM(RECURRENT_UNITS, return_sequences=False, name="lstm")
    elif architecture == "GRU":
        recurrent = layers.GRU(RECURRENT_UNITS, return_sequences=False, name="gru")
    else:
        raise ValueError(f"Unexpected architecture: {architecture}")
    dropout_1 = layers.Dropout(DROPOUT, name="temporal_dropout")
    dense = layers.Dense(128, activation="relu", name="dense_128")
    dropout_2 = layers.Dropout(0.3, name="dense_dropout")
    classifier = layers.Dense(1, activation="sigmoid", name="violence_probability")
    x = recurrent(x)
    x = dropout_1(x)
    x = dense(x)
    x = dropout_2(x)
    output = classifier(x)
    full = Model(sequence_input, output, name=f"sirch_{architecture.lower()}_grouped")

    feature_input = layers.Input(shape=(N_FRAMES, FEATURE_DIM), name="cached_features")
    y = recurrent(feature_input)
    y = dropout_1(y)
    y = dense(y)
    y = dropout_2(y)
    head = Model(feature_input, classifier(y), name=f"sirch_{architecture.lower()}_head")
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return full, backbone, head


def json_metric(value: object) -> object:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()


class BestStateCallback(tf.keras.callbacks.Callback):
    def __init__(self, path: Path, initial_best: float | None) -> None:
        super().__init__()
        self.path = path
        self.best = initial_best

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        value = (logs or {}).get("val_loss")
        if value is None:
            return
        numeric = float(value)
        if self.best is None or numeric < self.best:
            self.best = numeric
            write_json(
                self.path,
                {"best_epoch": epoch + 1, "val_loss": numeric, "updated_utc": utc_now()},
            )


class StatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, path: Path, base: dict[str, Any], best_state_path: Path) -> None:
        super().__init__()
        self.path = path
        self.base = base
        self.best_state_path = best_state_path

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        write_json(
            self.path,
            {**self.base, "status": "training", "current_epoch": epoch + 1, "updated_utc": utc_now()},
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        best = (
            json.loads(self.best_state_path.read_text(encoding="utf-8"))
            if self.best_state_path.exists()
            else None
        )
        write_json(
            self.path,
            {
                **self.base,
                "status": "training",
                "last_completed_epoch": epoch + 1,
                "metrics": {key: json_metric(value) for key, value in (logs or {}).items()},
                "best": best,
                "updated_utc": utc_now(),
            },
        )


def prepare_hdf5_export(model: Model) -> None:
    for layer in model.submodules:
        if not isinstance(layer, layers.Rescaling):
            continue
        for attribute in ("scale", "offset"):
            value = getattr(layer, attribute)
            if tf.is_tensor(value):
                array = value.numpy()
                setattr(layer, attribute, float(array) if array.ndim == 0 else array.tolist())


def latest_checkpoint(checkpoints: Path, key: str) -> tuple[Path | None, int]:
    pattern = re.compile(rf"{re.escape(key)}_epoch_(\d+)\.weights\.h5$")
    candidates = []
    for path in checkpoints.glob(f"{key}_epoch_*.weights.h5"):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None, 0
    epoch, path = max(candidates)
    return path, epoch


def indices_for_rows(
    rows: list[dict[str, str]], cache_index: list[dict[str, Any]]
) -> tuple[list[int], list[int]]:
    lookup = {str(row["row_key"]): int(row["feature_index"]) for row in cache_index}
    indices: list[int] = []
    labels: list[int] = []
    for row in rows:
        key = row_key(row)
        if key not in lookup:
            raise RuntimeError(f"Manifest row missing from feature cache: {key}")
        indices.append(lookup[key])
        labels.append(int(row["label"]))
    return indices, labels


def train_experiment(
    experiment: Experiment,
    project_root: Path,
    datasets_root: Path,
    output_root: Path,
    feature_path: Path,
    cache_index: list[dict[str, Any]],
    feature_metadata: dict[str, Any],
    manifests: dict[str, list[dict[str, str]]],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    model_dir = output_root / experiment.key
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = model_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    status_path = model_dir / "training_status.json"
    result_path = model_dir / "training_result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete":
            print(f"Already complete: {experiment.key}", flush=True)
            return prior

    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    full, backbone, head = build_full_and_head(experiment.architecture)
    backbone_hash = sha256_weights(backbone)
    if backbone_hash != feature_metadata["backbone_weights_sha256"]:
        raise RuntimeError(f"Backbone differs from feature cache for {experiment.key}.")

    train_rows = manifests[experiment.train_manifest]
    validation_rows = manifests["validation_manifest_grouped.csv"]
    train_indices, train_labels = indices_for_rows(train_rows, cache_index)
    validation_indices, validation_labels = indices_for_rows(validation_rows, cache_index)
    train_sequence = FeatureSequence(feature_path, train_indices, train_labels, shuffle=True)
    validation_sequence = FeatureSequence(
        feature_path, validation_indices, validation_labels, shuffle=False
    )

    checkpoint, initial_epoch = latest_checkpoint(checkpoints, experiment.key)
    if checkpoint is not None:
        head.load_weights(checkpoint)
        print(
            f"Resume {experiment.key} from epoch {initial_epoch}; Adam state is reset.",
            flush=True,
        )

    log_path = model_dir / f"training_log_{experiment.key}.csv"
    best_head_path = model_dir / f"best_{experiment.key}_head.weights.h5"
    best_full_weights_path = model_dir / f"best_{experiment.key}.weights.h5"
    best_state_path = model_dir / "best_state.json"
    full_model_path = model_dir / f"sirch_model_{experiment.key}.h5"
    initial_best = None
    if best_state_path.exists():
        initial_best = float(json.loads(best_state_path.read_text(encoding="utf-8"))["val_loss"])

    base_status = {
        "experiment": experiment.key,
        "architecture": experiment.architecture,
        "enrichment": "augmented" if "augmented" in experiment.key else "original",
        "pid": os.getpid(),
        "cpu_only": True,
        "seed": SEED,
        "train_videos": len(train_rows),
        "validation_videos": len(validation_rows),
        "n_frames": N_FRAMES,
        "feature_dim": FEATURE_DIM,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "initial_epoch": initial_epoch,
        "train_manifest": experiment.train_manifest,
        "train_manifest_sha256": manifest_hashes[experiment.train_manifest],
        "validation_manifest_sha256": manifest_hashes["validation_manifest_grouped.csv"],
        "feature_cache_sha256": feature_metadata["feature_file_sha256"],
        "frozen_backbone_weights_sha256": backbone_hash,
        "model_path": str(full_model_path),
        "best_weights_path": str(best_full_weights_path),
        "training_log": str(log_path),
        "checkpoint_dir": str(checkpoints),
    }
    environment = {
        **base_status,
        "started_utc": utc_now(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": 6,
        "early_stopping_restore_best_weights": True,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,
        "minimum_learning_rate": 1e-6,
        "training_optimization": "frozen EfficientNet features precomputed once",
        "test_manifest_read": False,
    }
    write_json(model_dir / "training_environment.json", environment)
    write_json(status_path, {**base_status, "status": "starting", "updated_utc": utc_now()})

    checkpoint_pattern = checkpoints / f"{experiment.key}_epoch_{{epoch:03d}}.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_pattern), save_best_only=False, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_head_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=initial_best,
            verbose=1,
        ),
        BestStateCallback(best_state_path, initial_best),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        StatusCallback(status_path, base_status, best_state_path),
    ]

    print(
        f"TRAIN {experiment.key}: train={len(train_rows)} validation={len(validation_rows)} "
        f"initial_epoch={initial_epoch}",
        flush=True,
    )
    try:
        if initial_epoch < EPOCHS:
            head.fit(
                train_sequence,
                validation_data=validation_sequence,
                initial_epoch=initial_epoch,
                epochs=EPOCHS,
                callbacks=callbacks,
                workers=1,
                use_multiprocessing=False,
                verbose=1,
            )
        if not best_head_path.exists():
            raise RuntimeError(f"Missing best head weights: {best_head_path}")
        head.load_weights(best_head_path)

        verification_row = validation_rows[0]
        verification_path = resolve_path(verification_row, project_root, datasets_root)
        frames = load_video_frames(verification_path)
        cached_index = validation_indices[0]
        cached_features = np.asarray(np.load(feature_path, mmap_mode="r")[cached_index])
        head_score = float(head.predict(cached_features[None, ...], verbose=0)[0, 0])
        full_score = float(full.predict(frames[None, ...], verbose=0)[0, 0])
        absolute_difference = abs(head_score - full_score)
        if not np.isclose(head_score, full_score, rtol=1e-5, atol=1e-6):
            raise RuntimeError(
                f"Head/full equivalence failed for {experiment.key}: "
                f"head={head_score}, full={full_score}"
            )
        equivalence = {
            "status": "PASS",
            "video": verification_row["relative_path"],
            "video_sha256": verification_row["sha256"],
            "head_score": head_score,
            "full_model_score": full_score,
            "absolute_difference": absolute_difference,
            "rtol": 1e-5,
            "atol": 1e-6,
        }
        write_json(model_dir / "feature_training_equivalence.json", equivalence)

        full.save_weights(best_full_weights_path)
        prepare_hdf5_export(full)
        full.save(full_model_path, include_optimizer=False)
        best = json.loads(best_state_path.read_text(encoding="utf-8"))
        result = {
            **base_status,
            "status": "complete",
            "completed_utc": utc_now(),
            "best_epoch": int(best["best_epoch"]),
            "best_val_loss": float(best["val_loss"]),
            "best_head_weights_sha256": sha256_file(best_head_path),
            "best_full_weights_sha256": sha256_file(best_full_weights_path),
            "full_model_sha256": sha256_file(full_model_path),
            "feature_training_equivalence": equivalence,
            "test_manifest_read": False,
        }
        write_json(result_path, result)
        write_json(status_path, result)
        print(
            f"COMPLETE {experiment.key}: best epoch={result['best_epoch']} "
            f"val_loss={result['best_val_loss']:.6f}",
            flush=True,
        )
        return result
    except BaseException as error:
        failure = {
            **base_status,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_utc": utc_now(),
        }
        write_json(status_path, failure)
        raise


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train the four grouped-split SIRCH factorial models without reading test."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "SIRCH_EXP13_MODEL_DIR",
                r"C:\SIRCH_ENV\models\revision\exp13_grouped_retraining",
            )
        ),
    )
    parser.add_argument(
        "--only",
        choices=[experiment.key for experiment in EXPERIMENTS],
        help="Train only one model after preparing the shared feature cache.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and verify shared frozen-backbone features, then stop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pipeline_status_path = args.output_root / "pipeline_status.json"
    configure_runtime()
    pipeline = {
        "status": "running",
        "stage": "verify_manifests",
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "project_root": str(args.project_root),
        "datasets_root": str(args.datasets_root),
        "output_root": str(args.output_root),
        "protocol": "grouped scene splits; validation only; test never read",
    }
    write_json(pipeline_status_path, pipeline)
    try:
        manifests, manifest_hashes = load_and_verify_manifests(args.project_root)
        pipeline.update({"stage": "precompute_frozen_features", "manifest_sha256": manifest_hashes})
        write_json(pipeline_status_path, pipeline)
        feature_path, cache_index, feature_metadata = precompute_features(
            args.project_root,
            args.datasets_root,
            args.output_root,
            manifests,
            manifest_hashes,
        )
        if args.prepare_only:
            pipeline.update({"status": "prepared", "stage": "complete", "updated_utc": utc_now()})
            write_json(pipeline_status_path, pipeline)
            return

        selected_experiments = [
            experiment
            for experiment in EXPERIMENTS
            if args.only is None or experiment.key == args.only
        ]
        results = []
        for experiment in selected_experiments:
            pipeline.update(
                {"stage": "training", "current_experiment": experiment.key, "updated_utc": utc_now()}
            )
            write_json(pipeline_status_path, pipeline)
            results.append(
                train_experiment(
                    experiment,
                    args.project_root,
                    args.datasets_root,
                    args.output_root,
                    feature_path,
                    cache_index,
                    feature_metadata,
                    manifests,
                    manifest_hashes,
                )
            )
        pipeline.update(
            {
                "status": "complete",
                "stage": "complete",
                "completed_utc": utc_now(),
                "experiments": [result["experiment"] for result in results],
                "test_manifest_read": False,
            }
        )
        write_json(pipeline_status_path, pipeline)
    except BaseException as error:
        pipeline.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": utc_now(),
            }
        )
        write_json(pipeline_status_path, pipeline)
        raise


if __name__ == "__main__":
    main()

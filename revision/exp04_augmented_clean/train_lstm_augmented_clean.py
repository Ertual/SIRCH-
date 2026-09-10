from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0


N_FRAMES = 20
IMG_SIZE = 224
LSTM_UNITS = 256
DROPOUT = 0.5
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def resolve_path(row: dict[str, str], project_root: Path, datasets_root: Path) -> Path:
    if row["dataset"] == "RLVS":
        return datasets_root / "RLVS" / "Real Life Violence Dataset" / row["relative_path"]
    if row["dataset"] == "RWF-2000":
        return datasets_root / "RWF-2000" / row["relative_path"]
    if row["dataset"] == "historical_enrichment_v1":
        return project_root / "datasets" / row["relative_path"]
    raise RuntimeError(f"Dataset inattendu: {row['dataset']}")


def load_video_frames(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video illisible: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Nombre de frames invalide: {video_path}")
    indices = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
    frames = []
    last_frame: np.ndarray | None = None
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            if last_frame is None:
                capture.release()
                raise RuntimeError(f"Premiere frame illisible: {video_path}")
            frame = last_frame.copy()
        last_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32))
    capture.release()
    return tf.keras.applications.efficientnet.preprocess_input(np.stack(frames))


class VideoSequence(tf.keras.utils.Sequence):
    def __init__(self, paths: list[Path], labels: list[int], shuffle: bool) -> None:
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.float32)
        self.shuffle = shuffle
        self.indices = np.arange(len(paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.paths) / BATCH_SIZE))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        selected = self.indices[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        batch = np.zeros(
            (len(selected), N_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32
        )
        for position, sample_index in enumerate(selected):
            batch[position] = load_video_frames(self.paths[int(sample_index)])
        return batch, self.labels[selected]

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model() -> Model:
    backbone = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False
    sequence_input = layers.Input(shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    features = layers.TimeDistributed(backbone)(sequence_input)
    x = layers.LSTM(LSTM_UNITS, return_sequences=False)(features)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = Model(sequence_input, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_metric(value: object) -> object:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.tolist()


def prepare_hdf5_export(model: Model) -> None:
    # TensorFlow 2.12 leaves EfficientNet's RGB Rescaling scale as an
    # EagerTensor, which the legacy HDF5 JSON serializer cannot encode.
    for layer in model.submodules:
        if not isinstance(layer, layers.Rescaling):
            continue
        for attribute in ("scale", "offset"):
            value = getattr(layer, attribute)
            if tf.is_tensor(value):
                array = value.numpy()
                setattr(layer, attribute, float(array) if array.ndim == 0 else array.tolist())


class StatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, path: Path, base: dict[str, object]) -> None:
        super().__init__()
        self.path = path
        self.base = base

    def on_epoch_begin(self, epoch: int, logs=None) -> None:
        write_status(
            self.path,
            {**self.base, "status": "training", "current_epoch": epoch + 1, "updated_utc": datetime.now(timezone.utc).isoformat()},
        )

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        write_status(
            self.path,
            {
                **self.base,
                "status": "training",
                "last_completed_epoch": epoch + 1,
                "metrics": {key: json_metric(value) for key, value in (logs or {}).items()},
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = Path(__file__).resolve().parent
    manifests_dir = experiment_dir / "outputs"
    anti_leak_path = manifests_dir / "anti_leak_report.json"
    augmented_manifest_path = manifests_dir / "augmented_train_manifest.csv"
    validation_manifest_path = (
        project_root / "revision" / "exp01_manifests_sha256" / "outputs" / "validation_manifest.csv"
    )
    output_root = Path(
        os.environ.get(
            "SIRCH_EXP04_MODEL_DIR",
            r"C:\SIRCH_ENV\models\revision\exp04_augmented_clean",
        )
    )
    datasets_root = Path(os.environ.get("SIRCH_DATASETS_DIR", r"C:\SIRCH_ENV\datasets"))
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = output_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    model_path = output_root / "sirch_model_lstm_augmented_clean.h5"
    best_weights_path = output_root / "best_lstm_augmented_clean.weights.h5"
    best_state_path = output_root / "best_state.json"
    log_path = output_root / "training_log_lstm_augmented_clean.csv"
    status_path = output_root / "training_status.json"

    anti_leak = json.loads(anti_leak_path.read_text(encoding="utf-8"))
    if anti_leak.get("status") != "sealed_clean":
        raise RuntimeError("Le rapport anti-fuite n'est pas propre.")
    if any(int(value) != 0 for value in anti_leak["overlap_sha256_groups"].values()):
        raise RuntimeError("Le rapport anti-fuite signale un chevauchement.")
    if sha256_file(augmented_manifest_path) != anti_leak["augmented_train_manifest_sha256"]:
        raise RuntimeError("Le manifeste de train augmente a change.")
    if sha256_file(validation_manifest_path) != anti_leak["source_manifest_sha256"]["validation"]:
        raise RuntimeError("Le manifeste de validation a change.")

    train_rows = read_csv(augmented_manifest_path)
    validation_rows = read_csv(validation_manifest_path)
    train_paths = [resolve_path(row, project_root, datasets_root) for row in train_rows]
    validation_paths = [resolve_path(row, project_root, datasets_root) for row in validation_rows]
    missing = [str(path) for path in train_paths + validation_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos absentes, premiere: {missing[0]}")

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("Un GPU est visible alors que le protocole exige CPU seul.")

    model = build_model()
    checkpoint_pattern = checkpoints / "lstm_augmented_clean_epoch_{epoch:03d}.weights.h5"
    checkpoint_regex = re.compile(r"lstm_augmented_clean_epoch_(\d+)\.weights\.h5$")
    checkpoint_files = sorted(
        checkpoints.glob("lstm_augmented_clean_epoch_*.weights.h5"),
        key=lambda path: int(checkpoint_regex.search(path.name).group(1)),
    )
    initial_epoch = 0
    if checkpoint_files:
        latest = checkpoint_files[-1]
        initial_epoch = int(checkpoint_regex.search(latest.name).group(1))
        model.load_weights(latest)
        print(f"Reprise depuis {latest}, epoch suivant={initial_epoch + 1}", flush=True)

    best_val_loss = None
    if log_path.exists():
        for row in read_csv(log_path):
            if row.get("val_loss"):
                value = float(row["val_loss"])
                best_val_loss = value if best_val_loss is None else min(best_val_loss, value)
    if best_state_path.exists():
        recovered_best = json.loads(best_state_path.read_text(encoding="utf-8"))
        value = float(recovered_best["val_loss"])
        best_val_loss = value if best_val_loss is None else min(best_val_loss, value)

    base_status = {
        "experiment": "lstm_augmented_clean",
        "pid": os.getpid(),
        "cpu_only": True,
        "seed": SEED,
        "train_videos": len(train_rows),
        "validation_videos": len(validation_rows),
        "n_frames": N_FRAMES,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "initial_epoch": initial_epoch,
        "model_path": str(model_path),
        "best_weights_path": str(best_weights_path),
        "training_log": str(log_path),
        "checkpoint_dir": str(checkpoints),
        "augmented_train_manifest_sha256": anti_leak["augmented_train_manifest_sha256"],
        "validation_manifest_sha256": anti_leak["source_manifest_sha256"]["validation"],
    }
    environment = {
        **base_status,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "early_stopping_patience": 6,
        "early_stopping_restore_best_weights": True,
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 3,
        "minimum_learning_rate": 1e-6,
    }
    (output_root / "training_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    write_status(status_path, {**base_status, "status": "starting", "updated_utc": datetime.now(timezone.utc).isoformat()})

    train_gen = VideoSequence(train_paths, [int(row["label"]) for row in train_rows], shuffle=True)
    validation_gen = VideoSequence(
        validation_paths, [int(row["label"]) for row in validation_rows], shuffle=False
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_pattern), save_best_only=False, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(log_path), append=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_weights_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=best_val_loss,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        StatusCallback(status_path, base_status),
    ]

    try:
        if initial_epoch < EPOCHS:
            model.fit(
                train_gen,
                validation_data=validation_gen,
                initial_epoch=initial_epoch,
                epochs=EPOCHS,
                callbacks=callbacks,
                workers=1,
                use_multiprocessing=False,
                verbose=1,
            )
        if best_weights_path.exists():
            model.load_weights(best_weights_path)
        prepare_hdf5_export(model)
        model.save(model_path, include_optimizer=False)
        write_status(
            status_path,
            {**base_status, "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()},
        )
        print(f"Entrainement termine. Meilleur modele: {model_path}", flush=True)
    except BaseException as error:
        write_status(
            status_path,
            {
                **base_status,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    main()

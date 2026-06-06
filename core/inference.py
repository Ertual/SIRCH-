from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

from config import IMG_SIZE, LSTM_UNITS, MODEL_PATH, N_FRAMES


def build_model() -> Model:
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    sequence_input = layers.Input(shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    features = layers.TimeDistributed(base_model)(sequence_input)
    x = layers.LSTM(LSTM_UNITS, return_sequences=False)(features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs=sequence_input, outputs=output)


class ViolenceInference:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Model | None = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modele introuvable : {self.model_path}. "
                "Entraine le modele puis place sirch_model.h5 dans models/."
            )
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except (OSError, TypeError, ValueError):
            self.model = build_model()
            self.model.load_weights(self.model_path)

    def predict(self, sequence: np.ndarray) -> float:
        if self.model is None:
            self.load()
        prediction = self.model.predict(sequence, verbose=0)
        return float(prediction.ravel()[0])

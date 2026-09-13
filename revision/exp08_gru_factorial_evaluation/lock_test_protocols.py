from __future__ import annotations

import json
from datetime import datetime, timezone

from gru_factorial_common import (
    AUGMENTED_WEIGHTS_SHA256,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPECTED_VALIDATION_MANIFEST_SHA256,
    EXP01_OUTPUTS,
    EXPERIMENT_DIR,
    ORIGINAL_MODEL_SHA256,
    SOURCE_BEST_EPOCH,
    sha256_file,
)


def main() -> None:
    protocol_path = EXPERIMENT_DIR / "locked_test_protocols.json"
    if protocol_path.exists():
        raise RuntimeError("Les protocoles GRU sont deja verrouilles.")

    selected_paths = {
        key: EXPERIMENT_DIR / "outputs" / key / "selected_config.json"
        for key in ("gru_original", "gru_augmented")
    }
    missing = [str(path) for path in selected_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("Selections validation absentes: " + ", ".join(missing))
    selected = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in selected_paths.items()
    }
    for key, config in selected.items():
        if config.get("status") != "selected_on_validation_only":
            raise RuntimeError(f"Statut de selection invalide pour {key}.")
        if config.get("test_manifest_read") is not False:
            raise RuntimeError(f"La selection {key} indique une lecture du test.")
        if config.get("validation_manifest_sha256") != EXPECTED_VALIDATION_MANIFEST_SHA256:
            raise RuntimeError(f"Le manifeste de validation a change pour {key}.")
    if selected["gru_original"].get("model_sha256") != ORIGINAL_MODEL_SHA256:
        raise RuntimeError("Le GRU original selectionne n'est pas l'artefact attendu.")
    augmented = selected["gru_augmented"]
    if (
        augmented.get("weights_sha256") != AUGMENTED_WEIGHTS_SHA256
        or int(augmented.get("source_best_epoch", -1)) != SOURCE_BEST_EPOCH
        or augmented.get("epoch_11_checkpoint_used") is not False
    ):
        raise RuntimeError("La selection enrichie ne repose pas exclusivement sur l'epoch 5.")

    seal = json.loads(
        (EXP01_OUTPUTS / "manifest_seal.json").read_text(encoding="utf-8")
    )
    if seal.get("status") != "sealed_clean":
        raise RuntimeError("Le sceau des manifestes n'est pas propre.")
    if seal["manifest_sha256"]["test"] != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le hash scelle du test a change.")

    locked = {}
    for key, config in selected.items():
        chosen = config["selected"]
        locked[key] = {
            "selected_config_path": str(selected_paths[key]),
            "selected_config_sha256": sha256_file(selected_paths[key]),
            "selected_on": "validation_only",
            "theta": float(chosen["threshold"]),
            "k": int(chosen["k"]),
            "n_frames": int(config["n_frames"]),
            "stride_frames": int(config["stride_frames"]),
        }

    protocol = {
        "schema_version": 1,
        "status": "preregistered_before_final_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "One final paired evaluation of the validation-locked original and "
            "augmented GRU configurations on the cleaned test split."
        ),
        "allowed_final_evaluations_per_model": 1,
        "shared_video_decode_and_backbone_pass": True,
        "models": {
            "gru_original": {
                "architecture": "EfficientNetB0 + GRU(256) + Dense(128)",
                "model_sha256": ORIGINAL_MODEL_SHA256,
            },
            "gru_augmented": {
                "architecture": "EfficientNetB0 + GRU(256) + Dense(128)",
                "weights_sha256": AUGMENTED_WEIGHTS_SHA256,
                "source_best_epoch": SOURCE_BEST_EPOCH,
                "epoch_11_checkpoint_used": False,
            },
        },
        "selected_configurations": locked,
        "test": {
            "manifest_path": "revision/exp01_manifests_sha256/outputs/test_manifest.csv",
            "manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
            "videos": 592,
            "violence": 297,
            "non_violence": 295,
            "cross_split_sha256_duplicate_groups": 0,
            "read_before_lock": False,
        },
        "decision_rule": (
            "A video is positive when at least one complete rolling mean of K "
            "consecutive sequence scores is greater than or equal to theta."
        ),
        "score_generation": (
            "Same implementation as validation: 20-frame windows, stride 5, "
            "one bit-identical EfficientNetB0 backbone pass and separate GRU heads."
        ),
        "short_clip_policy": "Repeat the last frame exactly as in validation.",
        "no_post_test_adjustment": True,
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPECTED_WEIGHTS_SHA256 = (
    "4ccfe5bb3838cd829fa5e4eaba0332128d14e52380509eee2d4083c321aeaa13"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "933a26378a88236beafeb51fc91b53c7abab5185aba9d97daa0b68dd2310e397"
)
EXPECTED_TEST_MANIFEST_SHA256 = (
    "13bfda2224edb723a75be0deb4bd067d37fb7c481830a00cd11284485994cb1f"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    selected_path = EXPERIMENT_DIR / "outputs" / "selected_config.json"
    protocol_path = EXPERIMENT_DIR / "locked_test_protocol.json"
    seal_path = (
        PROJECT_ROOT
        / "revision"
        / "exp01_manifests_sha256"
        / "outputs"
        / "manifest_seal.json"
    )
    if protocol_path.exists():
        raise RuntimeError("Le protocole du test enrichi est deja verrouille.")
    if not selected_path.exists():
        raise RuntimeError("La selection sur validation doit etre terminee avant le verrou.")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if selected.get("status") != "selected_on_validation_only":
        raise RuntimeError("Statut de selection invalide.")
    if selected.get("test_manifest_read") is not False:
        raise RuntimeError("La selection indique que le test a ete consulte.")
    if selected.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError("La selection ne repose pas sur les poids restaures de l'epoch 6.")
    if selected.get("validation_manifest_sha256") != EXPECTED_VALIDATION_MANIFEST_SHA256:
        raise RuntimeError("Le manifeste de validation a change.")
    if seal.get("status") != "sealed_clean":
        raise RuntimeError("Le sceau des manifestes n'est pas propre.")
    if seal["manifest_sha256"]["test"] != EXPECTED_TEST_MANIFEST_SHA256:
        raise RuntimeError("Le hash scelle du test a change.")

    chosen = selected["selected"]
    protocol = {
        "schema_version": 1,
        "status": "preregistered_before_final_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "One final evaluation of the validation-locked augmented LSTM "
            "configuration on the cleaned test split."
        ),
        "allowed_final_evaluations": 1,
        "model": {
            "architecture": "EfficientNetB0 + LSTM(256) + Dense(128)",
            "weights_path": selected["weights_path"],
            "weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "source_best_epoch": 6,
            "epoch_12_checkpoint_used": False,
        },
        "selected_configuration": {
            "path": "revision/exp05_augmented_evaluation/outputs/selected_config.json",
            "sha256": sha256_file(selected_path),
            "selected_on": "validation_only",
            "theta": float(chosen["threshold"]),
            "k": int(chosen["k"]),
            "n_frames": int(selected["n_frames"]),
            "stride_frames": int(selected["stride_frames"]),
        },
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
            "Same implementation as validation selection: 20-frame windows, "
            "stride 5, EfficientNetB0 features and the epoch-6 augmented LSTM head."
        ),
        "short_clip_policy": (
            "Repeat the last frame exactly as in validation selection so all "
            "candidate K values remain comparable."
        ),
        "no_post_test_adjustment": True,
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

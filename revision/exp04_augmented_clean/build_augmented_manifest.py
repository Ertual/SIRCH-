from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ENRICHMENT_FILES = (
    "basketball_01_WwMLOcTVUmc.mp4",
    "basketball_02_Czaz0AiIfkQ.mp4",
    "basketball_03_TWLLv2ikavc.mp4",
    "basketball_04_N8ir6EaDceI.mp4",
    "basketball_05_wtpobZDZv3A.mp4",
    "boxe_01_DBeTvfwjAug.mp4",
    "boxe_02_ewfo7pWZ3DU.mp4",
    "boxe_03_CycVhzGQazQ.mp4",
    "boxe_04_UsCSozlLlhs.mp4",
    "boxe_05_37dgPBSUosE.mp4",
    "football_01_eMr6OOmLlJs.mp4",
    "football_02_O1x9Nrp5TTU.mp4",
    "football_03_wnVJcY_ECH4.mp4",
    "football_04_7I2Ge845PXg.mp4",
    "football_05_qh7UrU9wVMo.mp4",
    "contemporaine_01_CoPUfvwcHmE.mp4",
    "contemporaine_02_bHAgWyVXytg.mp4",
    "contemporaine_03_LN9Y1JnGCUU.mp4",
    "contemporaine_04_ZN8iSru6_7k.mp4",
    "contemporaine_05_FHTD9GHVhx8.mp4",
    "hiphop_01_gqE3cXWHQRo.mp4",
    "hiphop_02_FYzyvVwYvWk.mp4",
    "hiphop_03_4JltE1OcI8U.mp4",
    "hiphop_04_Gf6YmTG-s3k.mp4",
    "hiphop_05_T-1uuAbQNRc.mp4",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    outputs = Path(__file__).resolve().parent / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    exp01 = project_root / "revision" / "exp01_manifests_sha256" / "outputs"
    principal_paths = {
        "train": exp01 / "train_manifest.csv",
        "validation": exp01 / "validation_manifest.csv",
        "test": exp01 / "test_manifest.csv",
        "hard_negative_v2": exp01 / "hard_negative_v2_manifest.csv",
    }
    seal_path = exp01 / "manifest_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_clean":
        raise RuntimeError(f"Sceau principal non propre: {seal.get('status')}")
    if int(seal.get("cross_split_sha256_duplicate_groups", -1)) != 0:
        raise RuntimeError("Le sceau principal contient encore des doublons inter-splits.")

    principal_rows: dict[str, list[dict[str, str]]] = {}
    for split, path in principal_paths.items():
        expected = seal["manifest_sha256"][split]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Manifeste {split} modifie: {actual} != {expected}")
        principal_rows[split] = read_csv(path)

    expected_sport = {name for name in EXPECTED_ENRICHMENT_FILES if name.startswith(("basketball_", "boxe_", "football_"))}
    expected_dance = set(EXPECTED_ENRICHMENT_FILES) - expected_sport
    source_dirs = {
        "test_sport": project_root / "datasets" / "test_sport",
        "test_danse": project_root / "datasets" / "test_danse",
    }
    actual_sport = {path.name for path in source_dirs["test_sport"].glob("*.mp4")}
    actual_dance = {path.name for path in source_dirs["test_danse"].glob("*.mp4")}
    if actual_sport != expected_sport:
        raise RuntimeError(f"Liste sport historique differente: {sorted(actual_sport ^ expected_sport)}")
    if actual_dance != expected_dance:
        raise RuntimeError(f"Liste danse historique differente: {sorted(actual_dance ^ expected_dance)}")

    enrichment_rows: list[dict[str, object]] = []
    for index, name in enumerate(EXPECTED_ENRICHMENT_FILES):
        group = "test_sport" if name in expected_sport else "test_danse"
        path = source_dirs[group] / name
        category, _, video_id = path.stem.split("_", 2)
        enrichment_rows.append(
            {
                "split": "train",
                "split_index": index,
                "dataset": "historical_enrichment_v1",
                "category": category,
                "label": 0,
                "label_name": "non_violence",
                "relative_path": f"{group}/{name}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "video_id": video_id,
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "license": "Creative Commons Attribution license (reuse allowed)",
                "historical_evidence": "SIRCH_CONTEXT_CODEX_V2_CLAUDECODE.md and test_faux_positifs_seuils.txt",
            }
        )

    hashes_by_split = {
        split: {row["sha256"] for row in rows}
        for split, rows in principal_rows.items()
    }
    enrichment_hashes = [str(row["sha256"]) for row in enrichment_rows]
    duplicate_enrichment = len(enrichment_hashes) - len(set(enrichment_hashes))
    overlaps = {
        split: sorted(set(enrichment_hashes) & hashes)
        for split, hashes in hashes_by_split.items()
    }
    if duplicate_enrichment or any(overlaps.values()):
        raise RuntimeError(
            f"Echec anti-fuite: doublons internes={duplicate_enrichment}, overlaps={overlaps}"
        )

    enrichment_fields = [
        "split", "split_index", "dataset", "category", "label", "label_name",
        "relative_path", "size_bytes", "sha256", "video_id", "source_url",
        "license", "historical_evidence",
    ]
    enrichment_path = outputs / "enrichment_v1_manifest.csv"
    write_csv(enrichment_path, enrichment_rows, enrichment_fields)

    combined_rows: list[dict[str, object]] = []
    combined_fields = [
        "split", "split_index", "dataset", "category", "label", "label_name",
        "relative_path", "size_bytes", "sha256", "source_group",
    ]
    for row in principal_rows["train"]:
        combined_rows.append(
            {
                **{field: row.get(field, "") for field in combined_fields[:-1]},
                "split": "train",
                "split_index": len(combined_rows),
                "source_group": "principal_train_clean",
            }
        )
    for row in enrichment_rows:
        combined_rows.append(
            {
                **{field: row.get(field, "") for field in combined_fields[:-1]},
                "split": "train",
                "split_index": len(combined_rows),
                "source_group": "historical_enrichment_v1",
            }
        )
    combined_path = outputs / "augmented_train_manifest.csv"
    write_csv(combined_path, combined_rows, combined_fields)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "sealed_clean",
        "historical_set_recovered": True,
        "historical_evidence": [
            "SIRCH_CONTEXT_CODEX_V2_CLAUDECODE.md at commit dbb71cb",
            "test_faux_positifs_seuils.txt",
            "physical source folders datasets/test_sport and datasets/test_danse",
        ],
        "principal_manifest_seal_sha256": sha256_file(seal_path),
        "source_manifest_sha256": {
            split: sha256_file(path) for split, path in principal_paths.items()
        },
        "enrichment_manifest_sha256": sha256_file(enrichment_path),
        "augmented_train_manifest_sha256": sha256_file(combined_path),
        "counts": {
            "principal_train": len(principal_rows["train"]),
            "enrichment": len(enrichment_rows),
            "augmented_train": len(combined_rows),
            "augmented_train_violence": sum(int(row["label"]) == 1 for row in combined_rows),
            "augmented_train_non_violence": sum(int(row["label"]) == 0 for row in combined_rows),
            "validation_unchanged": len(principal_rows["validation"]),
            "test_unchanged": len(principal_rows["test"]),
            "hard_negative_v2_unchanged": len(principal_rows["hard_negative_v2"]),
        },
        "internal_enrichment_sha256_duplicates": duplicate_enrichment,
        "overlap_sha256_groups": {split: len(values) for split, values in overlaps.items()},
        "old_test_sport_and_dance_status": "training_only_after_this_point",
        "hard_negative_evaluation_set": "hard_negative_v2_only",
    }
    report_path = outputs / "anti_leak_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = [
        "# Corpus augmente propre",
        "",
        "Les 25 videos historiques ont ete retrouvees. Aucun nouveau sourcing n'a ete necessaire.",
        "",
        f"- Train principal nettoye : {report['counts']['principal_train']}",
        f"- Enrichissement historique : {report['counts']['enrichment']} videos non violentes",
        f"- Train augmente total : {report['counts']['augmented_train']}",
        f"- Validation inchangee : {report['counts']['validation_unchanged']}",
        f"- Test principal inchange : {report['counts']['test_unchanged']}",
        f"- Hard-negative v2 inchange : {report['counts']['hard_negative_v2_unchanged']}",
        "- Doublons SHA-256 internes a l'enrichissement : 0",
        "- Chevauchement avec train principal : 0",
        "- Chevauchement avec validation : 0",
        "- Chevauchement avec test principal : 0",
        "- Chevauchement avec hard-negative v2 : 0",
        "",
        "Les anciens dossiers test_sport et test_danse deviennent exclusivement des sources d'entrainement.",
        "Ils ne doivent plus etre presentes comme un jeu de test. L'evaluation hard-negative utilise uniquement v2.",
    ]
    (outputs / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path


OUT = Path(__file__).resolve().parent / "outputs"
ANNOTATIONS = {
    "train/NonFight/rwf_train_nonfight_0542.avi": (
        "CCTV interieur, personnes en circulation", "proximite de plusieurs personnes et angle de surveillance; contexte potentiellement surinterprete", "contexte_surveillance", 1,
    ),
    "val/NonFight/rwf_val_nonfight_0034.avi": (
        "commerce, homme gesticulant devant comptoir", "gestes amples et flou de mouvement ressemblant a une lutte", "gestes_non_violents", 1,
    ),
    "train/NonFight/rwf_train_nonfight_0057.avi": (
        "escalier filme par CCTV", "deplacement rapide et silhouettes partiellement masquees par l'escalier", "mouvement_et_occlusion", 1,
    ),
    "NonViolence/NV_40.mp4": (
        "saut a la perche en stade", "mouvement explosif et chute sportive assimiles a une violence", "sport", 1,
    ),
    "NonViolence/NV_14.mp4": (
        "tribune sportive et drapeaux", "foule agitee, bras leves et occlusions entre personnes", "foule", 2,
    ),
    "train/NonFight/rwf_train_nonfight_0449.avi": (
        "rue nocturne filmee par CCTV", "faible eclairage et silhouettes proches rendent l'interaction ambigue", "faible_eclairage", 2,
    ),
    "train/Fight/rwf_train_fight_0444.avi": (
        "rue nocturne en noir et blanc, plusieurs personnes", "plan large, protagonistes petits et faible contraste", "plan_large_et_nuit", 2,
    ),
    "train/Fight/rwf_train_fight_0376.avi": (
        "rue nocturne en noir et blanc, bandeau televise", "protagonistes petits, contraste faible et bandeau graphique intrusif", "nuit_et_bandeau", 2,
    ),
    "val/Fight/rwf_val_fight_0006.avi": (
        "commerce vu d'en haut derriere un comptoir", "contact physique partiellement cache par le comptoir et angle distant", "occlusion", 1,
    ),
    "Violence/V_233.mp4": (
        "altercation exterieure en video verticale", "personnes petites dans l'image utile a cause des bandes noires et interaction breve", "cadrage_vertical", 2,
    ),
    "Violence/V_671.mp4": (
        "altercation sur terrain de sport", "contexte sportif et groupe de joueurs brouillent la distinction sport/bagarre", "sport_et_foule", 2,
    ),
    "Violence/V_634.mp4": (
        "altercation dans une galerie commerciale", "plan large et mouvements rapides des deux protagonistes", "plan_large_et_mouvement", 1,
    ),
}


def main() -> None:
    with (OUT / "all_locked_test_errors.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        annotation = ANNOTATIONS.get(row["relative_path"])
        if annotation is None:
            continue
        scene, mechanism, taxonomy, sheet_number = annotation
        selected.append({
            **row,
            "scene_type_observed": scene,
            "probable_failure_mechanism": mechanism,
            "taxonomy": taxonomy,
            "inspection_frames_pct": "10;50;90",
            "inspection_sheet": f"inspection/error_inspection_{sheet_number}.jpg",
            "interpretation_caveat": "Hypothesis from three sampled frames, not a causal explanation or full video annotation",
        })
    if len({row["relative_path"] for row in selected}) != len(ANNOTATIONS):
        raise RuntimeError("Some annotated cases are not errors")
    destination = OUT / "qualitative_error_examples.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(f"{len(selected)} model-error rows from {len(ANNOTATIONS)} visually reviewed videos")
    for path in ANNOTATIONS:
        entries = [row for row in selected if row["relative_path"] == path]
        summary = "; ".join(f"{r['model']} p={float(r['score_violence']):.3f} conf={float(r['confidence_predicted_class']):.3f}" for r in entries)
        print(f"{path}: {summary}")


if __name__ == "__main__":
    main()

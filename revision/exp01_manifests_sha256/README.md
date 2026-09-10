# Exp01 - Manifestes scelles SHA-256

## Objectif

Figer les ensembles utilises par SIRCH avant toute nouvelle evaluation :

- entrainement principal ;
- validation principale ;
- test principal ;
- test hard-negative v2 (sport, danse et scenes calmes).

Le script reproduit le split 70/15/15 du notebook original avec `seed=42`. Il
verifie aussi que le test reconstruit correspond exactement aux 599 videos deja
referencees dans `phase2_results/metriques_avancees_scores_599.csv`.

Chaque ligne contient la taille et le SHA-256 du fichier. Les contenus identiques
qui traversent plusieurs ensembles sont consignes dans un rapport et dans le
sceau. Aucun fichier source n'est retire automatiquement : le manifeste historique
reste reproductible et son etat de contamination est explicite.

## Commande

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp01_manifests_sha256\build_manifests.py
```

## Sorties

- `outputs/train_manifest.csv`
- `outputs/validation_manifest.csv`
- `outputs/test_manifest.csv`
- `outputs/hard_negative_v2_manifest.csv`
- `outputs/cross_split_duplicates.csv`
- `outputs/manifest_seal.json`
- `outputs/summary.md`

Les empreintes des quatre CSV inscrites dans `manifest_seal.json` constituent
le sceau du protocole. Toute modification ulterieure d'un manifeste change son
empreinte.

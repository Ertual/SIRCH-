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
sont resolus avec la priorite deterministe `train > validation > test` : une seule
occurrence canonique est conservee et les autres sont exclues des manifestes.
Aucun fichier video source n'est supprime.

Le seul groupe contradictoire, `V_504.mp4` / `NV_226.mp4`, a ete examine sur 20
images reparties sur tout le clip. Il montre un match de tennis : son etiquette
canonique est donc corrigee en `non_violence`. La preuve visuelle est conservee
dans `outputs/ground_truth_conflict_contact_sheet.png`.

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
- `outputs/duplicate_resolution.csv`
- `outputs/ground_truth_conflict_contact_sheet.png`
- `outputs/manifest_seal.json`
- `outputs/summary.md`

Les empreintes des quatre manifestes et du rapport de resolution inscrites dans
`manifest_seal.json` constituent le sceau du protocole. Apres resolution,
`cross_split_duplicates.csv` doit contenir uniquement son en-tete et le statut du
sceau doit etre `sealed_clean`.

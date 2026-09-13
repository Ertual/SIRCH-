# exp11 - Correction visuelle et splits groupes par scene

Cette experience administrative corrige les conflits de labels identifies par l'audit
perceptuel `exp10`, puis reconstruit les manifestes principaux en maintenant chaque
scene/source dans un seul split.

Elle ne lance aucun entrainement.

## Procedure

1. `render_label_conflict_review.py` produit 17 planches temporelles couvrant les 129
   videos impliquees dans les 90 paires probables aux labels opposes.
2. `render_zoom_review.py` produit une seconde lecture a 24 instants pour les cas ambigus.
3. `build_grouped_manifests.py` enregistre les decisions, applique les corrections,
   fusionne les 191 groupes perceptuels avec les groupes SHA-256 exacts, et ajoute les
   videos isolees comme singletons.
4. Une optimisation MILP attribue chaque groupe entier a un split. Elle impose les
   tailles 2785/597/597 et les comptes de classes cibles, puis minimise le nombre de
   videos qui changent de split.
5. Les manifestes produits sont scelles par SHA-256 et soumis a des controles de
   non-chevauchement par groupe et par hash exact.

Les 25 videos d'enrichissement historique restent exclusivement dans le manifeste
d'entrainement enrichi. Elles ne participent pas aux tailles des splits principaux.

## Execution reproductible

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp11_grouped_scene_splits\render_label_conflict_review.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp11_grouped_scene_splits\render_zoom_review.py V096 V102 V109 V113 V116 --output-dir revision\exp11_grouped_scene_splits\outputs\label_correction_zoom
C:\SIRCH_ENV\Scripts\python.exe revision\exp11_grouped_scene_splits\build_grouped_manifests.py
```

Consulter `outputs/summary.md` et `outputs/integrity_checks.json` avant toute reprise
d'entrainement.

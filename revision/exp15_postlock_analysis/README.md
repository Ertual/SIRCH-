# Exp15 - Analyses complementaires post-verrouillage

Cette experience ne reentraine rien et ne reselectionne aucun seuil. Elle utilise les
quatre configurations deja verrouillees de `exp14_grouped_factorial_evaluation`.

- `analyze_saved_scores.py` verifie l'average precision (PR-AUC) sur les 597
  predictions archivees, reconstruit le premier franchissement de seuil sur les
  297 flux violents caches et produit la liste exhaustive des erreurs.
- `benchmark_cpu_grouped.py` mesure des predictions CPU sur dix clips de la
  validation groupee, en utilisant les quatre fichiers `.h5` finaux dont les
  SHA-256 figurent dans les metriques du test verrouille.
- `make_inspection_sheets.py` extrait trois images de chacun des douze cas
  qualitatifs; `annotate_errors.py` joint les observations visuelles aux scores.

Depuis la racine du depot, avec l'environnement Python SIRCH :

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp15_postlock_analysis\analyze_saved_scores.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp15_postlock_analysis\benchmark_cpu_grouped.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp15_postlock_analysis\make_inspection_sheets.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp15_postlock_analysis\annotate_errors.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp15_postlock_analysis\seal_outputs.py
```

Le rapport et tous les fichiers produits se trouvent sous `outputs/`. Le
benchmark utilise les medias de validation, pas les videos du test pour sa
mesure de performance. Les scores du test sont seulement relus, jamais recalcules
par inference.

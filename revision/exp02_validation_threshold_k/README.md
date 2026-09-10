# Exp02 - Selection de theta et K sur validation

## Objectif

Selectionner le seuil `theta` et la longueur de moyenne glissante `K` sans lire
le jeu de test principal.

Le script accepte uniquement `validation_manifest.csv`. Il verifie son SHA-256
contre le sceau propre de l'experience 01, exige zero doublon SHA-256 inter-splits
et refuse toute ligne dont le champ `split` n'est pas `validation`.

## Protocole preregistre

- Modele : `C:\SIRCH_ENV\models\sirch_model.h5` (modele original, epoch 4).
- Execution : CPU uniquement.
- Sequence du modele : 20 frames consecutives.
- Pas entre deux scores : 5 frames, comme le mode temps reel par defaut.
- Valeurs de K : 1, 3, 5, 7 et 10.
- Seuils : 0,30 a 0,90 par pas de 0,05.
- Decision video : positive si au moins une moyenne complete de K scores atteint
  le seuil.
- Les clips trop courts sont completes par repetition de leur derniere frame afin
  que tous soient compares sur la meme grille de K.
- Critere principal : F1 video sur validation.
- Departages successifs : balanced accuracy, rappel, plus petit taux de faux
  positifs, plus petit K, puis seuil le plus proche de 0,50.

Le jeu de test n'est ni charge ni reference par le script de selection.

## Commande

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp02_validation_threshold_k\select_threshold_k.py
```

## Sorties

- `outputs/validation_score_streams.csv`
- `outputs/selection_grid.csv`
- `outputs/selected_config.json`
- `outputs/summary.md`
- `outputs/cache/` pour la reprise apres interruption.

Le cache est indexe par SHA-256 et n'est pas versionne. Les resultats publies sont
toujours recalcules sur la liste exacte du manifeste scelle courant.

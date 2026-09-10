# Exp04 - Reentrainement LSTM augmente propre

## But

Reentrainer le modele LSTM augmente sans contaminer la validation, le test
principal ou le corpus hard-negative v2.

L'ancien modele `sirch_model_augmente.h5` a ete entraine apres une nouvelle
partition de l'ensemble complet enrichi. Il ne peut donc pas servir de preuve
finale face au test principal scelle actuel, car la composition des splits
n'etait pas identique. Ce modele historique n'est ni modifie ni supprime.

## Corpus

- Train principal nettoye et scelle : 2 793 videos.
- Enrichissement historique retrouve : 25 videos non violentes, soit 15 sport
  et 10 danse.
- Train augmente : 2 818 videos.
- Validation inchangee : 594 videos.
- Test principal inchange : 592 videos, non lu pendant l'entrainement.
- Hard-negative v2 inchange : 30 videos, non utilise pour l'entrainement.

Les 25 videos historiques sont identifiees dans
`outputs/enrichment_v1_manifest.csv`. Elles ont toutes un SHA-256 distinct de
train, validation, test principal et hard-negative v2. Les anciens dossiers
`test_sport` et `test_danse` deviennent des sources d'entrainement uniquement.

## Architecture et hyperparametres

- EfficientNetB0 ImageNet gele, puis LSTM 256.
- Deux dropouts : 0,5 puis 0,3.
- Adam, learning rate initial 1e-4.
- Batch 8, N=20, maximum 30 epochs, seed 42.
- EarlyStopping sur `val_loss`, patience 6, `restore_best_weights=True`.
- ReduceLROnPlateau, facteur 0,5, patience 3, minimum 1e-6.
- Checkpoint de poids apres chaque epoch et CSV appendable pour la reprise.
- CPU seul, 8 threads TensorFlow demandes.

## Commandes

```powershell
C:\SIRCH_ENV\Scripts\python.exe revision\exp04_augmented_clean\build_augmented_manifest.py
C:\SIRCH_ENV\Scripts\python.exe revision\exp04_augmented_clean\train_lstm_augmented_clean.py
```

Les modeles, checkpoints et journaux sont ecrits uniquement dans
`C:\SIRCH_ENV\models\revision\exp04_augmented_clean`. Aucun modele existant
n'est ecrase.

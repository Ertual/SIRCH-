# Selection theta/K sur validation

Le jeu de test principal n'a pas ete lu.

- Videos de validation : 594
- Videos completees par repetition : 13
- K selectionne : 7
- Seuil selectionne : 0.70
- Accuracy validation : 0.8670
- Balanced accuracy validation : 0.8671
- Precision validation : 0.8489
- Rappel validation : 0.8919
- F1 validation : 0.8699
- Matrice : TN=251, FP=47, FN=32, TP=264

## Comparaison avant/apres deduplication

| Version validation | N | theta | K | Accuracy | Precision | Rappel | F1 | TN/FP/FN/TP |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Avant correction | 599 | 0.70 | 7 | 0.8681 | 0.8503 | 0.8930 | 0.8711 | 253/47/32/267 |
| Apres correction | 594 | 0.70 | 7 | 0.8670 | 0.8489 | 0.8919 | 0.8699 | 251/47/32/264 |

Le nettoyage change legerement les metriques, mais pas le choix de theta=0.70 et K=7.

Le manifeste utilise est scelle sans doublon SHA-256 entre les splits.

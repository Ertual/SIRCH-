# SIRCH - Synthese Phase 2

Date : 2026-06-09

Cette synthese consolide les resultats de la phase 2 scientifique du projet SIRCH.
Le modele de reference reste `C:\SIRCH_ENV\models\sirch_model.h5` et le modele
augmente est `C:\SIRCH_ENV\models\sirch_model_augmente.h5`.

## Etape A - Modele augmente et faux positifs

Evaluation principale sur 599 videos :

| Modele | Seuil | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.50 | 86.81% | 82.74% | 92.98% | 87.56% | 58 | 21 |
| Augmente | 0.50 | 91.99% | 88.62% | 96.32% | 92.31% | 37 | 11 |
| Original | 0.65 | 87.31% | 87.04% | 87.63% | 87.33% | 39 | 37 |
| Augmente | 0.65 | 91.65% | 91.64% | 91.64% | 91.64% | 25 | 25 |
| Original | 0.70 | 86.64% | 87.12% | 85.95% | 86.53% | 38 | 42 |
| Augmente | 0.70 | 91.82% | 93.10% | 90.30% | 91.68% | 20 | 29 |

Faux positifs sur les nouvelles videos propres v2 :

| Modele | Seuil | Sport v2 | Danse v2 | Calme v2 | Global |
|---|---:|---:|---:|---:|---:|
| Original | 0.50 | 3/15 = 20.00% | 8/10 = 80.00% | 3/5 = 60.00% | 14/30 = 46.67% |
| Original | 0.65 | 2/15 = 13.33% | 7/10 = 70.00% | 3/5 = 60.00% | 12/30 = 40.00% |
| Original | 0.70 | 2/15 = 13.33% | 5/10 = 50.00% | 3/5 = 60.00% | 10/30 = 33.33% |
| Augmente | 0.50 | 4/15 = 26.67% | 7/10 = 70.00% | 4/5 = 80.00% | 15/30 = 50.00% |
| Augmente | 0.65 | 3/15 = 20.00% | 6/10 = 60.00% | 1/5 = 20.00% | 10/30 = 33.33% |
| Augmente | 0.70 | 2/15 = 13.33% | 5/10 = 50.00% | 1/5 = 20.00% | 8/30 = 26.67% |

## Etape B - ROC/AUC et Precision-Recall

| Modele | ROC AUC | Average Precision | PR AUC |
|---|---:|---:|---:|
| Original | 0.9446 | 0.9415 | 0.9414 |
| Augmente | 0.9767 | 0.9738 | 0.9737 |

Le modele augmente presente une meilleure separabilite globale sur le jeu de test
principal de 599 videos.

## Etape C - Ablation partielle

| Configuration | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| EfficientNetB0 + LSTM, N=20 | 86.81% | 82.74% | 92.98% | 87.56% | 94.46% |
| Ablation sans dynamique temporelle | 87.31% | 83.28% | 93.31% | 88.01% | 94.11% |
| N=10 reechantillonne vers 20 | 87.48% | 83.14% | 93.98% | 88.23% | 94.32% |
| N=30 compresse vers 20 | 87.15% | 82.84% | 93.65% | 87.91% | 94.51% |

Note methodologique : l'ablation EfficientNetB0 seul est une ablation d'inference
non destructive. Le modele n'a pas de classifieur frame-par-frame separe ; chaque
frame a donc ete repetee sur 20 pas temporels puis les scores ont ete moyennes.

## Etape D - Biais des datasets

Resultats a theta = 0.50 :

| Modele | Dataset | FP | Taux FP | FN | Taux FN | F1 |
|---|---|---:|---:|---:|---:|---:|
| Original | RLVS | 17/150 | 11.33% | 3/145 | 2.07% | 93.42% |
| Original | RWF-2000 | 41/150 | 27.33% | 18/154 | 11.69% | 82.18% |
| Augmente | RLVS | 10/150 | 6.67% | 2/145 | 1.38% | 95.97% |
| Augmente | RWF-2000 | 27/150 | 18.00% | 9/154 | 5.84% | 88.96% |

Le biais est plus marque dans RWF-2000, notamment sur les faux positifs.
Le modele augmente reduit ce biais, mais ne le supprime pas completement.

## Etape E - Delai d'alerte

Modele original, theta = 0.50, 10 videos violentes du test :

| K | Videos detectees | Delai moyen a 25 fps | Ecart-type |
|---:|---:|---:|---:|
| 3 | 10/10 | 0.864 s | 0.076 s |
| 5 | 10/10 | 0.940 s | 0.063 s |
| 7 | 10/10 | 1.016 s | 0.051 s |

Note methodologique : le delai est mesure depuis le debut du clip, car les datasets
ne fournissent pas d'annotation precise du debut reel de la violence.

## Fichiers produits

- `C:\SIRCH_ENV\models\resultats_comparaison_modeles.csv`
- `C:\SIRCH_ENV\models\resultats_faux_positifs_v2.csv`
- `C:\SIRCH_ENV\models\metriques_avancees.csv`
- `C:\SIRCH_ENV\models\ablation_results.csv`
- `C:\SIRCH_ENV\models\ablation_fenetre.csv`
- `C:\SIRCH_ENV\models\biais_datasets.csv`
- `C:\SIRCH_ENV\models\delai_alerte.csv`
- `C:\SIRCH_ENV\models\figures\`

## Conclusion scientifique

La phase 2 confirme que SIRCH obtient de bonnes performances globales sur le test
principal, mais que les faux positifs sur activites physiques intenses restent un
enjeu important. L'analyse par dataset montre que le biais est particulierement
visible dans RWF-2000. Le modele augmente ameliore les metriques globales et reduit
plusieurs erreurs sur le test principal, mais les videos v2 montrent que la robustesse
hors distribution reste a documenter avec prudence.

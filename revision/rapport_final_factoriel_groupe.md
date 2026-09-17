# SIRCH - rapport final du factoriel groupe par scene

Date : 17 septembre 2026. Version des artefacts analyses : commit `e832716833b817793bfa58a6fcb2959bdd3a4564` sur `origin/main`.

## Protocole et lecture des resultats

Les quatre cellules LSTM/GRU x original/enrichi ont ete reentrainees sur les splits regroupes par scene/source. La nouvelle validation comporte 597 videos ; elle seule a servi a choisir `theta` et `K`. Les quatre configurations ont ete verrouillees avant l'unique evaluation du nouveau test de 597 videos (300 non violentes, 297 violentes). Les 30 videos hard-negative ont ensuite ete evaluees sans nouveau reglage. La classe positive est la violence. Dans les matrices ci-dessous, l'ordre est **TN / FP / FN / TP**.

La revue visuelle prealable a examine 90 paires aux labels opposes et corrige 5 labels. Le regroupement a deplace 279 videos entre splits ; aucune violation groupe/split ou doublon SHA-256 exact entre splits n'a ete relevee. L'audit perceptuel complementaire hard-negative contre le nouveau train a conclu a 0 quasi-doublon probable ou possible sur les 30 videos. Le nouveau manifeste de test est scelle par SHA-256 `a37aa9cba85d28ac315c1ea813c902330a3e99525042c13012023914e0a9887a`.

## Quatre modeles sur le test groupe

Toutes les metriques en pourcentage, sauf ROC-AUC. Les poids retenus sont les meilleurs poids restaures, pas le dernier checkpoint.

| Modele | Meilleur epoch | theta | K | Accuracy | Precision | Rappel | F1 | ROC-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LSTM original | 3 | 0,70 | 5 | 86,10 % | 82,04 % | 92,26 % | 86,85 % | 0,9495 | 240 / 60 / 23 / 274 |
| LSTM enrichi | 3 | 0,70 | 7 | 86,26 % | 82,28 % | 92,26 % | 86,98 % | 0,9557 | 241 / 59 / 23 / 274 |
| GRU original | 4 | 0,65 | 5 | 87,27 % | 84,86 % | 90,57 % | 87,62 % | 0,9510 | 252 / 48 / 28 / 269 |
| GRU enrichi | 4 | 0,65 | 10 | 86,77 % | 87,33 % | 85,86 % | 86,59 % | 0,9453 | 263 / 37 / 42 / 255 |

Le GRU original a la meilleure accuracy et le meilleur F1 du nouveau test. Le LSTM enrichi a la meilleure ROC-AUC. Le GRU enrichi reduit les faux positifs sur ce test (37 contre 48 pour le GRU original), au prix de davantage de faux negatifs (42 contre 28). Ces constats descriptifs ne constituent pas, a eux seuls, une preuve de superiorite statistique.

## McNemar et bootstrap apparie

Les comparaisons original/enrichi sont faites separement au sein de chaque architecture, sur les **memes 597 videos**. `b` signifie original correct / enrichi incorrect ; `c` signifie original incorrect / enrichi correct. Le bootstrap percentile apparie utilise 10 000 repetitions. Les deltas sont *enrichi moins original* ; les intervalles sont a 95 %.

| Architecture | b | c | McNemar exact, p | Delta accuracy (points) [IC 95 %] | Delta F1 (points) [IC 95 %] | Delta ROC-AUC [IC 95 %] |
|---|---:|---:|---:|---|---|---|
| LSTM | 18 | 19 | 1,0000 | +0,17 [-1,84 ; +2,18] | +0,14 [-1,72 ; +1,97] | +0,0062 [-0,0024 ; +0,0149] |
| GRU | 24 | 21 | 0,7660 | -0,50 [-2,68 ; +1,68] | -1,03 [-3,26 ; +1,10] | -0,0057 [-0,0132 ; +0,0015] |

Aucune difference d'accuracy original/enrichi n'est detectee au seuil de 5 % dans l'une ou l'autre architecture. Pour le GRU, l'enrichissement deplace toutefois le compromis : rappel -4,71 points [IC 95 % -7,80 ; -1,97] et specificite +3,67 points [IC 95 % +0,34 ; +6,90]. Ces intervalles sont des analyses par metrique ; ils ne changent pas la conclusion du test de McNemar sur l'accuracy globale.

## Interaction architecture x enrichissement

Effet de l'enrichissement sur l'accuracy : **+0,17 point** pour le LSTM et **-0,50 point** pour le GRU. La difference de differences `(GRU enrichi - GRU original) - (LSTM enrichi - LSTM original)` vaut **-0,67 point**, IC bootstrap 95 % **[-3,52 ; +2,18] points**. Le test apparie exact par inversion de signes donne **p = 0,7317**. Aucune interaction n'est detectee au seuil de 5 % ; cela ne demontre pas l'absence de toute interaction.

## Corpus hard-negative

Les 30 videos sont non violentes : sport 15, danse 10, calme 5. Un faux positif est donc une video signalee comme violente. Les `theta`/`K` sont ceux verrouilles plus haut, sans optimisation sur ce corpus. Les fractions sont fournies pour rendre visibles les petits effectifs.

| Modele | Sport (n=15) | Danse (n=10) | Calme (n=5) | Ensemble (n=30) |
|---|---:|---:|---:|---:|
| LSTM original | 6/15 (40,00 %) | 9/10 (90,00 %) | 3/5 (60,00 %) | 18/30 (60,00 %) |
| LSTM enrichi | 5/15 (33,33 %) | 5/10 (50,00 %) | 2/5 (40,00 %) | 12/30 (40,00 %) |
| GRU original | 7/15 (46,67 %) | 8/10 (80,00 %) | 2/5 (40,00 %) | 17/30 (56,67 %) |
| GRU enrichi | 5/15 (33,33 %) | 9/10 (90,00 %) | 2/5 (40,00 %) | 16/30 (53,33 %) |

Le LSTM enrichi presente le taux global le plus faible sur ce petit corpus (40 %), mais les faux positifs restent frequents, surtout en danse. Ces taux sont descriptifs ; avec 5 videos dans la categorie calme, une seule erreur represente 20 points.

## Avant/apres le regroupement par scene

L'ancien test propre comptait **592 videos** et provenait du manifeste initial ; le nouveau test groupe en compte **597**. Les quatre modeles ont ete reentraines et leurs propres `theta`/`K` reselectionnes sur la nouvelle validation. Les tableaux ci-dessous comparent donc deux protocoles complets, **pas les memes poids sur exactement les memes videos**. Les ecarts ne permettent pas d'attribuer causalement toute baisse a une fuite precise, ni de calculer un test apparié avant/apres sur ces seules valeurs agregees.

| Modele | theta/K avant | theta/K apres | Accuracy avant | Accuracy apres | Delta accuracy |
|---|---|---|---:|---:|---:|
| LSTM original | 0,70 / 7 | 0,70 / 5 | 88,01 % | 86,10 % | -1,91 point |
| LSTM enrichi | 0,60 / 7 | 0,70 / 7 | 90,88 % | 86,26 % | -4,61 points |
| GRU original | 0,60 / 10 | 0,65 / 5 | 87,67 % | 87,27 % | -0,40 point |
| GRU enrichi | 0,60 / 7 | 0,65 / 10 | 88,18 % | 86,77 % | -1,41 point |

| Modele | Precision avant -> apres | Rappel avant -> apres | F1 avant -> apres | ROC-AUC avant -> apres | TN / FP / FN / TP avant -> apres |
|---|---|---|---|---|---|
| LSTM original | 85,09 % -> 82,04 % | 92,26 % -> 92,26 % | 88,53 % -> 86,85 % | 0,9527 -> 0,9495 | 247/48/23/274 -> 240/60/23/274 |
| LSTM enrichi | 89,32 % -> 82,28 % | 92,93 % -> 92,26 % | 91,09 % -> 86,98 % | 0,9665 -> 0,9557 | 262/33/21/276 -> 241/59/23/274 |
| GRU original | 83,73 % -> 84,86 % | 93,60 % -> 90,57 % | 88,39 % -> 87,62 % | 0,9553 -> 0,9510 | 241/54/19/278 -> 252/48/28/269 |
| GRU enrichi | 83,88 % -> 87,33 % | 94,61 % -> 85,86 % | 88,92 % -> 86,59 % | 0,9572 -> 0,9453 | 241/54/16/281 -> 263/37/42/255 |

L'avantage d'accuracy du LSTM enrichi sur le LSTM original passe de **+2,87 points** avant regroupement a **+0,17 point** apres regroupement. Pour le GRU, l'ecart enrichi moins original passe de **+0,51** a **-0,50 point**. Sur le nouveau test commun, ces deux ecarts d'accuracy ne sont pas significatifs selon les comparaisons appariees ci-dessus.

## Sources et tracabilite

- Ancien test et configurations : `revision/exp02_validation_threshold_k/outputs/final_test_locked/final_test_metrics.json`, `revision/exp05_augmented_evaluation/outputs/final_test_locked/final_test_metrics.json`, et les deux `final_test_metrics.json` sous `revision/exp08_gru_factorial_evaluation/outputs/final_test_locked/`.
- Nouveaux poids, epochs, logs et courbes : `revision/exp13_grouped_retraining/outputs/`.
- Verrouillage, metriques par modele, predictions, matrices et ROC : `revision/exp14_grouped_factorial_evaluation/outputs/locked_protocol.json` et `final_test_locked/`.
- McNemar/bootstrap : `revision/exp14_grouped_factorial_evaluation/outputs/paired_comparisons/` ; interaction : `factorial_2x2/` ; hard-negative : `hard_negative_locked/`.
- Revue des labels et splits : `revision/exp11_grouped_scene_splits/outputs/summary.md` ; audit hard-negative/train : `revision/exp12_hard_negative_leakage_audit/outputs/summary.md`.
- Les fichiers `artifact_manifest.json` des experiences 13 et 14 portent les SHA-256 des artefacts sources. Ce rapport autonome est ajoute hors de leurs repertoires `outputs/` pour ne pas modifier retrospectivement ces manifestes scelles.

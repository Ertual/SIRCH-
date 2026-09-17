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

## Complements post-verrouillage (17 septembre 2026)

Les quatre points ci-dessous sont documentes en detail, avec mesures brutes,
methodes et exemples visuels, dans
`revision/exp15_postlock_analysis/outputs/summary.md`. Aucun modele n'a ete
reentraine, aucun theta/K n'a ete modifie et aucune nouvelle inference sur le
test n'a ete lancee.

**PR-AUC (average precision) du test groupe verrouille, 597 entrees :**

| Modele | LSTM original | LSTM enrichi | GRU original | GRU enrichi |
|---|---:|---:|---:|---:|
| PR-AUC | 0,952651 | 0,956246 | 0,951444 | 0,946416 |

**Benchmark CPU des quatre modeles verrouilles :** Intel Core i5-12450H,
8 coeurs physiques/12 logiques, 7,73 Gio de RAM, Windows build 26200,
Python 3.10.11, TensorFlow/Keras 2.12.0, OpenCV 4.11.0, NumPy 1.23.5 ;
batch de 1 clip de 20 frames, TensorFlow 8 threads intra-op
et 2 inter-op. Dix videos de validation groupee (5 par classe), 5 warm-ups et
20 mesures par modele. Les temps excluent le chargement du modele.

| Modele | Inference pure moyenne (ms/clip) | Bout en bout moyen (ms/clip) | Debit bout en bout (clips/s) | CPU systeme moyen | RAM processus pic (Mio) |
|---|---:|---:|---:|---:|---:|
| LSTM original | 1 322 | 1 952 | 0,512 | 41,7 % | 793 |
| LSTM enrichi | 1 290 | 1 921 | 0,520 | 37,1 % | 774 |
| GRU original | 1 294 | 1 931 | 0,518 | 37,7 % | 812 |
| GRU enrichi | 1 324 | 1 972 | 0,507 | 38,3 % | 839 |

Le debit de 20 frames par clip normalise correspond a environ 10,1-10,4
frames/s bout en bout, **pas a un debit de camera continue**. Pendant ces
mesures, environ 12 threads du processus ont consomme du CPU par fenetre de
0,5 s (pics de 22-25 ; non simultanes), tandis que 59-63 threads OS etaient
presents en moyenne. Les usages CPU processus moyens etaient de 262-270 %
au sens psutil, ou 100 % vaut un coeur logique. La vitesse a varie entre
passages exploratoires ; les chiffres sont descriptifs de cette machine.

| Modele | CPU processus moyen / pic | RAM processus moyenne / pic (Mio) | Threads CPU actifs moyens / pic par 0,5 s | Threads OS presents moyens / pic |
|---|---:|---:|---:|---:|
| LSTM original | 261,9 / 329,3 % | 583 / 793 | 12,4 / 25 | 63,0 / 68 |
| LSTM enrichi | 268,0 / 336,3 % | 646 / 774 | 12,2 / 24 | 59,7 / 66 |
| GRU original | 270,2 / 319,2 % | 682 / 812 | 12,1 / 22 | 58,7 / 64 |
| GRU enrichi | 262,2 / 326,3 % | 733 / 839 | 11,9 / 22 | 60,8 / 64 |

Les threads actifs sont ceux ayant consomme du temps CPU dans une fenetre de
0,5 s, pas autant de threads executes simultanement. La RAM correspond au
processus Python complet et les quatre modeles ont ete charges successivement
dans le meme processus. Mesures brutes :
`revision/exp15_postlock_analysis/outputs/cpu_benchmark_raw.csv` et
`cpu_benchmark_summary.csv` ; environnement et echantillon :
`cpu_benchmark_environment.json`.

**Delai d'alerte a 25 fps, depuis le debut du clip** sur les 297 videos
violentes du test, avec theta/K verrouilles et les scores deja caches :

| Modele | Detection | Delai moyen parmi detectees | Mediane | P95 |
|---|---:|---:|---:|---:|
| LSTM original | 274/297 | 1,847 s | 1,600 s | 3,000 s |
| LSTM enrichi | 274/297 | 2,196 s | 2,000 s | 3,000 s |
| GRU original | 269/297 | 1,917 s | 1,600 s | 3,400 s |
| GRU enrichi | 255/297 | 2,875 s | 2,600 s | 4,000 s |

Ce delai n'est pas la latence murale d'une alerte en production ; le debut
reel de la violence n'est pas annote. L'ancien chiffre de 0,940 s porte sur
un autre protocole et n'est pas directement comparable.

**Erreurs qualitatives :** 12 videos et 39 lignes modele-erreur inspectees.
Les faux positifs comprennent pole vault et foule sportive (mouvement et bras
leves), gesticulations non violentes et scenes CCTV ambigues. Les faux
negatifs comprennent plans larges ou nocturnes, comptoir occultant le contact,
video verticale avec sujets petits et altercation en contexte sportif. Chaque
cas, son `p(violence)`, la confiance de la classe predite, le dataset source,
le type de scene et le mecanisme probable figurent dans
`revision/exp15_postlock_analysis/outputs/qualitative_error_examples.csv` ;
les 12 extraits visuels sont dans `outputs/inspection/`. Ces mecanismes restent
des hypotheses issues de trois frames inspectees par video.

Le tableau donne un modele errone representatif par video. `pV` est le score
de violence ; `conf.` vaut `pV` pour un faux positif et `1-pV` pour un faux
negatif. Ces scores ne sont pas calibres ; avec theta > 0,5, une prediction
negative peut avoir une `conf.` inferieure a 0,5. Le CSV cite ci-dessus donne
les valeurs exactes pour **tous** les modeles errones sur chaque cas.

| Erreur | Video et dataset source | Modele | pV | conf. | Type de scene | Mecanisme probable |
|---|---|---|---:|---:|---|---|
| FP | `rwf_train_nonfight_0542.avi` (RWF-2000) | LSTM original | 0,976 | 0,976 | CCTV interieur | Proximite de personnes et contexte de surveillance surinterpretes |
| FP | `rwf_val_nonfight_0034.avi` (RWF-2000) | LSTM enrichi | 0,973 | 0,973 | Commerce | Gestes amples ressemblant a une lutte |
| FP | `rwf_train_nonfight_0057.avi` (RWF-2000) | GRU original | 0,887 | 0,887 | Escalier CCTV | Deplacement rapide et silhouettes masquees |
| FP | `NV_40.mp4` (RLVS) | GRU enrichi | 0,755 | 0,755 | Saut a la perche | Mouvement explosif et chute sportive |
| FP | `NV_14.mp4` (RLVS) | GRU original | 0,957 | 0,957 | Tribune sportive | Foule agitee et bras leves |
| FP | `rwf_train_nonfight_0449.avi` (RWF-2000) | LSTM enrichi | 0,937 | 0,937 | Rue nocturne CCTV | Faible eclairage, interaction ambigue |
| FN | `rwf_train_fight_0444.avi` (RWF-2000) | LSTM original | 0,297 | 0,703 | Rue nocturne noir/blanc | Plan large, sujets petits, faible contraste |
| FN | `rwf_train_fight_0376.avi` (RWF-2000) | GRU original | 0,231 | 0,769 | Rue nocturne avec bandeau TV | Sujets petits et bandeau intrusif |
| FN | `rwf_val_fight_0006.avi` (RWF-2000) | LSTM enrichi | 0,631 | 0,369 | Commerce vu d'en haut | Comptoir masquant le contact |
| FN | `V_233.mp4` (RLVS) | LSTM original | 0,555 | 0,445 | Altercation en video verticale | Sujets petits dans l'image utile |
| FN | `V_671.mp4` (RLVS) | LSTM enrichi | 0,603 | 0,397 | Altercation sur terrain de sport | Contexte sportif et groupe ambigus |
| FN | `V_634.mp4` (RLVS) | GRU enrichi | 0,588 | 0,412 | Galerie commerciale | Plan large et mouvement rapide |

Limite du test verrouille : 597 entrees correspondent a 596 contenus SHA-256
uniques, car deux chemins RWF-2000 non violents referencent le meme contenu
dans le meme split test. Ce n'est pas une fuite entre splits ; les metriques
historiques n'ont pas ete modifiees.

## Sources et tracabilite

- Ancien test et configurations : `revision/exp02_validation_threshold_k/outputs/final_test_locked/final_test_metrics.json`, `revision/exp05_augmented_evaluation/outputs/final_test_locked/final_test_metrics.json`, et les deux `final_test_metrics.json` sous `revision/exp08_gru_factorial_evaluation/outputs/final_test_locked/`.
- Nouveaux poids, epochs, logs et courbes : `revision/exp13_grouped_retraining/outputs/`.
- Verrouillage, metriques par modele, predictions, matrices et ROC : `revision/exp14_grouped_factorial_evaluation/outputs/locked_protocol.json` et `final_test_locked/`.
- McNemar/bootstrap : `revision/exp14_grouped_factorial_evaluation/outputs/paired_comparisons/` ; interaction : `factorial_2x2/` ; hard-negative : `hard_negative_locked/`.
- Revue des labels et splits : `revision/exp11_grouped_scene_splits/outputs/summary.md` ; audit hard-negative/train : `revision/exp12_hard_negative_leakage_audit/outputs/summary.md`.
- PR-AUC, benchmark CPU des quatre modeles, delais et revue qualitative : `revision/exp15_postlock_analysis/outputs/summary.md` et ses CSV/JSON/planches d'inspection.
- Les fichiers `artifact_manifest.json` des experiences 13 et 14 portent les SHA-256 des artefacts sources. Ce rapport autonome est ajoute hors de leurs repertoires `outputs/` pour ne pas modifier retrospectivement ces manifestes scelles.

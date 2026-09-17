# SIRCH - analyses complementaires des modeles groupes verrouilles

Date : 17 septembre 2026. Les configurations sont celles de
`revision/exp14_grouped_factorial_evaluation/outputs/locked_protocol.json`.
Aucun entrainement, nouveau choix de theta/K ou nouvel appel d'inference sur le
test n'a ete effectue pour cette analyse. La classe positive est la violence.

Abreviations : **LO** = LSTM original, **LA** = LSTM enrichi, **GO** = GRU
original, **GA** = GRU enrichi.

## 1. PR-AUC du test groupe verrouille

La PR-AUC est ici l'**average precision** de `sklearn`, calculee a partir du
score video deja archive (maximum des moyennes glissantes de taille K). Les
valeurs recalculees sur les 597 lignes correspondent aux JSON d'evaluation
finale a moins de 1e-9. La prevalence positive est 297/597, soit 49,75 %.

| Modele | theta | K | PR-AUC / AP |
|---|---:|---:|---:|
| LO | 0,70 | 5 | 0,952651 |
| LA | 0,70 | 7 | 0,956246 |
| GO | 0,65 | 5 | 0,951444 |
| GA | 0,65 | 10 | 0,946416 |

La courbe precision-rappel de chaque modele etait deja archivee dans
`exp14_grouped_factorial_evaluation/outputs/final_test_locked/<modele>/precision_recall_curve.csv`.
Les valeurs verifiees et le SHA-256 du manifeste de test figurent dans
`pr_auc_locked_test.csv`.

## 2. Benchmark CPU detaille

**Machine et logiciels.** Intel Core i5-12450H (12e generation), 8 coeurs
physiques, 12 processeurs logiques, 7,73 Gio de RAM ; Windows build 26200 ;
Python 3.10.11 ; TensorFlow 2.12.0 ; Keras 2.12.0 ; OpenCV 4.11.0 ; NumPy
1.23.5 ; psutil 7.2.2. GPU non utilise. TensorFlow configure a 8 threads
intra-operation et 2 inter-operation ; OpenCV annonce 12 threads. Le batch est
de **1 clip de 20 images** par appel. Il s'agit de **chacun des quatre modeles
finaux**, pas seulement de LO.

**Protocole.** Dix clips fixes de la validation groupee scellee (5 violents,
5 non violents), cinq appels de chauffe par modele puis deux passages :
20 observations mesurees par modele. L'inference pure recoit les 20 images
uniformement echantillonnees ; la mesure bout en bout inclut l'ouverture,
le decodage/echelonnement de ces images et la prediction. Chargement du modele
et verification des SHA-256 exclus des latences. Les quatre modeles utilisent
les memes dix videos. Les mesures sont sur ce poste, et non un banc de temps
reel continu. Le test groupe n'a pas servi a ce benchmark.

| Modele | Inference moyenne / p95 (ms/clip) | Debit inference (clips/s) | Bout en bout moyen / p95 (ms/clip) | Debit bout en bout (clips/s) | FPS normalises bout en bout |
|---|---:|---:|---:|---:|---:|
| LO | 1 322 / 1 421 | 0,756 | 1 952 / 3 515 | 0,512 | 10,24 |
| LA | 1 290 / 1 342 | 0,775 | 1 921 / 3 499 | 0,520 | 10,41 |
| GO | 1 294 / 1 349 | 0,773 | 1 931 / 3 503 | 0,518 | 10,36 |
| GA | 1 324 / 1 414 | 0,755 | 1 972 / 3 514 | 0,507 | 10,14 |

Les FPS normalises valent 20 divise par la duree d'un clip ; ils **ne sont pas
un debit de camera continu**. En particulier, le scoreur verrouille travaille
sur des fenetres glissantes de pas 5, alors que ce benchmark utilise un
echantillonnage uniforme de 20 images pour un appel de modele. La mesure ne
demontre donc pas 10 fps reels de video continue.

| Modele | CPU systeme moyen / pic | CPU processus moyen / pic | RAM processus moyenne / pic (Mio) | Threads CPU actifs / fenetre de 0,5 s, moyenne / pic | Threads presents, moyenne / pic |
|---|---:|---:|---:|---:|---:|
| LO | 41,7 / 51,0 % | 261,9 / 329,3 % | 583 / 793 | 12,4 / 25 | 63,0 / 68 |
| LA | 37,1 / 45,2 % | 268,0 / 336,3 % | 646 / 774 | 12,2 / 24 | 59,7 / 66 |
| GO | 37,7 / 43,6 % | 270,2 / 319,2 % | 682 / 812 | 12,1 / 22 | 58,7 / 64 |
| GA | 38,3 / 44,2 % | 262,2 / 326,3 % | 733 / 839 | 11,9 / 22 | 60,8 / 64 |

Pour `psutil`, 100 % de CPU processus signifie environ un coeur logique
pleinement utilise ; cette mesure peut depasser 100 %. Un thread dit actif a
consomme du temps CPU pendant une fenetre de 0,5 s : le pic peut donc depasser
les 12 processeurs logiques sans signifier 25 executions simultanees.
La RAM est le *resident set size* du processus Python, incluant TensorFlow et
les images temporaires, pas seulement les poids. Les modeles sont charges
sequentiellement dans un meme processus : la RAM moyenne tend donc a augmenter
avec l'ordre d'execution. Le premier passage exploratoire, avec une frequence
d'echantillonnage differente, a donne des latences sensiblement differentes ;
les chiffres ci-dessus sont ceux de la passe finale a intervalle de 0,5 s.
Cette variabilite interdit d'attribuer de petits ecarts de vitesse a la seule
architecture. Le vieux benchmark `exp03_cpu_benchmark` ne portait que sur le
modele de reference anterieur, avec un autre echantillon et 300 observations ;
il ne constitue pas une comparaison appariee avec ces quatre lignes.

Fichiers : `cpu_benchmark_environment.json` (CPU, logiciels, threads, batch,
echantillon), `cpu_benchmark_raw.csv` et `cpu_benchmark_summary.csv`.

## 3. Delai de declenchement d'alerte

Les 297 videos violentes du test groupe, et non les dix anciennes videos, ont
ete relues dans le **cache de scores deja produit**. Pour chaque modele,
on cherche le premier indice ou la moyenne glissante complete de taille K
atteint le theta verrouille. Avec une sequence de 20 frames et un pas de 5,
le numero de frame de declenchement (base 1) est `20 + 5 * indice_score`.
Comme precedemment, la duree est estimee depuis le debut du clip a 25 fps ;
le debut exact de l'acte violent n'est pas annote. Ce n'est **pas** la latence
murale d'une alerte en production et ce n'est pas directement comparable aux
0,940 s de l'ancien protocole (autres modeles, K/theta, clips et pas temporel).
Le maximum des moyennes glissantes et la decision ont ete verifies contre la
prediction archivee pour chacune des 297 videos et des quatre modeles.

| Modele | theta/K | Detectees / 297 | Delai moyen des detectees | Mediane | P95 |
|---|---:|---:|---:|---:|---:|
| LO | 0,70 / 5 | 274/297 | 1,847 s | 1,600 s | 3,000 s |
| LA | 0,70 / 7 | 274/297 | 2,196 s | 2,000 s | 3,000 s |
| GO | 0,65 / 5 | 269/297 | 1,917 s | 1,600 s | 3,400 s |
| GA | 0,65 / 10 | 255/297 | 2,875 s | 2,600 s | 4,000 s |

Les clips non detectes n'ont pas de delai fini et sont exclus des moyennes,
mais restent dans le denominateur de detection. Fichiers :
`alert_delay_by_video.csv` et `alert_delay_summary.json`.

## 4. Erreurs qualitatives

Les 12 videos ci-dessous ont ete selectionnees intentionnellement parmi les
erreurs du test verrouille et inspectees a 10 %, 50 % et 90 % de leur duree.
Le tableau donne un modele representatif par cas ; `qualitative_error_examples.csv`
contient **39 lignes modele-erreur** avec, pour chaque modele concerne, le
score exact `p(violence)`, la probabilite de la classe predite, le dataset et
le chemin source. Ces scores ne sont pas calibres. Pour un faux negatif,
la probabilite de la classe predite est `1 - p(violence)` et peut etre
inferieure a 0,5, car theta vaut 0,65 ou 0,70. Les mecanismes sont des
**hypotheses visuelles**, pas une attribution causale prouvee.

| Erreur | Exemple (dataset source) | Modele representatif | p(violence) | Confiance classe predite | Type de scene | Mecanisme probable |
|---|---|---|---:|---:|---|---|
| FP | `rwf_train_nonfight_0542.avi` (RWF-2000) | LO | 0,976 | 0,976 | CCTV interieur, personnes circulant | Proximite et contexte de surveillance surinterpretes |
| FP | `rwf_val_nonfight_0034.avi` (RWF-2000) | LA | 0,973 | 0,973 | Commerce, homme gesticulant | Gestes amples ressemblant a une lutte |
| FP | `rwf_train_nonfight_0057.avi` (RWF-2000) | GO | 0,887 | 0,887 | Escalier CCTV | Deplacement rapide, silhouettes masquees |
| FP | `NV_40.mp4` (RLVS) | GA | 0,755 | 0,755 | Saut a la perche | Mouvement explosif et chute sportive |
| FP | `NV_14.mp4` (RLVS) | GO | 0,957 | 0,957 | Tribune sportive | Foule agitee et bras leves |
| FP | `rwf_train_nonfight_0449.avi` (RWF-2000) | LA | 0,937 | 0,937 | Rue nocturne CCTV | Faible eclairage, interaction ambigue |
| FN | `rwf_train_fight_0444.avi` (RWF-2000) | LO | 0,297 | 0,703 | Rue nocturne noir/blanc | Plan large, sujets petits, faible contraste |
| FN | `rwf_train_fight_0376.avi` (RWF-2000) | GO | 0,231 | 0,769 | Rue nocturne avec bandeau TV | Sujets petits et bandeau intrusif |
| FN | `rwf_val_fight_0006.avi` (RWF-2000) | LA | 0,631 | 0,369 | Commerce vu d'en haut | Comptoir masquant le contact |
| FN | `V_233.mp4` (RLVS) | LO | 0,555 | 0,445 | Altercation en video verticale | Image utile et protagonistes petits |
| FN | `V_671.mp4` (RLVS) | LA | 0,603 | 0,397 | Altercation sur terrain de sport | Contexte sportif et groupe ambigus |
| FN | `V_634.mp4` (RLVS) | GA | 0,588 | 0,412 | Galerie commerciale | Plan large et mouvement rapide |

Taxonomie simple : (1) faux positifs par sport/foule ou gestes non violents
energiques ; (2) faux positifs sur scenes de surveillance ambigues, parfois
mal eclairees ; (3) faux negatifs par plan large, occlusion ou faible contraste ;
(4) faux negatifs par cadrage vertical ou contexte sportif brouillant les
indices. L'echantillon visuel est raisonne et ne permet pas d'estimer la
frequence de ces mecanismes sur toutes les erreurs.

Fichiers : `all_locked_test_errors.csv` (toutes les erreurs),
`qualitative_error_examples.csv` (cas revus),
`inspection/error_inspection_1.jpg` a `error_inspection_3.jpg` (preuves visuelles).

## Integrite et limites

Le manifeste scelle du test contient 597 **entrees** pour 596 contenus
SHA-256 uniques : une video non violente RWF-2000 apparait sous deux chemins
(`train/NonFight/rwf_train_nonfight_0157.avi` et
`val/NonFight/rwf_val_nonfight_0037.avi`), tous deux dans le **meme split test**
et le meme groupe de scene. Ce n'est pas une fuite inter-splits, mais les
metriques/AP historiques ponderent deux fois ce contenu. Aucune metrique du
test verrouille n'a ete corrigee retrospectivement ici.

Les chemins `train/` ou `val/` dans les noms RWF-2000 sont les dossiers du
dataset source ; ils ne designent pas le split groupe SIRCH actuel. Toutes
les videos citees dans l'analyse d'erreurs sont dans le nouveau **test**.

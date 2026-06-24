# Rapport global du projet SIRCH

Date de synthese : 2026-06-23
Projet : SIRCH - Systeme Intelligent de Reconnaissance de Comportements violents en temps reel
Dossier projet : `C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH`
Environnement local : `C:\SIRCH_ENV`

Ce rapport structure tout ce qui a ete fait dans le projet SIRCH depuis l'installation jusqu'aux experiences scientifiques de phase 3. Il resume les manipulations effectuees, les fichiers produits, les resultats obtenus, les limites observees et l'etat final du projet.

Important : les identifiants sensibles de `config.py` ne sont pas recopies ici. Gmail est decrit comme configure localement, mais le mot de passe d'application n'est pas expose dans ce rapport.

---

## 1. Objectif general du projet

SIRCH vise a detecter automatiquement des comportements violents dans un flux video ou webcam, en combinant :

- acquisition video,
- pretraitement image,
- modele d'apprentissage profond,
- decision temporelle par fenetre glissante,
- alertes,
- enregistrement d'incidents,
- tableau de bord Flask.

Le systeme final fonctionne autour d'un modele EfficientNetB0 + module recurrent, avec une decision par seuil `theta` et fenetre temporelle `K`.

Parametres operationnels principaux :

| Parametre | Valeur |
|---|---:|
| Nombre de frames par sequence `N_FRAMES` | 20 |
| Taille image | 224 x 224 |
| Seuil par defaut `theta` | 0.50 |
| Fenetre de decision `K` | 5 |
| Cooldown alerte | 30 s |
| Modele principal | `C:\SIRCH_ENV\models\sirch_model.h5` |
| Dashboard | `http://127.0.0.1:5000` |

---

## 2. Architecture logicielle finale

Le systeme a ete structure en 7 modules principaux.

| Module | Fichier | Role |
|---|---|---|
| 1. Acquisition | `core/acquisition.py` | Lecture webcam, fichier video ou flux |
| 2. Pretraitement | `core/preprocessing.py` | Redimensionnement et preparation des sequences |
| 3. Inference IA | `core/inference.py` | Chargement du modele et prediction violence/non-violence |
| 4. Decision | `core/decision.py` | Moyenne glissante sur `K` scores |
| 5. Alertes | `alerts/alertmanager.py` | Son, Gmail, Telegram si configure |
| 6. Base de donnees | `database/db.py` | SQLite, incidents et parametres |
| 7. Dashboard | `dashboard/app.py` | Interface Flask, historique, export CSV, captures |

Le fichier `main.py` orchestre ces modules de bout en bout :

- demarrage de la base SQLite,
- chargement du modele,
- lecture de la video,
- construction des sequences de 20 frames,
- prediction,
- decision avec `theta` et `K`,
- sauvegarde de capture en cas d'incident,
- enregistrement SQLite,
- envoi d'alerte.

Point important : le systeme sauvegarde actuellement des captures image d'incident, pas des clips video complets. Ces captures sont visibles dans le dashboard via la colonne `Capture`.

---

## 3. Installation et configuration realisees

### 3.1 Environnement local

Un environnement local a ete utilise dans :

`C:\SIRCH_ENV`

L'objectif etait de ne plus dependre de Google Drive pour les datasets, car Drive etait plein et Colab rencontrait des problemes d'espace.

Actions realisees :

- installation/usage de Python local,
- preparation des dossiers `datasets`, `models`, `checkpoints`,
- installation des dependances necessaires a l'entrainement et a l'evaluation,
- installation de `yt-dlp` pour telecharger des videos Creative Commons,
- usage local des datasets au lieu de Google Drive.

### 3.2 Gmail et Telegram

Gmail a ete configure localement pour les alertes email.

Telegram n'etait pas disponible immediatement, car le token du bot dependait d'une creation exterieure. Le code a donc ete adapte pour ne pas planter si :

- `TELEGRAM_TOKEN` est vide,
- `TELEGRAM_TOKEN = "METS_TON_TOKEN_ICI"`,
- `TELEGRAM_CHAT_ID` est vide,
- `TELEGRAM_CHAT_ID = "METS_TON_CHAT_ID_ICI"`.

Dans ce cas, l'alerte Telegram est ignoree silencieusement.

---

## 4. Donnees utilisees

### 4.1 Datasets principaux

Deux datasets principaux ont ete utilises :

- RLVS : Real Life Violence Situations Dataset
- RWF-2000

Les donnees ont ete placees localement sous :

`C:\SIRCH_ENV\datasets`

Le telechargement via Google Drive a ete abandonne pour les datasets lourds. RWF-2000 a ete gere localement, avec extraction/reparation a cause de problemes de Drive, de quota et de noms de fichiers trop longs. RLVS a ete telecharge et utilise localement.

### 4.2 Description globale des datasets

Source : `phase3_results/dataset_description.csv`

| Dataset | Total | Violence | Non-violence | Duree moyenne |
|---|---:|---:|---:|---:|
| RLVS | 2000 | 1000 | 1000 | 5.253 s |
| RWF-2000 | 1991 | 991 | 1000 | 5.000 s |
| Total | 3991 | 1991 | 2000 | 5.127 s |

### 4.3 Repartition finale train/validation/test

| Split | Total | Violence | Non-violence |
|---|---:|---:|---:|
| Train | 2793 | 1393 | 1400 |
| Validation | 599 | 299 | 300 |
| Test | 599 | 299 | 300 |

La repartition a ete faite avec `seed=42`, shuffle et stratification 70/15/15.

### 4.4 Videos supplementaires sport/danse/calme

Pour mesurer les faux positifs en conditions non violentes, 30 videos Creative Commons ont ete telechargees avec `yt-dlp` :

- 15 videos sport,
- 10 videos danse,
- 5 videos calmes.

Une premiere serie a ete utilisee, puis une deuxieme serie propre `v2` a ete creee apres suspicion de contamination par entrainement augmente. Les dossiers propres sont :

- `datasets/test_sport_v2/`
- `datasets/test_danse_v2/`
- `datasets/test_calme_v2/`

Toutes ces videos sont considerees non violentes. Toute prediction "violence" est donc un faux positif.

---

## 5. Modele original SIRCH

### 5.1 Architecture

Le modele original utilise :

- EfficientNetB0 pre-entraine ImageNet,
- `TimeDistributed(EfficientNetB0)` sur 20 frames,
- LSTM 256 unites,
- Dropout,
- Dense 128 ReLU,
- Dropout,
- Dense 1 sigmoid.

Fichier principal :

`C:\SIRCH_ENV\models\sirch_model.h5`

### 5.2 Entrainement original

L'entrainement local a ete interrompu plusieurs fois par des coupures de courant. Des checkpoints ont ete mis en place pour reprendre sans recommencer a zero.

Garde-fous ajoutes/verifies :

- `ModelCheckpoint` apres chaque epoch,
- checkpoints `.weights.h5`,
- reprise depuis le dernier checkpoint,
- `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`,
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`,
- `CSVLogger`.

Resultat de l'entrainement original :

- entrainement termine a l'epoch 15,
- meilleur modele conserve : epoch utilisateur 4,
- fichier final : `C:\SIRCH_ENV\models\sirch_model.h5`,
- meilleur `val_loss` dans le log : epoch CSV 3, soit epoch utilisateur 4.

Meilleure ligne de validation du modele original :

| Epoch utilisateur | accuracy | loss | val_accuracy | val_loss | val_precision | val_recall |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9227 | 0.1890 | 0.8664 | 0.3197 | 0.8411 | 0.9030 |

Le surentrainement a commence a etre visible ensuite : la loss d'entrainement continuait a baisser pendant que `val_loss` augmentait.

---

## 6. Modele augmente

Un modele augmente a ete entraine dans :

`C:\SIRCH_ENV\models\sirch_model_augmente.h5`

Notebook :

`colab/train_sirch_augmente.ipynb`

Garde-fous confirmes :

- `ModelCheckpoint`,
- `EarlyStopping` avec `restore_best_weights=True`,
- `ReduceLROnPlateau`,
- reprise possible via checkpoint,
- log CSV.

Le meilleur `val_loss` observe dans le log augmente correspond a l'epoch CSV 2, soit epoch utilisateur 3.

| Epoch utilisateur | accuracy | loss | val_accuracy | val_loss | val_precision | val_recall |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.9082 | 0.2278 | 0.8654 | 0.3173 | 0.8423 | 0.8960 |

Ce modele a obtenu de meilleurs resultats sur le test principal de 599 videos, mais les tests de faux positifs v2 ont montre que cette amelioration n'est pas uniforme sur les videos hors distribution.

---

## 7. Evaluation principale sur 599 videos

Jeu de test :

- 599 videos,
- 299 videos violentes,
- 300 videos non violentes.

### 7.1 Resultats du modele original

Source : `phase2_results/metriques_avancees.csv`

| Seuil | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.8681 | 0.8274 | 0.9298 | 0.8756 | 242 | 58 | 21 | 278 |
| 0.65 | 0.8731 | 0.8704 | 0.8763 | 0.8733 | 261 | 39 | 37 | 262 |
| 0.70 | 0.8664 | 0.8712 | 0.8595 | 0.8653 | 262 | 38 | 42 | 257 |

Courbes et aires :

| Metrique | Valeur |
|---|---:|
| ROC-AUC | 0.9446 |
| Average Precision | 0.9415 |
| PR-AUC trapezoidale | 0.9414 |
| Temps inference mesure phase 2 | 36.87 ms/frame |

### 7.2 Resultats du modele augmente

| Seuil | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.9199 | 0.8862 | 0.9632 | 0.9231 | 263 | 37 | 11 | 288 |
| 0.65 | 0.9165 | 0.9164 | 0.9164 | 0.9164 | 275 | 25 | 25 | 274 |
| 0.70 | 0.9182 | 0.9310 | 0.9030 | 0.9168 | 280 | 20 | 29 | 270 |

Courbes et aires :

| Metrique | Valeur |
|---|---:|
| ROC-AUC | 0.9767 |
| Average Precision | 0.9738 |
| PR-AUC trapezoidale | 0.9737 |
| Temps inference mesure phase 2 | 38.33 ms/frame |

### 7.3 Conclusion principale

Sur le jeu de test principal de 599 videos, le modele augmente est meilleur que le modele original :

| Metrique | Original | Augmente | Gain |
|---|---:|---:|---:|
| Accuracy | 0.8681 | 0.9199 | +0.0518 |
| F1-score | 0.8756 | 0.9231 | +0.0475 |
| ROC-AUC | 0.9446 | 0.9767 | +0.0321 |
| Faux positifs | 58 | 37 | -21 |
| Faux negatifs | 21 | 11 | -10 |

---

## 8. Faux positifs sport/danse/calme

### 8.1 Premiere serie de 30 videos non violentes

Source : `phase2_results/resultats_comparaison_modeles.csv`

Modele original :

| Seuil | Faux positifs / 30 | Taux |
|---:|---:|---:|
| 0.50 | 7 | 23.33% |
| 0.65 | 5 | 16.67% |
| 0.70 | 3 | 10.00% |

Modele augmente :

| Seuil | Faux positifs / 30 | Taux |
|---:|---:|---:|
| 0.50 | 3 | 10.00% |
| 0.65 | 1 | 3.33% |
| 0.70 | 1 | 3.33% |

### 8.2 Deuxieme serie propre v2 de 30 videos non violentes

Source : `phase2_results/resultats_faux_positifs_v2.csv`

Modele original :

| Seuil | Sport | Danse | Calme | Global |
|---:|---:|---:|---:|---:|
| 0.50 | 3/15 = 20.00% | 8/10 = 80.00% | 3/5 = 60.00% | 14/30 = 46.67% |
| 0.65 | 2/15 = 13.33% | 7/10 = 70.00% | 3/5 = 60.00% | 12/30 = 40.00% |
| 0.70 | 2/15 = 13.33% | 5/10 = 50.00% | 3/5 = 60.00% | 10/30 = 33.33% |

Modele augmente :

| Seuil | Sport | Danse | Calme | Global |
|---:|---:|---:|---:|---:|
| 0.50 | 4/15 = 26.67% | 7/10 = 70.00% | 4/5 = 80.00% | 15/30 = 50.00% |
| 0.65 | 3/15 = 20.00% | 6/10 = 60.00% | 1/5 = 20.00% | 10/30 = 33.33% |
| 0.70 | 2/15 = 13.33% | 5/10 = 50.00% | 1/5 = 20.00% | 8/30 = 26.67% |

### 8.3 Interpretation

Les videos danse et certaines scenes calmes restent une source importante de faux positifs. Les mouvements rapides, mouvements corporels amples, changements de camera et foules peuvent etre interpretes comme violence.

Le seuil 0.70 reduit fortement les faux positifs, mais augmente le risque de faux negatifs sur les vraies videos violentes.

---

## 9. Analyse des biais RLVS / RWF-2000

Source : `phase2_results/biais_datasets.csv`

### 9.1 Modele original a seuil 0.50

| Dataset | Accuracy | Precision | Recall | F1 | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|
| RLVS | 0.9322 | 0.8931 | 0.9793 | 0.9342 | 0.1133 | 0.0207 |
| RWF-2000 | 0.8059 | 0.7684 | 0.8831 | 0.8218 | 0.2733 | 0.1169 |
| Global | 0.8681 | 0.8274 | 0.9298 | 0.8756 | 0.1933 | 0.0702 |

### 9.2 Modele augmente a seuil 0.50

| Dataset | Accuracy | Precision | Recall | F1 | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|---:|
| RLVS | 0.9593 | 0.9346 | 0.9862 | 0.9597 | 0.0667 | 0.0138 |
| RWF-2000 | 0.8816 | 0.8430 | 0.9416 | 0.8896 | 0.1800 | 0.0584 |
| Global | 0.9199 | 0.8862 | 0.9632 | 0.9231 | 0.1233 | 0.0368 |

### 9.3 Conclusion biais

Le modele performe mieux sur RLVS que sur RWF-2000. RWF-2000 est plus difficile, avec plus de faux positifs et faux negatifs. L'augmentation ameliore fortement RWF-2000, mais ne supprime pas completement le decalage entre datasets.

---

## 10. Delai d'alerte

Source : `phase2_results/delai_alerte.csv`

Evaluation sur 10 videos violentes detectees.

| K | Delai moyen a 25 fps | Delai moyen source fps | Videos detectees |
|---:|---:|---:|---:|
| 3 | 0.864 s | 0.720 s | 10/10 |
| 5 | 0.940 s | 0.783 s | 10/10 |
| 7 | 1.016 s | 0.847 s | 10/10 |

Interpretation :

- Plus `K` est grand, plus la decision est stable.
- Plus `K` est grand, plus l'alerte est retardee.
- `K=5` est un compromis raisonnable entre lissage et rapidite.

---

## 11. Optimisation du seuil theta

Source : `phase3_results/seuil_optimisation.csv`

Le seuil a ete teste de 0.30 a 0.90 par pas de 0.05 sur le modele original.

Resultats importants :

| Objectif | Seuil | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 maximal | 0.40 | 0.8781 | 0.8229 | 0.9632 | 0.8875 | 62 | 11 |
| Minimiser les FN | 0.35 | 0.8664 | 0.8050 | 0.9666 | 0.8784 | 70 | 10 |
| Meilleur equilibre precision/rappel | 0.65 | 0.8731 | 0.8704 | 0.8763 | 0.8733 | 39 | 37 |

Conclusion :

- `theta=0.40` maximise le F1.
- `theta=0.35` minimise les violences manquees.
- `theta=0.65` equilibre precision et rappel.
- `theta=0.70` est utile si la priorite est de reduire les fausses alertes.

---

## 12. Analyse statistique

Source : `phase3_results/statistical_analysis.csv`

### 12.1 Intervalles de confiance bootstrap 95%

| Modele | Metrique | Valeur | IC 95% bas | IC 95% haut |
|---|---|---:|---:|---:|
| Original | Accuracy | 0.8681 | 0.8431 | 0.8932 |
| Original | F1 | 0.8756 | 0.8503 | 0.9008 |
| Original | ROC-AUC | 0.9446 | 0.9291 | 0.9599 |
| Augmente | Accuracy | 0.9199 | 0.8982 | 0.9416 |
| Augmente | F1 | 0.9231 | 0.9016 | 0.9444 |
| Augmente | ROC-AUC | 0.9767 | 0.9668 | 0.9861 |

### 12.2 Gains bootstrap augmente - original

| Metrique | Gain | IC 95% bas | IC 95% haut |
|---|---:|---:|---:|
| Accuracy | +0.0518 | +0.0284 | +0.0785 |
| F1 | +0.0475 | +0.0251 | +0.0723 |
| ROC-AUC | +0.0321 | +0.0208 | +0.0445 |

### 12.3 Test de McNemar

| Element | Valeur |
|---|---:|
| Les deux corrects | 504 |
| Original correct, augmente faux | 16 |
| Original faux, augmente correct | 47 |
| Les deux faux | 32 |
| Paires discordantes | 63 |
| p-value exacte binomiale | 0.00011706 |

Conclusion : la difference entre le modele original et le modele augmente est statistiquement significative au seuil `p < 0.05`.

---

## 13. Complexite computationnelle

Source : `phase3_results/model_complexity.txt`

Modele analyse : `C:\SIRCH_ENV\models\sirch_model.h5`

| Element | Valeur |
|---|---:|
| Taille disque | 21.85 MB |
| Parametres totaux | 5 656 484 |
| Parametres entrainables | 1 606 913 |
| Parametres non entrainables | 4 049 571 |
| RAM avant chargement modele | 303.00 MB |
| RAM apres chargement modele | 387.81 MB |
| Delta chargement | 84.80 MB |
| RAM apres inference | 602.68 MB |
| Chargement modele | 5.89 s |

Inference CPU sur 100 frames :

| Protocole | Temps moyen/frame | Debit theorique |
|---|---:|---:|
| Inference pure | 28.87 ms/frame | 34.64 fps |
| End-to-end lecture + preprocessing + prediction | 47.33 ms/frame | 21.13 fps |

Conclusion embarquee :

- Raspberry Pi 5 : possible en demonstration, mais temps reel difficile sans TensorFlow Lite, quantification et reduction du pipeline.
- Jetson Nano : plus plausible avec TensorRT/FP16, batch 1 et pipeline leger.

---

## 14. Phase 3 - Experiences complementaires Bagula

### 14.1 P3-A - Description des datasets

Fichiers produits :

- `phase3_results/dataset_description.csv`
- `phase3_results/dataset_description_videos.csv`
- `phase3_results/figures/dataset_description.png`

Resultat : tableau complet des volumes, splits, labels et durees moyennes pour RLVS et RWF-2000.

### 14.2 P3-B - Complexite computationnelle

Fichier produit :

- `phase3_results/model_complexity.txt`

Resultat : mesure des parametres, taille disque, RAM, temps CPU et faisabilite embarquee.

### 14.3 P3-C - Matrices de confusion separees

Fichiers produits :

- `phase3_results/confusion_matrices.csv`
- `phase3_results/figures/confusion_matrix_rlvs.png`
- `phase3_results/figures/confusion_matrix_rwf_2000.png`
- `phase3_results/figures/confusion_matrix_global.png`

Resultats globaux a seuil 0.50 :

| Modele | Dataset | TN | FP | FN | TP | Accuracy | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | RLVS | 133 | 17 | 3 | 142 | 0.9322 | 0.9342 |
| Original | RWF-2000 | 109 | 41 | 18 | 136 | 0.8059 | 0.8218 |
| Original | Global | 242 | 58 | 21 | 278 | 0.8681 | 0.8756 |
| Augmente | RLVS | 140 | 10 | 2 | 143 | 0.9593 | 0.9597 |
| Augmente | RWF-2000 | 123 | 27 | 9 | 145 | 0.8816 | 0.8896 |
| Augmente | Global | 263 | 37 | 11 | 288 | 0.9199 | 0.9231 |

### 14.4 P3-D - Optimisation du seuil

Fichiers produits :

- `phase3_results/seuil_optimisation.csv`
- `phase3_results/figures/seuil_optimisation_f1.png`

Resultat : meilleur F1 a `theta=0.40`, meilleur equilibre precision/rappel a `theta=0.65`.

### 14.5 P3-E - Analyse statistique

Fichiers produits :

- `phase3_results/statistical_analysis.csv`
- `phase3_results/figures/statistical_analysis_bootstrap.png`

Resultat : avantage du modele augmente significatif (`p=0.00011706`).

### 14.6 P3-F - Captures d'erreurs

Fichiers produits :

- `phase3_results/erreurs_analyse.csv`
- `phase3_results/figures/erreurs/`

20 captures ont ete sauvegardees :

- 5 faux positifs, frame centrale et frame extreme,
- 5 faux negatifs, frame centrale et frame extreme.

Objectif : analyser qualitativement les erreurs du modele.

### 14.7 P3-G - Generalisation cross-dataset

Deux niveaux ont ete traites :

1. Analyse approximative a partir des scores existants du modele combine.
2. Reentrainements reels separes :
   - RLVS-only, test sur RWF-2000,
   - RWF-only, test sur RLVS.

#### Analyse approximative sur modele combine

Source : `phase3_results/cross_dataset_generalization.csv`

| Modele | Test domain | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Original | RWF-2000 | 0.8059 | 0.7684 | 0.8831 | 0.8218 | 0.8902 |
| Original | RLVS | 0.9322 | 0.8931 | 0.9793 | 0.9342 | 0.9811 |
| Augmente | RWF-2000 | 0.8816 | 0.8430 | 0.9416 | 0.8896 | 0.9548 |
| Augmente | RLVS | 0.9593 | 0.9346 | 0.9862 | 0.9597 | 0.9909 |

Limite : cette analyse est approximative car les modeles original et augmente ont ete entraines sur les deux datasets combines.

#### Reentrainement RLVS-only puis test RWF-2000

Source : `phase3_results/evaluation_rlvs_only_on_rwf.csv`

| Train | Test | N | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RLVS | RWF-2000 | 1991 | 0.5480 | 0.5296 | 0.8204 | 0.6437 | 0.6395 | 278 | 722 | 178 | 813 |

Interpretation : un modele entraine uniquement sur RLVS generalise mal vers RWF-2000, surtout a cause d'un grand nombre de faux positifs.

#### Reentrainement RWF-only puis test RLVS

Source : `phase3_results/evaluation_rwf_only_on_rlvs.csv`

| Train | Test | N | Accuracy | Precision | Recall | F1 | ROC-AUC | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RWF-2000 | RLVS | 2000 | 0.7450 | 0.8010 | 0.6520 | 0.7189 | 0.8217 | 838 | 162 | 348 | 652 |

Interpretation : un modele entraine uniquement sur RWF-2000 generalise mieux vers RLVS que l'inverse, mais manque beaucoup de violences.

### 14.8 P3-H - Ablation complete avec reentrainement

Fichiers produits :

- `colab/train_efficientnet_only.ipynb`
- `colab/train_gru.ipynb`
- `phase3_results/evaluation_efficientnet_only.csv`
- `phase3_results/evaluation_gru.csv`
- `phase3_results/ablation_complete.csv`
- `phase3_results/figures/ablation_complete.png`

Comparaison finale :

| Configuration | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet seul | 0.8531 | 0.8528 | 0.8528 | 0.8528 | 256 | 44 | 44 | 255 | 42.79 |
| EfficientNet + LSTM original | 0.8681 | 0.8274 | 0.9298 | 0.8756 | 242 | 58 | 21 | 278 | 36.87 |
| EfficientNet + GRU | 0.8998 | 0.8794 | 0.9264 | 0.9023 | 262 | 38 | 22 | 277 | 36.46 |

Conclusion :

- EfficientNet seul est le moins performant.
- Le LSTM apporte un gain de rappel important par rapport a EfficientNet seul.
- Le GRU obtient le meilleur compromis global parmi les trois architectures testees en phase 3.
- Le modele augmente reste toutefois le meilleur resultat global connu sur le test 599 videos avec F1 = 0.9231.

### 14.9 P3-I - Push GitHub

Commit pousse :

`21cc27e SIRCH - phase 3 - experiences complementaires Bagula`

Remote :

`https://github.com/Ertual/SIRCH-.git`

Fichiers inclus :

- notebooks phase 3,
- scripts `tools/`,
- resultats `phase3_results/`,
- figures,
- tableaux CSV.

Fichiers exclus :

- modeles `.h5`,
- checkpoints `.weights.h5`,
- datasets,
- base SQLite locale `database/sirch.db`.

---

## 15. Systeme complet et dashboard

Le dashboard Flask a ete lance avec succes via :

`C:\SIRCH_ENV\Scripts\python.exe -m dashboard.app`

Adresse :

`http://127.0.0.1:5000`

Fonctionnalites disponibles :

- affichage du seuil,
- affichage de `K`,
- modification des parametres,
- historique des incidents,
- export CSV,
- acces aux captures image.

Commande de lancement SIRCH webcam :

```powershell
cd "C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH"
C:\SIRCH_ENV\Scripts\python.exe main.py --source 0
```

Commande de lancement dashboard :

```powershell
cd "C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH"
C:\SIRCH_ENV\Scripts\python.exe -m dashboard.app
```

---

## 16. Manipulations importantes realisees

### 16.1 Datasets et stockage

- Montage Google Drive dans Colab au debut.
- Probleme de quota Google Drive.
- Interruption du telechargement RWF-2000.
- Passage a une strategie locale sur PC Windows.
- Liberation d'espace disque local.
- Telechargement local des datasets dans `C:\SIRCH_ENV\datasets`.
- Reparation/extraction de RWF-2000 apres erreurs 7-Zip et noms de fichiers trop longs.
- Verification des fichiers et structures de dossiers.

### 16.2 Entrainement

- Creation et adaptation de `train_sirch.ipynb`.
- Ajout des checkpoints.
- Ajout de la reprise automatique.
- Correction du format `.weights.h5`.
- Surveillance de l'entrainement pendant les epochs.
- Reprise apres coupures de courant.
- Sauvegarde du meilleur modele original.
- Creation de `train_sirch_augmente.ipynb`.
- Entrainement du modele augmente.
- Creation de notebooks phase 3 :
  - `train_rlvs_only.ipynb`,
  - `train_rwf_only.ipynb`,
  - `train_efficientnet_only.ipynb`,
  - `train_gru.ipynb`.

### 16.3 Evaluations

- Evaluation du modele original sur 599 videos.
- Evaluation CPU/GPU et temps par frame.
- Evaluation du modele augmente sur 599 videos.
- Tests de faux positifs sur videos sport/danse/calme.
- Telechargement de videos Creative Commons avec verification de licences.
- Creation d'une deuxieme serie propre v2 de videos non violentes.
- Courbes ROC et Precision-Recall.
- Analyse de biais par dataset.
- Analyse du delai d'alerte.
- Optimisation de seuil.
- Test statistique McNemar.
- Bootstrap 95%.
- Matrices de confusion separees.
- Captures d'erreurs.
- Cross-dataset avec reentrainements reels.
- Ablation EfficientNet seul / LSTM / GRU.

### 16.4 Application

- Creation/verification des 7 modules.
- Configuration Gmail.
- Neutralisation propre de Telegram si token absent.
- Lancement du dashboard Flask.
- Verification que le dashboard est accessible sur localhost.
- Verification que les incidents peuvent etre enregistres dans SQLite.

### 16.5 GitHub

Commits importants :

| Commit | Message |
|---|---|
| `dbb71cb` | `SIRCH - sauvegarde complete du projet - phase 2` |
| `feedd56` | `SIRCH - resultats phase 2 et validation finale` |
| `21cc27e` | `SIRCH - phase 3 - experiences complementaires Bagula` |

Etat actuel connu :

- `origin/main` pointe sur `21cc27e`.
- `database/sirch.db` reste modifie localement parce que SQLite reecrit ce fichier pendant l'usage du dashboard.
- Ce fichier n'a pas ete pousse car il s'agit d'une base runtime locale, pas d'un fichier source scientifique.

---

## 17. Fichiers de resultats principaux

### Phase 2

- `phase2_results/metriques_avancees.csv`
- `phase2_results/metriques_avancees_scores_599.csv`
- `phase2_results/resultats_comparaison_modeles.csv`
- `phase2_results/resultats_faux_positifs_v2.csv`
- `phase2_results/biais_datasets.csv`
- `phase2_results/delai_alerte.csv`
- `phase2_results/ablation_results.csv`
- `phase2_results/figures/`

### Phase 3

- `phase3_results/dataset_description.csv`
- `phase3_results/model_complexity.txt`
- `phase3_results/confusion_matrices.csv`
- `phase3_results/seuil_optimisation.csv`
- `phase3_results/statistical_analysis.csv`
- `phase3_results/cross_dataset_generalization.csv`
- `phase3_results/evaluation_rlvs_only_on_rwf.csv`
- `phase3_results/evaluation_rwf_only_on_rlvs.csv`
- `phase3_results/evaluation_efficientnet_only.csv`
- `phase3_results/evaluation_gru.csv`
- `phase3_results/ablation_complete.csv`
- `phase3_results/figures/`

### Modeles locaux non pousses sur GitHub

- `C:\SIRCH_ENV\models\sirch_model.h5`
- `C:\SIRCH_ENV\models\sirch_model_augmente.h5`
- `C:\SIRCH_ENV\models\phase3\sirch_model_rlvs_only.weights.h5`
- `C:\SIRCH_ENV\models\phase3\sirch_model_rwf_only.weights.h5`
- `C:\SIRCH_ENV\models\phase3\sirch_model_efficientnet_only.h5`
- `C:\SIRCH_ENV\models\phase3\sirch_model_gru.h5`

---

## 18. Conclusions scientifiques

1. Le modele original SIRCH fonctionne correctement et depasse une base solide avec F1 = 0.8756 sur 599 videos.

2. Le modele augmente ameliore nettement le test principal :
   - F1 = 0.9231,
   - ROC-AUC = 0.9767,
   - baisse des faux positifs et faux negatifs.

3. L'amelioration du modele augmente est statistiquement significative selon McNemar (`p=0.00011706`) et les intervalles bootstrap.

4. RWF-2000 reste plus difficile que RLVS. Le modele montre un biais de domaine important.

5. Les tests sur videos non violentes externes montrent une faiblesse importante : danse, sport et scenes calmes peuvent generer des faux positifs.

6. Le seuil doit etre choisi selon l'objectif :
   - `theta=0.40` pour maximiser F1,
   - `theta=0.35` pour minimiser les violences manquees,
   - `theta=0.65` pour equilibrer precision/rappel,
   - `theta=0.70` pour reduire les fausses alertes.

7. L'ablation montre que la composante recurrente est utile. Le GRU fait mieux que le LSTM original dans l'experience P3-H, mais le modele augmente reste le meilleur modele global observe.

8. Le systeme logiciel complet est operationnel en local : webcam, prediction, decision, alerte, SQLite, dashboard.

---

## 19. Galerie complete des figures et captures

Cette section integre directement toutes les images de resultats trouvees dans le projet au moment de la synthese. Elle permet de lire le rapport avec les graphiques, courbes, matrices de confusion et captures d'erreurs sans devoir parcourir les dossiers manuellement.

### 19.1 Figures phase 2

#### Biais datasets - comparaison FP/FN

![Biais datasets - comparaison FP/FN](phase2_results/figures/biais_datasets_fp_fn_comparaison.png)

#### Biais datasets - taux de faux positifs

![Biais datasets - taux de faux positifs](phase2_results/figures/biais_datasets_taux_faux_positifs.png)

#### Delai d'alerte selon K

![Delai d'alerte selon K](phase2_results/figures/delai_alerte_k_comparaison.png)

#### Courbe Precision-Recall - modele augmente

![Courbe Precision-Recall - modele augmente](phase2_results/figures/precision_recall_augmente.png)

#### Courbe Precision-Recall - modele original

![Courbe Precision-Recall - modele original](phase2_results/figures/precision_recall_original.png)

#### Courbes Precision-Recall - original vs augmente

![Courbes Precision-Recall - original vs augmente](phase2_results/figures/precision_recall_original_vs_augmente.png)

#### Courbe ROC - modele augmente

![Courbe ROC - modele augmente](phase2_results/figures/roc_augmente.png)

#### Courbe ROC - modele original

![Courbe ROC - modele original](phase2_results/figures/roc_original.png)

#### Courbes ROC - original vs augmente

![Courbes ROC - original vs augmente](phase2_results/figures/roc_original_vs_augmente.png)

### 19.2 Figures phase 3

#### Ablation complete

![Ablation complete](phase3_results/figures/ablation_complete.png)

#### Matrice de confusion globale

![Matrice de confusion globale](phase3_results/figures/confusion_matrix_global.png)

#### Matrice de confusion RLVS

![Matrice de confusion RLVS](phase3_results/figures/confusion_matrix_rlvs.png)

#### Matrice de confusion RWF-2000

![Matrice de confusion RWF-2000](phase3_results/figures/confusion_matrix_rwf_2000.png)

#### Description des datasets

![Description des datasets](phase3_results/figures/dataset_description.png)

#### Optimisation du seuil theta

![Optimisation du seuil theta](phase3_results/figures/seuil_optimisation_f1.png)

#### Analyse statistique bootstrap

![Analyse statistique bootstrap](phase3_results/figures/statistical_analysis_bootstrap.png)

### 19.3 Captures d'erreurs phase 3 - faux negatifs

#### Faux negatif 1 - RLVS V_377 - frame centrale

![Faux negatif 1 - RLVS V_377 - frame centrale](phase3_results/figures/erreurs/fn_01_RLVS_score_0.007_V_377_central.png)

#### Faux negatif 1 - RLVS V_377 - frame extreme

![Faux negatif 1 - RLVS V_377 - frame extreme](phase3_results/figures/erreurs/fn_01_RLVS_score_0.007_V_377_extreme.png)

#### Faux negatif 2 - RWF-2000 - frame centrale

![Faux negatif 2 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fn_02_RWF2000_score_0.021_rwf_train_fight_0612_central.png)

#### Faux negatif 2 - RWF-2000 - frame extreme

![Faux negatif 2 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fn_02_RWF2000_score_0.021_rwf_train_fight_0612_extreme.png)

#### Faux negatif 3 - RWF-2000 - frame centrale

![Faux negatif 3 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fn_03_RWF2000_score_0.030_rwf_train_fight_0623_central.png)

#### Faux negatif 3 - RWF-2000 - frame extreme

![Faux negatif 3 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fn_03_RWF2000_score_0.030_rwf_train_fight_0623_extreme.png)

#### Faux negatif 4 - RWF-2000 - frame centrale

![Faux negatif 4 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fn_04_RWF2000_score_0.043_rwf_train_fight_0611_central.png)

#### Faux negatif 4 - RWF-2000 - frame extreme

![Faux negatif 4 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fn_04_RWF2000_score_0.043_rwf_train_fight_0611_extreme.png)

#### Faux negatif 5 - RWF-2000 - frame centrale

![Faux negatif 5 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fn_05_RWF2000_score_0.044_rwf_train_fight_0624_central.png)

#### Faux negatif 5 - RWF-2000 - frame extreme

![Faux negatif 5 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fn_05_RWF2000_score_0.044_rwf_train_fight_0624_extreme.png)

### 19.4 Captures d'erreurs phase 3 - faux positifs

#### Faux positif 1 - RLVS NV_14 - frame centrale

![Faux positif 1 - RLVS NV_14 - frame centrale](phase3_results/figures/erreurs/fp_01_RLVS_score_0.990_NV_14_central.png)

#### Faux positif 1 - RLVS NV_14 - frame extreme

![Faux positif 1 - RLVS NV_14 - frame extreme](phase3_results/figures/erreurs/fp_01_RLVS_score_0.990_NV_14_extreme.png)

#### Faux positif 2 - RWF-2000 - frame centrale

![Faux positif 2 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fp_02_RWF2000_score_0.982_rwf_train_nonfight_0678_central.png)

#### Faux positif 2 - RWF-2000 - frame extreme

![Faux positif 2 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fp_02_RWF2000_score_0.982_rwf_train_nonfight_0678_extreme.png)

#### Faux positif 3 - RLVS NV_117 - frame centrale

![Faux positif 3 - RLVS NV_117 - frame centrale](phase3_results/figures/erreurs/fp_03_RLVS_score_0.981_NV_117_central.png)

#### Faux positif 3 - RLVS NV_117 - frame extreme

![Faux positif 3 - RLVS NV_117 - frame extreme](phase3_results/figures/erreurs/fp_03_RLVS_score_0.981_NV_117_extreme.png)

#### Faux positif 4 - RWF-2000 - frame centrale

![Faux positif 4 - RWF-2000 - frame centrale](phase3_results/figures/erreurs/fp_04_RWF2000_score_0.981_rwf_train_nonfight_0225_central.png)

#### Faux positif 4 - RWF-2000 - frame extreme

![Faux positif 4 - RWF-2000 - frame extreme](phase3_results/figures/erreurs/fp_04_RWF2000_score_0.981_rwf_train_nonfight_0225_extreme.png)

#### Faux positif 5 - RLVS NV_788 - frame centrale

![Faux positif 5 - RLVS NV_788 - frame centrale](phase3_results/figures/erreurs/fp_05_RLVS_score_0.980_NV_788_central.png)

#### Faux positif 5 - RLVS NV_788 - frame extreme

![Faux positif 5 - RLVS NV_788 - frame extreme](phase3_results/figures/erreurs/fp_05_RLVS_score_0.980_NV_788_extreme.png)

## 20. Points restants et recommandations

1. Ne pas pousser `database/sirch.db` sur GitHub. Idealement, le retirer du suivi Git avec `git rm --cached database/sirch.db` et ajouter `database/sirch.db` dans `.gitignore`.

2. Ne pas exposer `config.py` publiquement, car il contient des parametres sensibles. Il est deja ignore par `.gitignore`.

3. Ajouter plus de videos non violentes de danse, sport, marche, foule et scenes quotidiennes pour reduire les faux positifs hors distribution.

4. Documenter dans le memoire que le modele est tres bon sur les jeux de test principaux, mais plus fragile sur des videos externes Creative Commons.

5. Pour usage reel, envisager :
   - sauvegarde de courts clips autour de l'incident,
   - calibration du seuil par environnement,
   - quantification TensorFlow Lite,
   - benchmark embarque reel,
   - collecte de donnees locales supplementaires.

6. Quand le token Telegram sera disponible, remplir simplement dans `config.py` :

```python
TELEGRAM_TOKEN = "token_recu"
TELEGRAM_CHAT_ID = "chat_id_recu"
```

Tant que ces valeurs restent placeholders, SIRCH ignore Telegram sans planter.

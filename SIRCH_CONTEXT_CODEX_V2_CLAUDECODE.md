# SIRCH — Fichier Contexte pour Codex
## Système Intelligent de Reconnaissance de Comportements Humains
## Application à la détection de violence en temps réel

---

## RÈGLES ABSOLUES POUR CODEX

1. Tu suis ce fichier à la lettre. Tu ne t'en écartes pas.
2. Tu ne fais RIEN qui pourrait endommager le PC, le système, ou les fichiers existants.
3. Si tu rencontres un problème et envisages une alternative :
   - Tu ARRÊTES immédiatement
   - Tu expliques SIMPLEMENT et CLAIREMENT ce que tu veux faire et pourquoi
   - Tu vérifies que l'alternative respecte :
     * Les 7 modules du projet (acquisition, prétraitement, inférence, décision, alertes, base de données, dashboard)
     * Les technologies imposées : TensorFlow, OpenCV, Flask, SQLite, Telegram, Gmail SMTP
     * Les contraintes matérielles : Windows 11, 8 Go RAM, RTX 2050 4 Go VRAM, 17.9 Go disque libre
     * Les paramètres fixés : N=20 frames, IMG=224x224, LSTM 256 unités, θ=0.5, K=5, cooldown 30s
   - Tu DEMANDES la permission avant de procéder
   - Tu n'agis QUE si l'utilisateur dit oui
4. Tu ne supprimes jamais de fichiers sans permission explicite.
5. Tu ne modifies jamais un fichier existant sans permission explicite.
6. À chaque étape, tu dis ce que tu fais, tu le fais, puis tu confirmes que c'est fait.
7. RÈGLE IMPORTANTE : chaque fois que l'utilisateur doit faire quelque chose manuellement,
   tu t'arrêtes, tu affiches un message clair avec le lien exact si nécessaire,
   tu expliques exactement quoi faire étape par étape,
   et tu attends qu'il confirme avant de continuer.

---

## 1. DESCRIPTION DU PROJET

SIRCH est un système de détection automatique de violence dans des flux vidéo en temps réel.
Il combine EfficientNetB0 (extraction spatiale) et LSTM (mémoire temporelle) pour distinguer
les actes violents des activités physiques intenses comme le sport et la danse.

### Les 7 modules du système
- Module 1 : Acquisition vidéo — webcam USB, flux IP/RTSP, fichier .mp4
- Module 2 : Prétraitement — resize 224x224, normalisation, séquences de N=20 frames
- Module 3a : EfficientNetB0 — extraction spatiale, sortie vecteur 1280 dimensions
- Module 3b : LSTM 256 unités — mémoire temporelle, sortie p entre 0 et 1
- Module 4 : Décision — fenêtre glissante K=5, seuil θ=0.5 configurable
- Module 5 : Alertes multicanaux — son, email SMTP, Telegram, cooldown 30 secondes
- Module 6 : Base de données SQLite — horodatage, score, capture d'écran automatique
- Module 7 : Tableau de bord Web Flask — historique, replay, export CSV

---

## 2. CONFIGURATION MATÉRIELLE DE LA MACHINE

| Composant | Détail |
|---|---|
| Modèle | Lenovo 82XV |
| OS | Windows 11 Home 64-bit |
| CPU | Intel Core i5-12450H, 8 cœurs / 12 threads, 2.00 GHz |
| RAM | 8 Go DDR5, 7.73 Go utilisables |
| GPU dédié | NVIDIA GeForce RTX 2050, 4 Go VRAM |
| GPU intégré | Intel UHD Graphics |
| Stockage libre | ~17.9 Go seulement |
| Pilote NVIDIA | 576.57 |

### Contraintes à respecter absolument
- Espace disque très limité (~17.9 Go) : ne jamais stocker les datasets en local sur le PC
- RAM limitée (8 Go) : batch size 8 à 16 maximum, jamais plus
- RTX 2050 (4 Go VRAM) : suffisante pour l'inférence en local, pas pour l'entraînement
- L'entraînement se fait obligatoirement sur Google Colab
- L'inférence en temps réel se fait en local sur ce PC

---

## 3. INSTALLATIONS À FAIRE VIA LE TERMINAL

Codex exécute chaque commande dans le terminal et vérifie le résultat avant de passer à la suivante.

### 3.1 Vérifier Python
```
python --version
```
Si le résultat n'est pas Python 3.10.x, Codex s'arrête et affiche ce message exact :

"⛔ ARRÊT — Python 3.10 n'est pas installé.
👉 Fais ceci :
1. Va sur ce lien : https://www.python.org/downloads/
2. Télécharge Python 3.10
3. Lance l'installateur
4. IMPORTANT : coche la case 'Add Python to PATH'
5. Clique Install Now
6. Une fois terminé, dis-moi 'c'est fait' pour continuer."

### 3.2 Mettre à jour pip
```
python -m pip install --upgrade pip
```

### 3.3 Créer l'environnement virtuel
```
python -m venv sirch_env
sirch_env\Scripts\activate
```

### 3.4 Créer le fichier requirements.txt
Codex crée ce fichier avec exactement ce contenu :
```
tensorflow==2.12.0
opencv-python>=4.8.0
numpy>=1.24.0
flask>=3.0.0
python-telegram-bot>=20.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=10.0.0
playsound>=1.3.0
pandas>=2.0.0
notebook>=7.0.0
```

### 3.5 Installer les dépendances
```
pip install -r requirements.txt
```

### 3.6 Vérifier le GPU
```
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```
Si la liste est vide, Codex affiche :
"ℹ️ Le GPU RTX 2050 n'est pas détecté par TensorFlow.
L'inférence se fera sur CPU. C'est normal pour ce PC sous Windows 11.
Aucune action requise — le modèle SIRCH est conçu pour fonctionner sur CPU.
Je continue."

### 3.7 Créer la structure des dossiers
```
mkdir colab
mkdir core
mkdir alerts
mkdir database
mkdir database\captures
mkdir dashboard
mkdir dashboard\templates
mkdir models
mkdir tests
```

---

## 4. FICHIER config.py

Codex crée ce fichier avec exactement ces valeurs par défaut.
L'utilisateur remplira ses vraies valeurs manuellement après.

```python
# config.py — Paramètres globaux du système SIRCH
# ATTENTION : remplis tes vraies valeurs avant de lancer le système

# === MODÈLE IA ===
MODEL_PATH = "models/sirch_model.h5"
N_FRAMES = 20
IMG_SIZE = 224
LSTM_UNITS = 256

# === MODULE 4 — DÉCISION ===
THRESHOLD = 0.5
K_WINDOW = 5
COOLDOWN_SECONDS = 30

# === MODULE 5 — TELEGRAM ===
TELEGRAM_TOKEN = "METS_TON_TOKEN_ICI"
TELEGRAM_CHAT_ID = "METS_TON_CHAT_ID_ICI"

# === MODULE 5 — EMAIL GMAIL ===
EMAIL_SENDER = "ton_email@gmail.com"
EMAIL_PASSWORD = "ton_mot_de_passe_application_gmail"
EMAIL_RECEIVER = "email_destinataire@gmail.com"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# === MODULE 6 — BASE DE DONNÉES ===
DB_PATH = "database/sirch.db"
CAPTURES_FOLDER = "database/captures/"

# === MODULE 7 — DASHBOARD FLASK ===
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
```

Après avoir créé config.py, Codex affiche ce message exact :

"✅ config.py créé.
⚠️ ACTION REQUISE DE TA PART :
Tu dois remplir tes vraies valeurs dans config.py.
Pour l'instant ne touche à rien — je t'expliquerai quoi remplir
et comment obtenir chaque valeur quand le moment viendra.
Dis-moi 'ok' pour continuer."

---

## 5. ARCHITECTURE DU MODÈLE IA

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

N_FRAMES = 20
IMG_SIZE = 224
LSTM_UNITS = 256
DROPOUT = 0.5

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

sequence_input = layers.Input(shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
features = layers.TimeDistributed(base_model)(sequence_input)
x = layers.LSTM(LSTM_UNITS, return_sequences=False)(features)
x = layers.Dropout(DROPOUT)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=sequence_input, outputs=output)
```

---

## 6. DATASETS — TÉLÉCHARGEMENT DIRECTEMENT DANS COLAB

Les datasets ne sont jamais téléchargés sur le PC local (espace insuffisant).
Codex gère le téléchargement directement dans le notebook Colab vers Google Drive.

### 6.1 Avant de créer le notebook, Codex affiche ce message exact :

"⚠️ ACTION REQUISE AVANT DE CONTINUER — kaggle.json

Pour télécharger le dataset RLVS, j'ai besoin de ton fichier kaggle.json.
Voici comment l'obtenir (c'est gratuit) :

1. Va sur ce lien : https://www.kaggle.com/settings/account
2. Clique sur l'onglet 'API Tokens'
3. Clique sur le bouton 'Create Legacy API Key'
4. Un fichier kaggle.json va se télécharger sur ton PC
5. Garde ce fichier — le notebook Colab te demandera de l'uploader
   au moment du téléchargement du dataset

Tu n'as pas besoin de faire ça maintenant.
Je continue à créer le code. Tu uploaderas kaggle.json plus tard dans Colab.
Dis-moi 'ok' pour continuer."

### 6.2 Téléchargement RLVS dans Colab
Le notebook train_sirch.ipynb doit contenir une cellule qui :
```python
# ============================================================
# CELLULE 1 — Monter Google Drive
# ============================================================
# INSTRUCTION POUR L'UTILISATEUR :
# Quand tu exécutes cette cellule, une popup va apparaître.
# Clique sur le lien, connecte-toi avec ton compte Google,
# et copie-colle le code d'autorisation ici.
# ============================================================
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive monté avec succès.")

# ============================================================
# CELLULE 2 — Créer les dossiers sur Drive
# ============================================================
import os
os.makedirs('/content/drive/MyDrive/SIRCH/datasets', exist_ok=True)
os.makedirs('/content/drive/MyDrive/SIRCH/models', exist_ok=True)
print("✅ Dossiers SIRCH créés sur ton Drive.")

# ============================================================
# CELLULE 3 — Uploader kaggle.json
# ============================================================
# INSTRUCTION POUR L'UTILISATEUR :
# Quand tu exécutes cette cellule, un bouton "Choisir un fichier"
# va apparaître. Clique dessus et sélectionne ton fichier kaggle.json
# que tu as téléchargé depuis https://www.kaggle.com/settings/account
# ============================================================
from google.colab import files
print("👇 Clique sur 'Choisir un fichier' et sélectionne ton fichier kaggle.json")
uploaded = files.upload()
os.makedirs('/root/.kaggle', exist_ok=True)
os.system('cp kaggle.json /root/.kaggle/')
os.system('chmod 600 /root/.kaggle/kaggle.json')
print("✅ kaggle.json configuré.")

# ============================================================
# CELLULE 4 — Télécharger RLVS depuis Kaggle
# ============================================================
os.system('pip install kaggle -q')
os.system('kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset -p /content/drive/MyDrive/SIRCH/datasets/ --unzip')
print("✅ Dataset RLVS téléchargé et extrait sur ton Drive.")
```

### 6.3 Téléchargement RWF-2000 dans Colab
```python
# ============================================================
# CELLULE 5 — Télécharger RWF-2000
# ============================================================
# INSTRUCTION POUR L'UTILISATEUR :
# RWF-2000 est distribué via un formulaire officiel.
# Va sur ce lien et remplis le formulaire :
# https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
# Tu recevras un lien Google Drive par email.
# Copie l'ID du fichier Drive (la partie entre /d/ et /view dans le lien)
# et remplace METS_ICI_L_ID ci-dessous.
# ============================================================
import gdown
file_id = "METS_ICI_L_ID_DU_FICHIER_DRIVE_RWF2000"
output = '/content/drive/MyDrive/SIRCH/datasets/RWF-2000.zip'
gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)
os.system(f'unzip {output} -d /content/drive/MyDrive/SIRCH/datasets/')
print("✅ Dataset RWF-2000 téléchargé et extrait sur ton Drive.")
```

### 6.4 Structure attendue sur Google Drive après téléchargement
```
Mon Drive/
└── SIRCH/
    ├── datasets/
    │   ├── RLVS/
    │   │   ├── Violence/
    │   │   └── NonViolence/
    │   └── RWF-2000/
    │       ├── train/
    │       │   ├── Fight/
    │       │   └── NonFight/
    │       └── val/
    │           ├── Fight/
    │           └── NonFight/
    └── models/
        └── sirch_model.h5
```

### 6.5 Répartition des données
- Train : 70% (2800 clips)
- Validation : 15% (600 clips)
- Test : 15% (600 clips)

---

## 7. PARAMÈTRES D'ENTRAÎNEMENT (sur Google Colab uniquement)

```python
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    '/content/drive/MyDrive/SIRCH/models/sirch_model.h5',
    monitor='val_loss',
    save_best_only=True
)
```

Après avoir créé le notebook d'entraînement, Codex affiche ce message exact :

"✅ Notebook d'entraînement créé : colab/train_sirch.ipynb

⚠️ QUAND TU SERAS PRÊT À ENTRAÎNER, fais ceci :
1. Va sur ce lien : https://colab.research.google.com
2. Clique sur Fichier → Ouvrir un notebook → Google Drive
3. Navigue vers SIRCH/colab/train_sirch.ipynb
   (ou uploade-le directement depuis ton PC)
4. Clique sur Exécution → Modifier le type d'exécution
5. Sélectionne GPU T4 dans 'Accélérateur matériel'
6. Clique Enregistrer
7. Clique sur Exécution → Tout exécuter
8. Ne ferme pas l'onglet pendant l'entraînement (2 à 4 heures)
9. Le modèle sera sauvegardé automatiquement sur ton Drive

Ne fais pas ça maintenant — je te le rappellerai à la fin.
Dis-moi 'ok' pour continuer."

---

## 8. MODULE DÉCISIONNEL — FENÊTRE GLISSANTE

```python
import collections

scores_buffer = collections.deque(maxlen=K_WINDOW)

def decide(new_score, threshold=THRESHOLD):
    scores_buffer.append(new_score)
    if len(scores_buffer) < K_WINDOW:
        return False
    mean_score = sum(scores_buffer) / K_WINDOW
    return mean_score >= threshold
```

---

## 9. STRUCTURE COMPLÈTE DU PROJET

```
SIRCH/
├── colab/
│   ├── train_sirch.ipynb
│   └── evaluate_sirch.ipynb
├── core/
│   ├── acquisition.py
│   ├── preprocessing.py
│   ├── inference.py
│   └── decision.py
├── alerts/
│   └── alertmanager.py
├── database/
│   ├── db.py
│   └── captures/
├── dashboard/
│   ├── app.py
│   └── templates/
│       ├── index.html
│       └── incidents.html
├── models/
│   └── sirch_model.h5
├── tests/
│   └── test_modules.py
├── config.py
├── main.py
└── requirements.txt
```

---

## 10. MÉTRIQUES D'ÉVALUATION OBLIGATOIRES

- Accuracy : (TP+TN) / (TP+TN+FP+FN)
- Précision : TP / (TP+FP)
- Rappel : TP / (TP+FN)
- F1-score : 2 * (Précision * Rappel) / (Précision + Rappel)
- Temps d'inférence moyen en ms/frame
- Taux de faux positifs sur vidéos de sport
- Taux de faux positifs sur vidéos de danse
- Référence à battre : F1 = 86.39% (Abdullah et al. 2023 sur RLVS seul)

---

## 11. MESSAGE FINAL QUE CODEX DOIT AFFICHER À LA TOUTE FIN

Quand tous les fichiers sont créés et tous les tests passent,
Codex affiche ce message exact :

"🎉 SIRCH EST PRÊT — Voici ce que tu dois faire maintenant :

─────────────────────────────────────────
ÉTAPE 1 — Remplir config.py
─────────────────────────────────────────
Ouvre le fichier config.py et remplis :

Pour Telegram :
1. Ouvre Telegram
2. Cherche @BotFather
3. Tape /newbot et suis les instructions
4. Copie le TOKEN reçu → colle dans TELEGRAM_TOKEN
5. Envoie un message à ton bot
6. Va sur : https://api.telegram.org/bot<TON_TOKEN>/getUpdates
7. Note le champ 'id' → colle dans TELEGRAM_CHAT_ID

Pour Gmail :
1. Va sur : https://myaccount.google.com/security
2. Active la validation en deux étapes si pas encore fait
3. Cherche 'Mots de passe des applications'
4. Crée un mot de passe pour 'Autre' → nomme-le SIRCH
5. Copie le mot de passe → colle dans EMAIL_PASSWORD

─────────────────────────────────────────
ÉTAPE 2 — Télécharger RWF-2000
─────────────────────────────────────────
1. Va sur : https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection
2. Remplis le formulaire pour recevoir le lien Drive
3. Quand tu reçois le lien, copie l'ID du fichier
4. Ouvre colab/train_sirch.ipynb et remplace METS_ICI_L_ID par cet ID

─────────────────────────────────────────
ÉTAPE 3 — Lancer l'entraînement sur Colab
─────────────────────────────────────────
1. Va sur : https://colab.research.google.com
2. Ouvre colab/train_sirch.ipynb
3. Exécution → Modifier le type d'exécution → GPU T4
4. Exécution → Tout exécuter
5. Attends 2 à 4 heures sans fermer l'onglet

─────────────────────────────────────────
ÉTAPE 4 — Télécharger le modèle
─────────────────────────────────────────
1. Va sur : https://drive.google.com
2. Navigue vers SIRCH/models/
3. Télécharge sirch_model.h5
4. Place-le dans le dossier models/ de ton projet

─────────────────────────────────────────
ÉTAPE 5 — Lancer SIRCH
─────────────────────────────────────────
Dans le terminal :
   sirch_env\Scripts\activate
   python main.py

Le dashboard sera accessible sur : http://localhost:5000"

---

## 12. POINTS D'ATTENTION ABSOLUS

1. Ne jamais stocker les datasets en local sur le PC
2. Tout téléchargement des datasets se fait dans Colab vers Google Drive
3. Tout entraînement sur Google Colab uniquement
4. Batch size maximum 16
5. Modèle sauvegardé uniquement en .h5
6. Seuil θ et fenêtre K modifiables via le dashboard Flask
7. Cooldown 30 secondes obligatoire dans alertmanager.py
8. SQLite uniquement
9. Flask uniquement
10. TensorFlow uniquement
11. Tests obligatoires sur vidéos de sport et danse
12. Tout le code doit fonctionner sur Windows 11
13. Ne rien faire qui endommage le PC ou les fichiers existants
14. Demander permission avant toute alternative
15. Afficher un message clair avec lien et instructions chaque fois
    que l'utilisateur doit faire quelque chose manuellement
16. Attendre la confirmation de l'utilisateur avant de continuer

---

## 13. ÉTAT ACTUEL DU PROJET — CE QUI A DÉJÀ ÉTÉ FAIT

> Cette section documente tout ce qui a été accompli. Ne refais rien de ce qui est listé ici.

### 13.1 Entraînement terminé

L'entraînement a été réalisé **en local sur le PC** (pas sur Colab — Google Drive était plein).
- Batch size utilisé : 2 (adapté aux 8 Go de RAM)
- Epochs réalisées : 15 (arrêt automatique via EarlyStopping patience=6)
- Meilleur modèle : **epoch 4**, val_loss = 0.3197
- Fichier du meilleur modèle : `C:\SIRCH_ENV\models\sirch_model.h5` (21.85 MB)
- Dernière epoch : 15, val_loss = 0.5927 (surapprentissage confirmé après epoch 4)

Callbacks actifs pendant l'entraînement :
- EarlyStopping : monitor=val_loss, patience=6, restore_best_weights=True
- ReduceLROnPlateau : monitor=val_loss, factor=0.5, patience=3, min_lr=1e-6
- ModelCheckpoint : save_best_only=True, monitor=val_loss

Progression complète des epochs :

| Epoch | Acc. train | Loss train | Val. accuracy | Val. loss | Val. précision | Val. rappel |
|-------|-----------|------------|---------------|-----------|----------------|-------------|
| 1 | 75.33% | 0.4958 | 83.64% | 0.3470 | 79.47% | 90.64% |
| 2 | 84.82% | 0.3264 | 84.47% | 0.3239 | 81.21% | 89.63% |
| 3 | 89.51% | 0.2448 | 83.97% | 0.3332 | 85.37% | 81.94% |
| 4 ✓ | 92.27% | 0.1890 | 86.64% | 0.3197 | 84.11% | 90.30% |
| 5 | 93.84% | 0.1425 | 86.48% | 0.3678 | 82.44% | 92.64% |
| 6 | 94.95% | 0.1356 | 86.81% | 0.3551 | 89.01% | 83.95% |
| 7 | 96.03% | 0.1034 | 86.81% | 0.3631 | 85.48% | 88.63% |
| 8 | 97.46% | 0.0727 | 88.15% | 0.3913 | 87.01% | 89.63% |
| 9 | 97.49% | 0.0626 | 89.15% | 0.3882 | 87.03% | 91.97% |
| 10 | 98.17% | 0.0543 | 88.31% | 0.3902 | 89.08% | 87.29% |
| 11 | 97.96% | 0.0455 | 87.31% | 0.4542 | 84.52% | 91.30% |
| 12 | 98.99% | 0.0320 | 86.98% | 0.4834 | 84.21% | 90.97% |
| 13 | 99.07% | 0.0251 | 88.81% | 0.5070 | 87.66% | 90.30% |
| 14 | 99.57% | 0.0140 | 88.81% | 0.5210 | 87.90% | 89.97% |
| 15 | 99.32% | 0.0213 | 87.48% | 0.5927 | 87.33% | 87.63% |

### 13.2 Évaluation finale — jeu de test (599 vidéos)

Le modèle epoch 4 a été évalué sur le jeu de TEST (jamais vu pendant l'entraînement).

| Métrique | Résultat |
|----------|---------|
| Accuracy | 88.98% |
| Précision | 86.07% |
| Rappel | 92.98% |
| F1-score | 89.39% |
| Temps CPU par frame | 54.27 ms |
| FPS effectif (CPU) | ~18 fps |
| GPU (RTX 2050) | Non détecté par TensorFlow 2.12 sous Windows 11 |

Matrice de confusion :
- Vrais négatifs (TN) : 255
- Faux positifs (FP) : 45
- Faux négatifs (FN) : 21
- Vrais positifs (TP) : 278

Comparaison avec la référence Abdullah et al. 2023 :
- Leur F1 sur RLVS seul : 86.39%
- Notre F1 sur RLVS + RWF-2000 : **89.39% (+3.00%)**

### 13.3 Test faux positifs — 30 vidéos Creative Commons

30 vidéos non-violentes téléchargées avec yt-dlp, vérifiées CC Attribution.

Dossiers :
- `datasets\test_sport\` — 15 vidéos (boxe, basketball, football)
- `datasets\test_danse\` — 10 vidéos (hip-hop, danse contemporaine)
- `datasets\test_calme\` — 5 vidéos (scènes calmes, référence)

Résultats par seuil :

| Seuil θ | Sport (15) | Danse (10) | Calme (5) | Global (30) |
|---------|-----------|-----------|----------|------------|
| 0.50 | 9/15 — 60.00% | 8/10 — 80.00% | 0/5 — 0.00% | 17/30 — 56.67% |
| 0.65 | 6/15 — 40.00% | 8/10 — 80.00% | 0/5 — 0.00% | 14/30 — 46.67% |
| 0.70 | 6/15 — 40.00% | 8/10 — 80.00% | 0/5 — 0.00% | 14/30 — 46.67% |

Résultats sauvegardés dans :
- `C:\SIRCH_ENV\models\test_faux_positifs.txt`
- `C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH\test_faux_positifs_seuils.txt`

### 13.4 Checkpoints disponibles sur le disque

Anciens checkpoints (format .h5) :
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_002.h5`
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_003.h5`
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_004.h5`
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_005.h5`
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_006.h5`
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_007.h5`

Nouveaux checkpoints (format .weights.h5, après correction du format) :
- `C:\SIRCH_ENV\models\checkpoints\sirch_weights_epoch_008.weights.h5`
- jusqu'à `sirch_weights_epoch_015.weights.h5`

Meilleur modèle final : `C:\SIRCH_ENV\models\sirch_model.h5` (epoch 4, 21.85 MB)

### 13.5 Corrections déjà appliquées au notebook

Dans `train_sirch.ipynb` :
- Format checkpoint corrigé : `.weights.h5` au lieu de `.h5`
- Logique de reprise mise à jour pour accepter les deux formats
- ReduceLROnPlateau ajouté : factor=0.5, patience=3, min_lr=1e-6, verbose=1
- EarlyStopping confirmé : patience=6, restore_best_weights=True, monitor=val_loss

---

## 14. CE QUI RESTE À FAIRE — PHASE 2 (RECHERCHE SCIENTIFIQUE)

> C'est la continuité du travail. Le directeur de recherche (Prof. Antoine Bagula) a validé les résultats et demande d'approfondir l'analyse scientifique. Voici exactement ce que tu dois faire, dans l'ordre.

### RÈGLE ABSOLUE POUR CETTE PHASE
- Ne jamais modifier sirch_model.h5 (epoch 4) — c'est le meilleur modèle final
- Demander permission avant chaque étape
- Sauvegarder tous les résultats dans des fichiers .txt ou .csv avant d'afficher
- Ne rien faire qui ralentisse ou endommage le PC pendant cette phase
- Tous les calculs se font sur le modèle epoch 4 uniquement

---

### ÉTAPE A — Réentraînement avec vidéos de sport et danse (PRIORITÉ 1)

**Objectif :** Vérifier si ajouter des vidéos de sport et de danse dans les données non-violentes réduit les faux positifs.

**Ce que tu dois faire :**

A1. Vérifie d'abord l'espace disque disponible. Demande ma permission avant de continuer.

A2. Crée un nouveau dossier de données augmentées :
```
datasets\train_augmente\Violence\     (copie des vidéos violentes existantes)
datasets\train_augmente\NonViolence\  (vidéos non-violentes existantes + sport + danse)
```

A3. Copie les 30 vidéos de test_sport et test_danse dans `datasets\train_augmente\NonViolence\` en plus des vidéos non-violentes existantes. Montre-moi le décompte avant et après.

A4. Crée un nouveau notebook `colab\train_sirch_augmente.ipynb` — copie exacte de `train_sirch.ipynb` avec :
- Données depuis `datasets\train_augmente\` au lieu des datasets originaux
- Nom du modèle sauvegardé : `sirch_model_augmente.h5` (jamais écraser sirch_model.h5)
- Même architecture, mêmes paramètres, même split 70/15/15
- Même EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

A5. Demande ma permission avant de lancer l'entraînement.

A6. Après l'entraînement, évalue `sirch_model_augmente.h5` sur :
- Le jeu de test standard (599 vidéos)
- Les 30 vidéos sport/danse/calme avec θ=0.5 et θ=0.65
- Compare les résultats avec ceux du modèle original (section 13.2 et 13.3)
- Sauvegarde la comparaison dans `resultats_comparaison_modeles.csv`

---

### ÉTAPE B — Métriques avancées (PRIORITÉ 2)

**Objectif :** Ajouter les courbes ROC, AUC, et courbes Precision-Recall sur le modèle epoch 4 ET sur le modèle augmenté si disponible.

B1. Sur le jeu de test (599 vidéos), calcule et sauvegarde :
- Courbe ROC avec l'AUC
- Courbe Precision-Recall avec l'AP (Average Precision)
- Matrice de confusion pour chaque seuil θ testé (0.50, 0.65, 0.70)

B2. Génère des graphiques matplotlib clairs pour chaque courbe. Sauvegarde en PNG haute résolution dans `models\figures\`.

B3. Sauvegarde toutes les valeurs numériques dans `models\metriques_avancees.csv`.

---

### ÉTAPE C — Étude d'ablation partielle (PRIORITÉ 3)

**Objectif :** Mesurer l'apport réel de chaque composant de l'architecture.

C1. Compare ces trois configurations sur le jeu de test (599 vidéos) :

| Configuration | Description |
|---|---|
| EfficientNetB0 seul | Sans LSTM — juste la moyenne des scores par frame |
| EfficientNetB0 + LSTM (modèle actuel) | Notre modèle epoch 4 |
| Impact du nombre de frames | Tester N=10 et N=30 si possible |

C2. Pour EfficientNetB0 seul : charge sirch_model.h5, désactive la couche LSTM et utilise la moyenne des scores frame par frame comme décision. Ne modifie pas le modèle — fais ça en inférence uniquement.

C3. Sauvegarde les résultats dans `models\ablation_results.csv`.

C4. Pour les différentes tailles de fenêtres glissantes K, utilise le modèle epoch 4 existant avec K=3, K=5, K=7, K=10. Sauvegarde dans `models\ablation_fenetre.csv`.

---

### ÉTAPE D — Analyse des biais des datasets (PRIORITÉ 4)

**Objectif :** Quantifier précisément le biais de composition des datasets. C'est la contribution scientifique principale selon le directeur de recherche.

D1. Sur le jeu de test standard (599 vidéos), calcule les FP et FN séparément par dataset d'origine (RLVS vs RWF-2000). Sauvegarde dans `models\biais_datasets.csv`.

D2. Compare les taux de faux positifs entre :
- Modèle entraîné sur RLVS + RWF-2000 (modèle actuel)
- Modèle entraîné avec augmentation sport/danse (modèle augmenté si disponible)

D3. Génère un graphique comparatif. Sauvegarde en PNG dans `models\figures\`.

---

### ÉTAPE E — Délai d'alerte (PRIORITÉ 5)

**Objectif :** Mesurer combien de secondes s'écoulent entre le début de la violence et le déclenchement de l'alerte.

E1. Avec K=5 séquences et N=20 frames à 25 fps :
- Latence buffer : 20/25 = 0.8 secondes
- Latence fenêtre glissante : K × 0.8 = 4.0 secondes au maximum

E2. Mesure le délai réel sur 10 vidéos violentes du jeu de test. Calcule la moyenne et l'écart-type. Sauvegarde dans `models\delai_alerte.csv`.

E3. Compare le délai pour K=3, K=5, K=7 pour montrer le compromis délai/précision.

---

### ORDRE D'EXÉCUTION

Commence par l'étape A. Quand elle est terminée, demande ma permission avant de passer à B. Et ainsi de suite.

Ne saute pas d'étape. Ne fais pas deux étapes en même temps.

---

## 15. FICHIERS ET DOSSIERS IMPORTANTS À CONNAÎTRE

```
C:\SIRCH_ENV\
├── models\
│   ├── sirch_model.h5                    ← MEILLEUR MODÈLE — NE JAMAIS ÉCRASER
│   ├── sirch_model_augmente.h5           ← sera créé à l'étape A
│   ├── training_log.csv                  ← log de l'entraînement original
│   ├── test_faux_positifs.txt            ← résultats test sport/danse
│   ├── test_faux_positifs_seuils.txt     ← résultats avec θ=0.65 et θ=0.70
│   ├── checkpoints\                      ← tous les checkpoints epochs 2-15
│   └── figures\                          ← sera créé pour les graphiques

C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH\
├── datasets\
│   ├── RLVS\                             ← 2000 vidéos originales
│   ├── RWF-2000\                         ← 1991 vidéos originales
│   ├── test_sport\                       ← 15 vidéos CC (boxe, basket, foot)
│   ├── test_danse\                       ← 10 vidéos CC (hip-hop, contemporaine)
│   ├── test_calme\                       ← 5 vidéos CC (scènes calmes)
│   └── train_augmente\                   ← sera créé à l'étape A
├── colab\
│   ├── train_sirch.ipynb                 ← notebook original (déjà corrigé)
│   ├── train_sirch_executed_*.ipynb      ← notebook exécuté sauvegardé
│   └── train_sirch_augmente.ipynb        ← sera créé à l'étape A
└── test_faux_positifs_seuils.txt         ← résultats comparaison seuils
```

---

## 16. CONTEXTE SCIENTIFIQUE — POUR COMPRENDRE LES DÉCISIONS

Le directeur de recherche Prof. Antoine Bagula a analysé les résultats et fait les observations suivantes :

1. La contribution scientifique la plus importante n'est pas le gain de F1 mais la mise en évidence du biais des datasets : RLVS et RWF-2000 contiennent peu de sport/danse dans leur catégorie non-violence, ce qui pousse le modèle à confondre activité physique intense et violence.

2. Il recommande d'approfondir cette analyse de biais plutôt que de chercher à tout prix à améliorer le F1.

3. Les métriques ROC/AUC et Precision-Recall sont nécessaires pour positionner le travail comme une recherche scientifique.

4. Toutes les améliorations doivent être comparées au modèle de base (epoch 4, sirch_model.h5) pour montrer clairement ce que chaque modification apporte.

La référence à battre reste Abdullah et al. 2023 : F1 = 86.39% sur RLVS seul. Notre modèle actuel obtient 89.39% sur RLVS + RWF-2000 combinés, ce qui est déjà une contribution valide.

---

## 17. ORDRE D'EXÉCUTION — CE QUE TU DOIS FAIRE QUAND JE T'ENVOIE CE FICHIER

Quand tu lis ce fichier, tu fais les choses exactement dans cet ordre. Tu ne passes pas à l'étape suivante sans ma permission explicite.

### ORDRE EXACT

**ÉTAPE 0 — Sauvegarde GitHub (AVANT TOUT)**

Avant de toucher quoi que ce soit dans le projet, tu sauvegardes tout sur GitHub.

0.1. Vérifie si git est installé :
```
git --version
```
Si git n'est pas installé, dis-le moi avec ce message exact et arrête :
"⛔ Git n'est pas installé sur ton PC. Va sur https://git-scm.com/download/win, télécharge et installe Git pour Windows, puis dis-moi 'c'est fait' pour continuer."

0.2. Va dans le dossier du projet :
```
cd C:\Users\djtra\Documents\Codex\2026-05-19\files-mentioned-by-the-user-sirch\SIRCH
```

0.3. Vérifie si git est déjà initialisé :
```
git status
```

0.4. Si git n'est pas initialisé, initialise-le :
```
git init
```

0.5. Crée le fichier `.gitignore` avec exactement ce contenu :
```
# Mots de passe et données sensibles
config.py

# Modèles trop lourds
*.h5
*.weights.h5

# Datasets trop lourds
datasets/RLVS/
datasets/RWF-2000/

# Vidéos de test (téléchargeables facilement)
datasets/test_sport/
datasets/test_danse/
datasets/test_calme/
datasets/train_augmente/

# Fichiers Python temporaires
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# Logs et résultats volumineux
*.csv
*.txt

# Captures d'écran automatiques
database/captures/
```

0.6. Montre-moi le contenu du .gitignore et demande ma permission avant de continuer.

0.7. Quand j'accepte, demande-moi l'URL de mon dépôt GitHub privé. Elle ressemble à :
`https://github.com/mon-username/SIRCH.git`

0.8. Quand je te donne l'URL, exécute dans l'ordre :
```
git add .
git commit -m "SIRCH - sauvegarde complète du projet - phase 2"
git branch -M main
git remote add origin URL_QUE_JE_T_AI_DONNÉE
git push -u origin main
```

0.9. Si git push demande un nom d'utilisateur et mot de passe GitHub, arrête-toi et affiche ce message exact :
"⚠️ GitHub te demande tes identifiants. Tu as deux options :
Option 1 — Entre ton nom d'utilisateur GitHub et un Personal Access Token (pas ton mot de passe).
Pour créer un token : va sur https://github.com/settings/tokens → Generate new token → coche 'repo' → copie le token → utilise-le comme mot de passe.
Option 2 — Dis-moi 'utilise SSH' et je configure ça autrement.
Qu'est-ce que tu préfères ?"

0.10. Confirme-moi quand le push est terminé avec le message exact :
"✅ Projet SIRCH sauvegardé sur GitHub. Tu peux vérifier sur ton dépôt GitHub."

---

**ÉTAPE 1 — Test du système complet (7 modules ensemble)**

Après la sauvegarde GitHub confirmée, tu testes que tout le système fonctionne de bout en bout sur le PC.

1.1. Vérifie que ces 7 fichiers existent :
- `core/acquisition.py`
- `core/preprocessing.py`
- `core/inference.py`
- `core/decision.py`
- `alerts/alertmanager.py`
- `database/db.py`
- `dashboard/app.py`

Pour chaque fichier manquant, dis-le moi clairement. Ne crée rien encore. Demande ma permission avant toute création.

1.2. Si tous les fichiers existent, lance un test de bout en bout avec une vidéo .mp4 du dossier `datasets/test_sport/`. Le test doit vérifier dans l'ordre :
- Module 1 : la vidéo s'ouvre et les frames se lisent
- Module 2 : les frames sont redimensionnées à 224x224 et normalisées
- Module 3 : EfficientNetB0 + LSTM produit un score entre 0 et 1
- Module 4 : la fenêtre glissante K=5 prend une décision
- Module 5 : une alerte est simulée (son + email — pas besoin d'envoyer un vrai email, juste vérifier que le code ne plante pas)
- Module 6 : un incident est enregistré dans SQLite
- Module 7 : le dashboard Flask démarre sur localhost:5000

1.3. Pour chaque module qui plante, montre-moi l'erreur exacte. Demande ma permission avant de corriger quoi que ce soit.

1.4. Quand tous les modules passent, affiche ce message :
"✅ Tous les 7 modules fonctionnent. Le système SIRCH est opérationnel sur ton PC."

---

**ÉTAPE 2 — Phase 2 scientifique**

Seulement après que l'étape 1 est confirmée, tu passes aux étapes A, B, C, D, E décrites dans la section 14.

Tu commences par l'étape A et tu demandes ma permission avant chaque étape suivante.


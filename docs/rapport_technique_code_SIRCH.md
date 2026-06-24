# Rapport technique du code SIRCH

## 1. Objectif

SIRCH analyse une webcam, une video ou un flux reseau afin d'estimer si une
sequence contient un comportement violent. Le systeme produit un score entre
0 et 1, stabilise la decision avec une moyenne temporelle, sauvegarde une
capture, enregistre l'incident dans SQLite et declenche les alertes configurees.

## 2. Architecture

Le programme est separe en sept modules :

| Module | Fichier | Responsabilite |
|---|---|---|
| Acquisition | `core/acquisition.py` | Ouvrir la source video et lire les frames. |
| Pretraitement | `core/preprocessing.py` | Transformer les frames et construire une sequence. |
| Inference | `core/inference.py` | Charger EfficientNetB0 + LSTM et calculer le score. |
| Decision | `core/decision.py` | Lisser les scores avec une fenetre K et appliquer le seuil. |
| Alertes | `alerts/alertmanager.py` | Son, Gmail et Telegram facultatif. |
| Donnees | `database/db.py` | Enregistrer incidents et reglages dans SQLite. |
| Dashboard | `dashboard/app.py` | Afficher historique, captures, reglages et export CSV. |

Le fichier `main.py` relie ces sept modules.

## 3. Acquisition video

La webcam Windows utilise DirectShow, une resolution demandee de 640 x 480,
15 FPS et un tampon minimal afin de reduire la latence.

```python
def open(self) -> None:
    if isinstance(self.source, int):
        self.capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
```

Le meme composant accepte aussi un chemin de fichier ou une URL RTSP/IP.

## 4. Pretraitement

Chaque frame BGR d'OpenCV est convertie en RGB, redimensionnee a 224 x 224 et
normalisee comme l'exige EfficientNetB0.

```python
def preprocess_frame(frame, img_size=IMG_SIZE):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    return preprocess_input(resized.astype(np.float32))
```

Le tampon conserve les 20 dernieres frames :

```python
def add_frame(self, frame):
    self.frames.append(preprocess_frame(frame, self.img_size))
    if len(self.frames) < self.n_frames:
        return None
    return np.expand_dims(np.stack(list(self.frames), axis=0), axis=0)
```

La forme envoyee au modele est `(1, 20, 224, 224, 3)`.

## 5. Modele d'intelligence artificielle

EfficientNetB0 extrait les caracteristiques visuelles de chaque image.
`TimeDistributed` applique le meme extracteur aux 20 frames. Le LSTM de
256 unites apprend ensuite l'evolution temporelle du mouvement.

```python
sequence_input = layers.Input(
    shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3)
)
features = layers.TimeDistributed(base_model)(sequence_input)
x = layers.LSTM(LSTM_UNITS, return_sequences=False)(features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)
```

La sortie sigmoid est une probabilite de violence entre 0 et 1.

## 6. Decision temporelle

Le systeme ne declenche pas une alerte sur un score isole. Il conserve les
`K=5` derniers scores et compare leur moyenne au seuil `theta`.

```python
def update(self, score):
    self.scores.append(float(score))
    mean_score = sum(self.scores) / len(self.scores)
    if len(self.scores) < self.k_window:
        return False, mean_score
    return mean_score >= self.threshold, mean_score
```

Le seuil et K sont lus depuis SQLite. Ils peuvent etre modifies dans le
dashboard sans changer le code.

## 7. Alertes

Le gestionnaire applique un cooldown de 30 secondes :

```python
def send_alerts(self, payload):
    if not self.can_send():
        return False
    self.last_alert_time = time.time()
    self.play_sound()
    self.send_email(payload)
    self.send_telegram(payload)
    return True
```

Telegram est ignore silencieusement lorsque le token ou le chat ID est vide ou
contient encore une valeur d'exemple. Gmail joint la capture de l'incident si
elle existe.

## 8. Base de donnees

La table `incidents` conserve :

- l'horodatage ;
- le score instantane ;
- la moyenne temporelle ;
- le chemin de la capture ;
- la source video.

```python
connection.execute(
    """
    INSERT INTO incidents
    (created_at, score, mean_score, capture_path, source)
    VALUES (?, ?, ?, ?, ?)
    """,
    (created_at, score, mean_score, capture_path, source),
)
```

La base locale et les captures personnelles sont exclues de GitHub.

## 9. Dashboard Flask

La page principale affiche les incidents et les reglages :

```python
@app.route("/")
def index():
    incidents = list_incidents(limit=10)
    settings = get_settings()
    return render_template(
        "index.html",
        incidents=incidents,
        threshold=settings["threshold"],
        k_window=settings["k_window"],
        cooldown=config.COOLDOWN_SECONDS,
    )
```

La route `/captures/<filename>` sert les images depuis le dossier absolu
calcule a partir de la racine du projet. La route `/export.csv` produit un
fichier exploitable pour l'analyse.

## 10. Orchestration temps reel

La boucle principale execute les etapes dans l'ordre :

```python
frame = video.read()
sequence = buffer.add_frame(frame)
score = inference.predict(sequence)
is_violence, mean_score = decision.update(score)

if is_violence:
    capture_path = save_capture(frame)
    add_incident(score, mean_score, capture_path, str(source))
    alerts.send_alerts(
        AlertPayload(score=mean_score, capture_path=capture_path)
    )
```

La fenetre OpenCV affiche en direct le score, la moyenne K, le seuil, le statut
et la derniere capture.

## 11. Resultats du modele original deploye

Evaluation sur 599 videos, avec `theta=0.50` :

| Metrique | Valeur |
|---|---:|
| Accuracy | 86,81 % |
| Precision | 82,74 % |
| Rappel | 92,98 % |
| F1-score | 87,56 % |
| ROC-AUC | 0,9446 |
| Matrice | TN=242, FP=58, FN=21, TP=278 |

## 12. Verification

Les controles effectues comprennent :

- sept tests unitaires reussis ;
- compilation Python sans erreur ;
- routes Flask `/`, `/incidents` et `/export.csv` en HTTP 200 ;
- ouverture des captures corrigee ;
- secrets et donnees personnelles exclus du depot ;
- presentation PowerPoint rendue et inspectee visuellement.

## 13. Commandes de demonstration

Depuis la racine du projet :

```powershell
C:\SIRCH_ENV\Scripts\python.exe -m dashboard.app
```

Puis dans un second terminal :

```powershell
C:\SIRCH_ENV\Scripts\python.exe main.py --source 0
```

Ouvrir `http://127.0.0.1:5000`.

Pour quitter, cliquer sur la fenetre camera puis appuyer sur `q`.

## 14. Limites

- Le modele peut produire des faux positifs sur des mouvements rapides.
- La qualite depend de l'angle, de l'eclairage et du cadrage.
- Le prototype enregistre une capture d'incident, pas un clip video complet.
- Une validation humaine reste necessaire avant toute action reelle.
- Les demonstrations physiques doivent etre simulees sans contact dangereux.

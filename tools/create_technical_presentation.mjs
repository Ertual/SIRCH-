import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const REPO = path.resolve(process.cwd());
const SCRATCH = path.join(os.tmpdir(), "codex-presentations", "sirch-technique", "tmp");
const OUTPUT = path.join(REPO, "docs", "presentation_technique_SIRCH.pptx");
const DASHBOARD_IMAGE = path.join(REPO, "docs", "dashboard_sirch.png");
const CONFUSION_IMAGE = path.join(
  REPO,
  "phase3_results",
  "figures",
  "confusion_matrix_global.png",
);
const ABLATION_IMAGE = path.join(
  REPO,
  "phase3_results",
  "figures",
  "ablation_complete.png",
);

const W = 1280;
const H = 720;
const C = {
  ink: "#101828",
  muted: "#667085",
  canvas: "#FFFFFF",
  panel: "#EDEDED",
  rule: "#B8BCC4",
  orange: "#FF6B35",
  green: "#117A65",
  red: "#B42318",
  blue: "#175CD3",
};

async function readImage(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, x, y, w, h, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: options.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: options.fontFamily || "Arial",
    fontSize: options.fontSize || 22,
    bold: options.bold || false,
    color: options.color || C.ink,
    alignment: options.alignment || "left",
    verticalAlignment: options.verticalAlignment || "top",
  };
  return shape;
}

function addRect(slide, x, y, w, h, fill = C.panel, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry || "rect",
    name: options.name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: {
      style: "solid",
      fill: options.lineFill || fill,
      width: options.lineWidth ?? 0,
    },
  });
}

function addHeader(slide, title, number, kicker = "SIRCH | PRESENTATION TECHNIQUE") {
  addText(slide, kicker, 42, 28, 540, 24, {
    fontSize: 14,
    bold: true,
    color: C.muted,
  });
  addText(slide, title, 42, 58, 1130, 70, {
    fontSize: 38,
    bold: true,
  });
  addText(slide, String(number).padStart(2, "0"), 1180, 660, 58, 22, {
    fontSize: 13,
    color: C.muted,
    alignment: "right",
  });
}

function addMetric(slide, x, y, w, value, label, accent = C.orange) {
  addRect(slide, x, y, w, 170, C.panel);
  addRect(slide, x, y, 8, 170, accent);
  addText(slide, value, x + 28, y + 25, w - 50, 64, {
    fontSize: 48,
    bold: true,
  });
  addText(slide, label, x + 28, y + 106, w - 50, 42, {
    fontSize: 18,
    color: C.muted,
  });
}

async function addImage(slide, imagePath, x, y, w, h, fit = "contain") {
  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
  slide.images.add({
    blob: await readImage(imagePath),
    contentType,
    alt: path.basename(imagePath),
    fit,
    position: { left: x, top: y, width: w, height: h },
  });
}

function addBulletList(slide, items, x, y, w, size = 21, gap = 54) {
  items.forEach((item, index) => {
    const top = y + index * gap;
    addRect(slide, x, top + 8, 12, 12, C.orange);
    addText(slide, item, x + 28, top, w - 28, gap - 4, {
      fontSize: size,
    });
  });
}

function addPipelineBox(slide, x, y, w, number, title, detail) {
  addRect(slide, x, y, w, 250, C.panel);
  addText(slide, number, x + 20, y + 18, 54, 52, {
    fontSize: 38,
    bold: true,
    color: C.orange,
  });
  addText(slide, title, x + 20, y + 82, w - 40, 56, {
    fontSize: 23,
    bold: true,
  });
  addText(slide, detail, x + 20, y + 145, w - 40, 86, {
    fontSize: 17,
    color: C.muted,
  });
}

function addCodePanel(slide, title, code, x, y, w, h, fontSize = 16) {
  addRect(slide, x, y, w, h, C.ink);
  addRect(slide, x, y, w, 42, "#1D2939");
  addText(slide, title, x + 18, y + 10, w - 36, 26, {
    fontSize: 17,
    bold: true,
    color: "#FFFFFF",
    fontFamily: "Consolas",
  });
  addText(slide, code, x + 18, y + 58, w - 36, h - 72, {
    fontSize,
    color: "#F2F4F7",
    fontFamily: "Consolas",
  });
}

function addExplanation(slide, title, items, x, y, w) {
  addText(slide, title, x, y, w, 44, {
    fontSize: 25,
    bold: true,
    color: C.green,
  });
  addBulletList(slide, items, x, y + 68, w, 18, 82);
}

async function buildDeck() {
  const deck = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Cover
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addRect(slide, 0, 0, 18, H, C.orange);
    addText(slide, "SIRCH", 54, 70, 480, 92, {
      fontSize: 72,
      bold: true,
    });
    addText(slide, "Presentation technique du code", 54, 190, 510, 120, {
      fontSize: 40,
      bold: true,
    });
    addText(
      slide,
      "Systeme intelligent de reconnaissance de comportements humains applique a la detection de violence en temps reel.",
      54,
      330,
      500,
      130,
      { fontSize: 23, color: C.muted },
    );
    addText(slide, "Code Python | TensorFlow | OpenCV | Flask | SQLite", 54, 582, 550, 40, {
      fontSize: 18,
      bold: true,
      color: C.green,
    });
    addRect(slide, 620, 48, 610, 610, "#F2F4F7");
    await addImage(slide, DASHBOARD_IMAGE, 648, 92, 554, 520, "contain");
    addText(slide, "01", 1180, 670, 58, 22, {
      fontSize: 13,
      color: C.muted,
      alignment: "right",
    });
  }

  // 2. Purpose
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Ce que fait SIRCH", 2);
    addText(
      slide,
      "Le systeme transforme un flux video en une decision exploitable, puis conserve une preuve et avertit l'utilisateur.",
      42,
      145,
      1120,
      74,
      { fontSize: 24, color: C.muted },
    );
    addPipelineBox(slide, 42, 260, 360, "1", "Observer", "Webcam, fichier video ou flux IP/RTSP.");
    addPipelineBox(slide, 460, 260, 360, "2", "Comprendre", "Analyse spatiale et temporelle de 20 frames.");
    addPipelineBox(slide, 878, 260, 360, "3", "Agir", "Alerte, capture, base SQLite et dashboard.");
    addText(
      slide,
      "Objectif: detecter rapidement une situation suspecte sans confondre systematiquement sport, danse et violence.",
      42,
      590,
      1190,
      58,
      { fontSize: 22, bold: true },
    );
  }

  // 3. Seven modules
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Architecture en 7 modules", 3);
    addRect(slide, 70, 304, 1140, 5, C.rule);
    const modules = [
      ["1", "Acquisition", "OpenCV"],
      ["2", "Pretraitement", "224 x 224"],
      ["3", "Inference IA", "EffNet + LSTM"],
      ["4", "Decision", "seuil + K"],
      ["5", "Alertes", "son, email, Telegram"],
      ["6", "Base", "SQLite + capture"],
      ["7", "Dashboard", "Flask"],
    ];
    modules.forEach((item, index) => {
      const x = 42 + index * 171;
      addRect(slide, x, 190, 150, 255, index === 3 ? "#FFF1EB" : C.panel, {
        lineFill: index === 3 ? C.orange : C.panel,
        lineWidth: index === 3 ? 2 : 0,
      });
      addText(slide, item[0], x + 16, 207, 50, 48, {
        fontSize: 34,
        bold: true,
        color: C.orange,
      });
      addText(slide, item[1], x + 16, 270, 118, 70, {
        fontSize: 19,
        bold: true,
      });
      addText(slide, item[2], x + 16, 360, 118, 55, {
        fontSize: 16,
        color: C.muted,
      });
    });
    addText(
      slide,
      "Chaque module a un fichier dedie. Cette separation facilite les tests, la maintenance et la demonstration en direct.",
      42,
      520,
      1160,
      76,
      { fontSize: 23, bold: true },
    );
  }

  // 4. AI core
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Le coeur IA: de la frame au score", 4);
    const nodes = [
      ["20 frames", "Fenetre temporelle"],
      ["224 x 224", "Redimensionnement"],
      ["EfficientNetB0", "Caracteristiques visuelles"],
      ["LSTM 256", "Evolution dans le temps"],
      ["Sigmoid", "Score p entre 0 et 1"],
    ];
    addRect(slide, 95, 320, 1090, 5, C.rule);
    nodes.forEach((node, index) => {
      const x = 42 + index * 245;
      addRect(slide, x, 225, 210, 210, index === 4 ? "#FFF1EB" : C.panel);
      addText(slide, node[0], x + 18, 255, 174, 58, {
        fontSize: index === 2 ? 24 : 30,
        bold: true,
        color: index === 4 ? C.orange : C.ink,
      });
      addText(slide, node[1], x + 18, 342, 174, 64, {
        fontSize: 17,
        color: C.muted,
      });
    });
    addText(slide, "Modele deploye: EfficientNetB0 gelee + LSTM + couches Dense/Dropout", 42, 500, 760, 44, {
      fontSize: 22,
      bold: true,
    });
    addText(slide, "Sortie: probabilite de violence", 42, 557, 500, 40, {
      fontSize: 20,
      color: C.green,
    });
    addText(slide, "Fichier principal: core/inference.py", 800, 520, 410, 52, {
      fontSize: 20,
      bold: true,
      alignment: "right",
    });
  }

  // 5. Code map
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Organisation du code", 5);
    const rows = [
      ["main.py", "Orchestre la boucle temps reel et l'affichage camera."],
      ["core/acquisition.py", "Ouvre webcam, video ou flux reseau."],
      ["core/preprocessing.py", "Construit les sequences de 20 frames."],
      ["core/inference.py", "Charge le modele et calcule le score."],
      ["core/decision.py", "Applique la moyenne K et le seuil."],
      ["alerts/alertmanager.py", "Declenche son, Gmail et Telegram."],
      ["database/db.py + dashboard/app.py", "Archive puis affiche les incidents."],
    ];
    addRect(slide, 42, 150, 1196, 52, C.ink);
    addText(slide, "Fichier", 62, 162, 330, 30, {
      fontSize: 18,
      bold: true,
      color: "#FFFFFF",
    });
    addText(slide, "Responsabilite", 400, 162, 800, 30, {
      fontSize: 18,
      bold: true,
      color: "#FFFFFF",
    });
    rows.forEach((row, index) => {
      const y = 202 + index * 62;
      addRect(slide, 42, y, 1196, 60, index % 2 === 0 ? "#F7F7F7" : C.canvas, {
        lineFill: C.rule,
        lineWidth: 1,
      });
      addText(slide, row[0], 62, y + 15, 320, 34, {
        fontSize: 17,
        bold: true,
        fontFamily: "Consolas",
      });
      addText(slide, row[1], 400, y + 15, 800, 34, {
        fontSize: 17,
      });
    });
  }

  // 6. Acquisition and preprocessing code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: acquisition et pretraitement", 6);
    addCodePanel(
      slide,
      "core/acquisition.py",
      `def open(self) -> None:
    if isinstance(self.source, int):
        self.capture = cv2.VideoCapture(
            self.source, cv2.CAP_DSHOW
        )
        self.capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            self.buffer_size,
        )
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )`,
      42,
      150,
      560,
      500,
    );
    addCodePanel(
      slide,
      "core/preprocessing.py",
      `def preprocess_frame(frame, img_size=IMG_SIZE):
    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )
    resized = cv2.resize(
        rgb, (img_size, img_size)
    )
    return preprocess_input(
        resized.astype(np.float32)
    )

def add_frame(self, frame):
    self.frames.append(
        preprocess_frame(frame)
    )
    if len(self.frames) < self.n_frames:
        return None
    return np.expand_dims(
        np.stack(list(self.frames)), axis=0
    )`,
      636,
      150,
      602,
      500,
    );
  }

  // 7. Inference code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: construction et chargement du modele", 7);
    addExplanation(
      slide,
      "Ce que montrent ces lignes",
      [
        "EfficientNetB0 extrait les caracteristiques de chaque frame.",
        "TimeDistributed applique le meme reseau aux 20 images.",
        "Le LSTM apprend l'evolution temporelle du mouvement.",
        "La couche sigmoid produit un score entre 0 et 1.",
      ],
      42,
      160,
      350,
    );
    addCodePanel(
      slide,
      "core/inference.py",
      `def build_model() -> Model:
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    sequence_input = layers.Input(
        shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3)
    )
    features = layers.TimeDistributed(
        base_model
    )(sequence_input)
    x = layers.LSTM(LSTM_UNITS)(features)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(
        1, activation="sigmoid"
    )(x)
    return Model(sequence_input, output)`,
      430,
      150,
      808,
      500,
    );
  }

  // 8. Decision code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: decision temporelle", 8);
    addCodePanel(
      slide,
      "core/decision.py",
      `class DecisionEngine:
    def update(self, score):
        self.scores.append(float(score))
        mean_score = (
            sum(self.scores) / len(self.scores)
        )

        if len(self.scores) < self.k_window:
            return False, mean_score

        return (
            mean_score >= self.threshold,
            mean_score,
        )

    def configure(self, threshold, k_window):
        self.threshold = float(threshold)
        if k_window != self.k_window:
            self.k_window = int(k_window)
            self.scores = collections.deque(
                maxlen=self.k_window
            )`,
      42,
      150,
      680,
      500,
    );
    addExplanation(
      slide,
      "Pourquoi cette logique",
      [
        "Un seul score eleve ne suffit pas.",
        "La moyenne de K=5 reduit les decisions instables.",
        "Le seuil est modifiable depuis le dashboard.",
        "La decision finale reste explicable et verifiable.",
      ],
      770,
      170,
      430,
    );
  }

  // 9. Alerts and database code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: alertes et enregistrement", 9);
    addCodePanel(
      slide,
      "alerts/alertmanager.py",
      `def send_alerts(self, payload):
    if not self.can_send():
        return False

    self.last_alert_time = time.time()
    self.play_sound()
    self.send_email(payload)
    self.send_telegram(payload)
    return True

def send_telegram(self, payload):
    token = (TELEGRAM_TOKEN or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or "").strip()
    if token == "" or chat_id == "":
        return
    if token == "METS_TON_TOKEN_ICI":
        return`,
      42,
      150,
      560,
      500,
    );
    addCodePanel(
      slide,
      "database/db.py",
      `def add_incident(
    score, mean_score,
    capture_path=None, source=None,
):
    init_db()
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO incidents "
            "(created_at, score, mean_score, "
            "capture_path, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                float(score), float(mean_score),
                capture_path, source,
            ),
        )
        connection.commit()`,
      636,
      150,
      602,
      500,
    );
  }

  // 10. Dashboard code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: dashboard Flask", 10);
    addExplanation(
      slide,
      "Fonctions exposees",
      [
        "La page principale affiche les dix derniers incidents.",
        "Le formulaire sauvegarde le seuil et la fenetre K.",
        "La route captures sert les images depuis le bon dossier.",
        "L'export CSV rend les incidents exploitables.",
      ],
      42,
      160,
      365,
    );
    addCodePanel(
      slide,
      "dashboard/app.py",
      `@app.route("/")
def index():
    incidents = list_incidents(limit=10)
    settings = get_settings()
    return render_template(
        "index.html",
        incidents=incidents,
        threshold=settings["threshold"],
        k_window=settings["k_window"],
    )

@app.route("/captures/<path:filename>")
def capture(filename):
    return send_from_directory(
        CAPTURES_DIR, Path(filename).name
    )`,
      445,
      150,
      793,
      500,
    );
  }

  // 11. Main loop code
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Code: orchestration temps reel", 11);
    addCodePanel(
      slide,
      "main.py",
      `with VideoSource(source) as video:
    while True:
        frame = video.read()
        if frame is None:
            break

        sequence = buffer.add_frame(frame)
        if sequence is None:
            continue

        score = inference.predict(sequence)
        is_violence, mean_score = decision.update(score)

        if is_violence:
            capture_path = save_capture(frame)
            add_incident(
                score, mean_score, capture_path, source
            )
            alerts.send_alerts(
                AlertPayload(mean_score, capture_path)
            )`,
      42,
      150,
      720,
      500,
    );
    addExplanation(
      slide,
      "La boucle relie les 7 modules",
      [
        "Lit une frame.",
        "Construit une sequence.",
        "Lance l'inference.",
        "Prend une decision.",
        "Capture, archive et alerte.",
      ],
      810,
      165,
      390,
    );
  }

  // 12. Live experience
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Fonctionnement en direct", 12);
    addText(slide, "Deux interfaces travaillent ensemble", 42, 142, 460, 46, {
      fontSize: 26,
      bold: true,
    });
    addBulletList(
      slide,
      [
        "Fenetre camera: image, score, moyenne K, seuil et statut.",
        "Dashboard web: reglages, historique, captures et export CSV.",
        "Mode faible latence: webcam 640 x 480, 15 FPS, prediction espacee.",
        "Le seuil et K peuvent etre modifies sans arreter le systeme.",
      ],
      42,
      220,
      500,
      19,
      76,
    );
    addRect(slide, 582, 142, 656, 500, "#F2F4F7");
    await addImage(slide, DASHBOARD_IMAGE, 604, 172, 612, 440, "contain");
  }

  // 13. Alerts and traceability
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Decision, alertes et tracabilite", 13);
    const items = [
      ["Score", "Le modele produit p."],
      ["Moyenne K=5", "Les scores sont lisses."],
      ["Seuil", "Violence si moyenne >= theta."],
      ["Capture", "Une image est sauvegardee."],
      ["Incident", "SQLite garde date et scores."],
    ];
    addRect(slide, 78, 280, 1120, 4, C.rule);
    items.forEach((item, index) => {
      const x = 42 + index * 245;
      addRect(slide, x, 200, 210, 178, index === 2 ? "#FFF1EB" : C.panel);
      addText(slide, item[0], x + 18, 225, 174, 40, {
        fontSize: 22,
        bold: true,
      });
      addText(slide, item[1], x + 18, 292, 174, 62, {
        fontSize: 16,
        color: C.muted,
      });
    });
    addText(slide, "Canaux d'alerte", 42, 450, 300, 42, {
      fontSize: 26,
      bold: true,
    });
    addText(slide, "Signal sonore", 42, 520, 250, 40, { fontSize: 21, bold: true });
    addText(slide, "Email Gmail + capture", 350, 520, 300, 40, { fontSize: 21, bold: true });
    addText(slide, "Telegram si configure", 720, 520, 300, 40, { fontSize: 21, bold: true });
    addText(slide, "Cooldown: 30 secondes", 1000, 520, 230, 40, {
      fontSize: 19,
      color: C.orange,
      alignment: "right",
    });
  }

  // 14. Main results
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Resultats valides sur 599 videos", 14);
    addText(slide, "Modele original deploye, seuil theta = 0,50", 42, 132, 530, 36, {
      fontSize: 19,
      color: C.muted,
    });
    addMetric(slide, 42, 190, 250, "86,81 %", "Accuracy");
    addMetric(slide, 310, 190, 250, "82,74 %", "Precision", C.blue);
    addMetric(slide, 42, 380, 250, "92,98 %", "Rappel", C.green);
    addMetric(slide, 310, 380, 250, "87,56 %", "F1-score", C.red);
    addRect(slide, 600, 145, 638, 500, "#F7F7F7");
    await addImage(slide, CONFUSION_IMAGE, 612, 162, 614, 466, "contain");
  }

  // 15. Research variants
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Ce que les experiences ont montre", 15);
    addRect(slide, 42, 132, 820, 520, "#F7F7F7");
    await addImage(slide, ABLATION_IMAGE, 58, 150, 788, 480, "contain");
    addText(slide, "Lecture simple", 900, 150, 300, 42, {
      fontSize: 26,
      bold: true,
    });
    addBulletList(
      slide,
      [
        "EfficientNet seul perd l'information temporelle.",
        "Le LSTM augmente fortement le rappel.",
        "Le GRU experimental obtient le meilleur compromis global.",
        "Le modele live reste le modele original valide et configure.",
      ],
      900,
      230,
      320,
      18,
      82,
    );
  }

  // 16. Demo procedure
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Demonstration live devant l'encadreur", 16);
    addText(slide, "Terminal 1 - dashboard", 42, 150, 360, 34, {
      fontSize: 20,
      bold: true,
      color: C.green,
    });
    addRect(slide, 42, 195, 1196, 82, "#101828");
    addText(
      slide,
      'C:\\SIRCH_ENV\\Scripts\\python.exe -m dashboard.app',
      64,
      220,
      1140,
      36,
      { fontSize: 20, color: "#FFFFFF", fontFamily: "Consolas" },
    );
    addText(slide, "Terminal 2 - detection webcam", 42, 315, 420, 34, {
      fontSize: 20,
      bold: true,
      color: C.green,
    });
    addRect(slide, 42, 360, 1196, 82, "#101828");
    addText(
      slide,
      'C:\\SIRCH_ENV\\Scripts\\python.exe main.py --source 0',
      64,
      385,
      1140,
      36,
      { fontSize: 20, color: "#FFFFFF", fontFamily: "Consolas" },
    );
    addText(slide, "Puis ouvrir http://127.0.0.1:5000", 42, 490, 650, 42, {
      fontSize: 26,
      bold: true,
    });
    addText(
      slide,
      "Test conseille: mouvements simules, sans contact physique dangereux. Montrer la fenetre camera, le score, l'alerte, la capture et l'incident dans le dashboard.",
      42,
      555,
      1160,
      76,
      { fontSize: 20, color: C.muted },
    );
  }

  // 17. Closing
  {
    const slide = deck.slides.add();
    slide.background.fill = C.canvas;
    addHeader(slide, "Projet reproductible et verifiable", 17);
    addMetric(slide, 42, 180, 270, "7", "modules integres", C.green);
    addMetric(slide, 335, 180, 270, "599", "videos de test", C.blue);
    addMetric(slide, 628, 180, 270, "21,85 MB", "modele original", C.orange);
    addMetric(slide, 921, 180, 270, "30 s", "cooldown alertes", C.red);
    addText(
      slide,
      "Le README explique l'installation, la configuration, les commandes de lancement, la structure du code, les tests et la demonstration live.",
      42,
      405,
      820,
      100,
      { fontSize: 25, bold: true },
    );
    addText(slide, "GitHub", 42, 545, 160, 34, {
      fontSize: 18,
      bold: true,
      color: C.muted,
    });
    addText(slide, "github.com/Ertual/SIRCH-", 42, 585, 650, 44, {
      fontSize: 28,
      bold: true,
      color: C.green,
    });
    addRect(slide, 930, 430, 260, 200, C.ink);
    addText(slide, "SIRCH", 965, 470, 190, 58, {
      fontSize: 42,
      bold: true,
      color: "#FFFFFF",
      alignment: "center",
    });
    addText(slide, "Code + preuve + live", 955, 550, 210, 42, {
      fontSize: 18,
      color: "#FFFFFF",
      alignment: "center",
    });
  }

  await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
  await fs.mkdir(`${SCRATCH}/preview`, { recursive: true });
  await fs.mkdir(`${SCRATCH}/layout`, { recursive: true });
  await fs.mkdir(`${SCRATCH}/qa`, { recursive: true });

  for (const [index, slide] of deck.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      `${SCRATCH}/preview/${stem}.png`,
      await deck.export({ slide, format: "png", scale: 1 }),
    );
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${SCRATCH}/layout/${stem}.json`, await layout.text());
  }
  await writeBlob(
    `${SCRATCH}/preview/montage.webp`,
    await deck.export({ format: "webp", montage: true, scale: 1 }),
  );

  const snapshot = await deck.inspect({
    kind: "slide,textbox,shape,image",
    maxChars: 12000,
  });
  await fs.writeFile(`${SCRATCH}/qa/deck-inspect.ndjson`, snapshot.ndjson);

  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUTPUT);
  console.log(OUTPUT);
}

buildDeck().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

# Parametres globaux de SIRCH.
# Copier ce fichier vers config.py, puis renseigner uniquement les valeurs locales.

# Modele IA
MODEL_PATH = r"C:\SIRCH_ENV\models\sirch_model.h5"
N_FRAMES = 20
IMG_SIZE = 224
LSTM_UNITS = 256

# Decision
THRESHOLD = 0.5
K_WINDOW = 5
COOLDOWN_SECONDS = 30

# Telegram: laisser les valeurs ci-dessous si Telegram n'est pas encore configure.
TELEGRAM_TOKEN = "METS_TON_TOKEN_ICI"
TELEGRAM_CHAT_ID = "METS_TON_CHAT_ID_ICI"

# Gmail: utiliser un mot de passe d'application, jamais le mot de passe Gmail normal.
EMAIL_SENDER = "votre_adresse@gmail.com"
EMAIL_PASSWORD = "ton_mot_de_passe_application_gmail"
EMAIL_RECEIVER = "adresse_destinataire@gmail.com"
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587

# Base de donnees et captures
DB_PATH = "database/sirch.db"
CAPTURES_FOLDER = "database/captures/"

# Dashboard Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

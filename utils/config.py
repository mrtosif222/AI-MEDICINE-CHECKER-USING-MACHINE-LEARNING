"""
config.py
App-wide constants and settings, kept in one place so nothing has to be
hardcoded (and duplicated) across multiple files.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# File upload settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
AUDIO_FOLDER = os.path.join(BASE_DIR, "static", "audio")
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_SIZE_MB = 5

# Supported UI / TTS languages
SUPPORTED_LANGUAGES = ["en", "hi", "ur", "mr"]
DEFAULT_LANGUAGE = "en"

# Interaction severity labels (must match preprocess.py / train_model.py)
SEVERITY_LEVELS = ["High", "Medium", "Low"]

# Paths to data/model files, used by the modules
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOCALES_DIR = os.path.join(BASE_DIR, "locales")

INTERACTIONS_CSV = os.path.join(DATA_DIR, "cleaned_interactions.csv")
MEDICINES_CSV = os.path.join(DATA_DIR, "cleaned_medicines.csv")
CATEGORY_MAP_CSV = os.path.join(DATA_DIR, "drug_category_map.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "interaction_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

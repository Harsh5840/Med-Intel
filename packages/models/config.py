import os
from pathlib import Path
import torch

# === Base Paths ===
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

# === Saved Artifact Paths ===
CLASSIFICATION_MODEL_PATH = SAVED_MODELS_DIR / "classification_model.pt"
LABEL_ENCODER_PATH = SAVED_MODELS_DIR / "label_encoder.pkl"
TOKENIZER_PATH = SAVED_MODELS_DIR  # HuggingFace tokenizer will load from directory

# === Training Config ===
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

# === Model Config ===
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLASSIFICATION_MODEL = "distilbert-base-uncased"

# === Classification Labels ===
LABELS = ["cardiology", "oncology", "neurology", "radiology", "immunology"]
NUM_LABELS = len(LABELS)

# === Runtime Device ===
DEVICE = torch.device("cuda" if os.environ.get("USE_CUDA", "1") == "1" and torch.cuda.is_available() else "cpu")

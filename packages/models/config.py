# packages/models/config.py

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

# Model filenames
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "classification_model.pkl"
EMBEDDING_MODEL_PATH = MODEL_DIR / "embedding_model.pkl"

# Training config
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 512

# Model names (for HuggingFace or local use)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLASSIFICATION_MODEL = "distilbert-base-uncased"

# Labels for classification (can be updated as needed)
LABELS = ["cardiology", "oncology", "neurology", "radiology", "immunology"]
NUM_LABELS = len(LABELS)

# Environment
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

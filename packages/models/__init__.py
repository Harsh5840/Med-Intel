# packages/models/__init__.py

from .model import MedicalTextClassifier, load_embedding_model
from .train import train_model
from .inference import classify_text, embed_text, load_classification_model
from .preprocess import clean_text, prepare_training_data, encode_labels, decode_label
from .utils import save_model, evaluate_model
from .config import (
    CLASSIFICATION_MODEL_PATH,
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEVICE,
    LABELS
)

__all__ = [
    "MedicalTextClassifier",
    "load_embedding_model",
    "train_model",
    "classify_text",
    "embed_text",
    "load_classification_model",
    "clean_text",
    "prepare_training_data",
    "encode_labels",
    "decode_label",
    "save_model",
    "evaluate_model",
    "CLASSIFICATION_MODEL_PATH",
    "DEFAULT_CLASSIFICATION_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEVICE",
    "LABELS"
]

# packages/models/inference.py

import torch
from transformers import AutoTokenizer
from .model import MedicalTextClassifier, load_embedding_model
from .config import CLASSIFICATION_MODEL_PATH, DEFAULT_CLASSIFICATION_MODEL, DEVICE
import pickle

def load_classification_model(model_path=CLASSIFICATION_MODEL_PATH):
    """
    Load the saved classification model from disk.
    """
    model = MedicalTextClassifier()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CLASSIFICATION_MODEL)
    return model, tokenizer


def classify_text(text, model, tokenizer, max_length=512):
    """
    Predict label for a given medical text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class


def embed_text(text, model, tokenizer, max_length=512):
    """
    Generate embedding vector for a given text input.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

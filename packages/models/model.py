# packages/models/model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch.nn as nn
from .config import DEFAULT_CLASSIFICATION_MODEL, NUM_LABELS, DEFAULT_EMBEDDING_MODEL

class MedicalTextClassifier(nn.Module):
    def __init__(self, model_name=DEFAULT_CLASSIFICATION_MODEL, num_labels=NUM_LABELS):
        super(MedicalTextClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def load_embedding_model(model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Load a transformer model for embedding generation.
    Returns the model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

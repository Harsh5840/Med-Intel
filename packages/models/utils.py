# packages/models/utils.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def save_model(model, path):
    """
    Save a PyTorch model to the given path.
    """
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved at {path}")


def load_pickle(path):
    """
    Load a pickled object (e.g., tokenizer, sklearn model).
    """
    return joblib.load(path)


def save_pickle(obj, path):
    """
    Save a Python object using joblib.
    """
    joblib.dump(obj, path)
    print(f"✅ Pickle saved at {path}")


def evaluate_model(predictions, true_labels):
    """
    Print evaluation metrics for classification.
    """
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=None)
    cm = confusion_matrix(true_labels, predictions)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)
    
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist()
    }

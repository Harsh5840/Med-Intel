# packages/models/preprocess.py

import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import LABELS

def clean_text(text):
    """
    Basic text cleaning for research abstracts or titles.
    """
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_labels(labels):
    """
    Encode text labels into numeric format for training.
    """
    le = LabelEncoder()
    le.fit(LABELS)
    return le.transform(labels)


def decode_label(label_idx):
    """
    Convert numeric prediction back to string label.
    """
    return LABELS[label_idx]


def prepare_training_data(csv_path, text_column="abstract", label_column="category"):
    """
    Load and preprocess data from a CSV file.
    Returns cleaned texts and encoded labels.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].apply(clean_text)
    labels = encode_labels(df[label_column].tolist())
    return df[text_column].tolist(), labels

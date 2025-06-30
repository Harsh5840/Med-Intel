import pandas as pd
from packages.models.train import train_model

# Load the dataset
df = pd.read_csv("data/processed/medical_dataset.csv")

# Prepare inputs
texts = df["abstract"].tolist()
labels = df["category"].tolist()

# Train the model
train_model(texts, labels)

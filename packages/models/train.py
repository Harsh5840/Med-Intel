import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import os

from .model import MedicalTextClassifier
from .config import (
    CLASSIFICATION_MODEL_PATH,
    LABEL_ENCODER_PATH,
    DEFAULT_CLASSIFICATION_MODEL,
    DEVICE,
    MAX_SEQ_LENGTH
)


def train_model(texts, labels, epochs=3, batch_size=16, learning_rate=2e-5):
    print("📊 Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    print("📦 Tokenizing texts...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CLASSIFICATION_MODEL)
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels_tensor = torch.tensor(encoded_labels)

    print("✂️ Splitting train/val...")
    train_idx, val_idx = train_test_split(range(len(texts)), test_size=0.1, random_state=42)
    train_inputs = {
        "input_ids": input_ids[train_idx],
        "attention_mask": attention_mask[train_idx],
        "labels": labels_tensor[train_idx]
    }
    val_inputs = {
        "input_ids": input_ids[val_idx],
        "attention_mask": attention_mask[val_idx],
        "labels": labels_tensor[val_idx]
    }

    print("🧠 Initializing model...")
    model = MedicalTextClassifier(num_labels=num_labels)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("🚀 Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = max(1, len(train_idx) // batch_size)

        for i in tqdm(range(0, len(train_idx), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_ids = train_inputs["input_ids"][i:i+batch_size].to(DEVICE)
            batch_mask = train_inputs["attention_mask"][i:i+batch_size].to(DEVICE)
            batch_labels = train_inputs["labels"][i:i+batch_size].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_ids, batch_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"✅ Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # Optional validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_ids = val_inputs["input_ids"][i:i+batch_size].to(DEVICE)
                batch_mask = val_inputs["attention_mask"][i:i+batch_size].to(DEVICE)
                batch_labels = val_inputs["labels"][i:i+batch_size].to(DEVICE)

                outputs = model(batch_ids, batch_mask)
                loss = criterion(outputs.logits, batch_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_idx) // batch_size)
        print(f"📉 Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")

    print("💾 Saving model & label encoder...")
    os.makedirs(CLASSIFICATION_MODEL_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFICATION_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    print(f"🎉 Done. Model saved to {CLASSIFICATION_MODEL_PATH}")

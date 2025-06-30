import os
import torch
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW

from .config import (
    DEFAULT_CLASSIFICATION_MODEL,
    CLASSIFICATION_MODEL_PATH,
    LABEL_ENCODER_PATH,
    TOKENIZER_PATH,
    LABELS,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    DEVICE
)
from .model import MedicalTextClassifier
from sklearn.preprocessing import LabelEncoder


class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_model(texts, labels):
    # Fit label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CLASSIFICATION_MODEL)

    # Prepare dataset
    dataset = MedicalDataset(texts, encoded_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model with correct number of labels
    model = MedicalTextClassifier(num_labels=len(label_encoder.classes_))
    model.to(DEVICE)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    # === Save artifacts ===
    os.makedirs(CLASSIFICATION_MODEL_PATH.parent, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), CLASSIFICATION_MODEL_PATH)

    # Save label encoder
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Save tokenizer
    tokenizer.save_pretrained(TOKENIZER_PATH)

    print(f"✅ Model, tokenizer, and encoder saved to {CLASSIFICATION_MODEL_PATH.parent}")

# packages/models/train.py

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm
import os

from .config import (
    DEFAULT_CLASSIFICATION_MODEL,
    CLASSIFICATION_MODEL_PATH,
    LABELS,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    DEVICE
)
from .model import MedicalTextClassifier

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
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CLASSIFICATION_MODEL)
    dataset = MedicalDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MedicalTextClassifier()
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0
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

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

    os.makedirs(CLASSIFICATION_MODEL_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), CLASSIFICATION_MODEL_PATH)
    print(f"✅ Model saved to {CLASSIFICATION_MODEL_PATH}")

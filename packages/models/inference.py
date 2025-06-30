import torch
import joblib
from transformers import AutoTokenizer
from .model import MedicalTextClassifier
from .config import (
    CLASSIFICATION_MODEL_PATH,
    LABEL_ENCODER_PATH,
    TOKENIZER_PATH,
    DEVICE,
    MAX_SEQ_LENGTH
)


# === Load all necessary components ===
def load_all():
    # Load label encoder
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))

    # Load model
    model = MedicalTextClassifier(num_labels=len(label_encoder.classes_))
    model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model, tokenizer, label_encoder


# === Perform inference ===
def classify_text(text: str, model, tokenizer, label_encoder, max_length=MAX_SEQ_LENGTH):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=1)
        pred_class_id = int(torch.argmax(probs, dim=1).item())
        confidence = probs[0][pred_class_id].item()
        predicted_label = label_encoder.inverse_transform([pred_class_id])[0]

    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4)
    }

# packages/models/diagnosis_model.py

from typing import List
import joblib
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "symptom_diagnosis_model.pkl")
VOCAB_PATH = os.path.join(BASE_DIR, "artifacts", "symptom_vocab.json")

class DiagnosisModel:
    def __init__(self, model_path: str = MODEL_PATH, vocab_path: str = VOCAB_PATH):
        self.model_path = model_path
        self.vocab_path = vocab_path

        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError("Diagnosis model not found. Train and save the model to 'artifacts/'.")

        if not os.path.exists(self.vocab_path):
            logger.error(f"Symptom vocabulary not found at {self.vocab_path}")
            raise FileNotFoundError("Symptom vocabulary not found. Add vocab file to 'artifacts/'.")

        logger.info("Loading diagnosis model...")
        self.model = joblib.load(self.model_path)
        self.symptom_vocab = self._load_symptom_vocab()

    def predict(self, symptoms: List[str]) -> str:
        """
        Predict the diagnosis based on input symptoms.
        """
        input_vector = self._vectorize(symptoms)
        prediction = self.model.predict([input_vector])[0]
        logger.info(f"Predicted diagnosis: {prediction}")
        return prediction

    def _vectorize(self, symptoms: List[str]):
        """
        Convert list of symptoms to binary vector based on vocab.
        """
        return [1 if symptom in symptoms else 0 for symptom in self.symptom_vocab]

    def _load_symptom_vocab(self) -> List[str]:
        with open(self.vocab_path, "r") as f:
            return json.load(f)

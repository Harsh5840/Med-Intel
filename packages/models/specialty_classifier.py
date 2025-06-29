from typing import List
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "speciality_classifier_model.pkl")

class SpecialityClassifier:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Speciality classifier model not found. Train and save to 'artifacts/'.")
        self.model = joblib.load(MODEL_PATH)

    def predict_speciality(self, symptoms: List[str]) -> str:
        """
        Predict the medical specialty based on user symptoms.
        E.g., returns 'Cardiology', 'Neurology', 'Oncology', etc.
        """
        input_vector = self._vectorize(symptoms)
        prediction = self.model.predict([input_vector])[0]
        return prediction

    def _vectorize(self, symptoms: List[str]):
        """
        Dummy symptom vectorizer — should be replaced with real feature extraction.
        """
        all_possible_symptoms = [
            "chest pain", "palpitations", "headache", "seizures", "fatigue", "nausea", "cough", "lump"
        ]
        vector = [1 if symptom in symptoms else 0 for symptom in all_possible_symptoms]
        return vector

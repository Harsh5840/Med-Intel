from typing import List
import joblib
import os

# Example path for the model (assumes a trained model is saved)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "symptom_diagnosis_model.pkl")

class DiagnosisModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Diagnosis model not found. Train and save the model to 'artifacts/'.")
        self.model = joblib.load(MODEL_PATH)

    def predict(self, symptoms: List[str]) -> str:
        """
        Predict the diagnosis based on input symptoms.
        Input should be a list of keywords like: ["fever", "cough", "headache"]
        """
        input_vector = self._vectorize(symptoms)
        prediction = self.model.predict([input_vector])[0]
        return prediction

    def _vectorize(self, symptoms: List[str]):
        """
        Convert list of symptoms to vector.
        For now, use a dummy bag-of-words encoding as a placeholder.
        """
        all_possible_symptoms = [
            "fever", "cough", "headache", "nausea", "fatigue", "rash", "shortness of breath"
        ]
        vector = [1 if symptom in symptoms else 0 for symptom in all_possible_symptoms]
        return vector

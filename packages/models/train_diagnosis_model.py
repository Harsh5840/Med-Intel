# Replace load_dummy_data() with real datasets (e.g., PubMed, MIMIC).

# SYMPTOM_VOCAB should eventually be synced between training and inference.

# Includes metrics for sanity check and saves model to artifacts directory.
import os
import joblib
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "symptom_diagnosis_model.pkl")

# Define your vocabulary for bag-of-words encoding
SYMPTOM_VOCAB = [
    "fever", "cough", "headache", "nausea", "fatigue",
    "rash", "shortness of breath", "dizziness", "chest pain"
]

def load_dummy_data() -> pd.DataFrame:
    """
    Load or simulate training data. In production, replace this with real data ingestion.
    """
    return pd.DataFrame([
        {"symptoms": ["fever", "cough"], "diagnosis": "Flu"},
        {"symptoms": ["headache", "nausea"], "diagnosis": "Migraine"},
        {"symptoms": ["fatigue", "rash"], "diagnosis": "Allergy"},
        {"symptoms": ["chest pain", "shortness of breath"], "diagnosis": "Heart Attack"},
        {"symptoms": ["dizziness", "nausea"], "diagnosis": "Vertigo"},
    ])

def vectorize(symptom_lists):
    """
    Vectorize a list of symptom lists into bag-of-words binary vectors.
    """
    vectors = []
    for symptoms in symptom_lists:
        vector = [1 if symptom in symptoms else 0 for symptom in SYMPTOM_VOCAB]
        vectors.append(vector)
    return vectors

def train_model():
    """
    Train and save a diagnosis model based on symptom inputs.
    """
    data = load_dummy_data()
    X = vectorize(data["symptoms"])
    y = data["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

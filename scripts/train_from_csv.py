import pandas as pd
from packages.models.train import train_model

def main():
    print("📂 Loading CSV dataset...")
    df = pd.read_csv("data/processed/medical_dataset.csv")

    print(f"✅ Total samples in CSV: {len(df)}")

    # 👇 Limit to first 5000 samples
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    texts = df["abstract"].tolist()
    labels = df["category"].tolist()

    print(f"🚀 Using {len(df)} samples for training...")
    train_model(texts, labels)

if __name__ == "__main__":
    main()

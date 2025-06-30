from datasets import load_dataset
import pandas as pd
import os

def prepare_dataset(output_path="data/processed/medical_dataset.csv", limit=20000):
    print("🔄 Downloading PubMed-RCT (20k)...")
    
    try:
        ds = load_dataset("armanc/pubmed-rct20k", split="train")
        print(f"Dataset loaded successfully. Shape: {ds.shape}")
        
        # Check the actual column names in the dataset
        print(f"Column names: {ds.column_names}")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(ds)
        
        # Check what columns actually exist before renaming
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Rename columns (adjust based on actual column names)
        # Common variations: 'text', 'sentence', 'abstract', 'input'
        if 'sentence' in df.columns:
            df = df.rename(columns={"sentence": "abstract"})
        elif 'text' in df.columns:
            df = df.rename(columns={"text": "abstract"})
        
        if 'label' in df.columns:
            df = df.rename(columns={"label": "category"})
        elif 'labels' in df.columns:
            df = df.rename(columns={"labels": "category"})
        
        # Map labels to categories
        label_map = {
            0: "background",
            1: "objective", 
            2: "methods",
            3: "results",
            4: "conclusions"
        }
        
        if 'category' in df.columns:
            # Check if labels are already strings or need mapping
            if df['category'].dtype in ['int64', 'int32']:
                df["category"] = df["category"].map(label_map)
                print(f"Mapped numeric labels to categories")
            else:
                print(f"Labels are already in string format: {df['category'].unique()}")
        
        # Shuffle and limit
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.head(limit)
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Categories distribution:\n{df['category'].value_counts()}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} samples to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Trying to inspect the dataset structure...")
        
        # Fallback: inspect dataset structure
        try:
            ds = load_dataset("armanc/pubmed-rct20k", split="train")
            print(f"Dataset info: {ds}")
            print(f"Features: {ds.features}")
            sample = ds[0]
            print(f"Sample entry: {sample}")
        except Exception as e2:
            print(f"Failed to load dataset: {e2}")

if __name__ == "__main__":
    prepare_dataset()
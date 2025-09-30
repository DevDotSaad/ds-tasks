from preprocessing import build_preprocessor
from models import train_and_evaluate
import pandas as pd
import os, json

ARTIFACTS_DIR = "artifacts"
CSV_PATH = "data/cars.csv"
TARGET = "Selling_Price"

def load_and_clean():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    if 'year' in df.columns:
        from datetime import datetime
        df['car_age'] = datetime.now().year - df['year']
    return df

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df = load_and_clean()

    print("Columns:", df.columns)
    print(df.head())

    TARGET = "Selling_Price"
    features = ["Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]

    X = df[[c for c in features if c in df.columns]]
    y = df[TARGET]
    preprocessor = build_preprocessor(X)
    results = train_and_evaluate(X, y, preprocessor, ARTIFACTS_DIR)
    with open(f"{ARTIFACTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Training complete. Metrics saved.")

if __name__ == "__main__":
    main()

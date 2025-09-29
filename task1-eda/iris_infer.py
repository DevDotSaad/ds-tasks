import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def predict_row(model, encoder, features):
    X = np.array(features, dtype=float).reshape(1, -1)
    y_pred = model.predict(X)[0]
    label = encoder.inverse_transform([y_pred])[0]
    return label

def predict_csv(model, encoder, csv_path: Path):
    df = pd.read_csv(csv_path)
    req = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")
    X = df[req].values
    y_pred = model.predict(X)
    labels = encoder.inverse_transform(y_pred)
    out = df.copy()
    out['PredictedSpecies'] = labels
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--features", type=float, nargs=4, help="Four features in order: SepalLengthCm SepalWidthCm PetalLengthCm PetalWidthCm")
    parser.add_argument("--csv_path", type=str, help="CSV with feature columns")
    args = parser.parse_args()

    model = joblib.load(args.model)
    encoder = joblib.load(args.encoder)

    if args.csv_path:
        out = predict_csv(model, encoder, Path(args.csv_path))
        out_path = Path(args.csv_path).with_suffix(".predicted.csv")
        out.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
    else:
        if not args.features or len(args.features) != 4:
            raise SystemExit("Provide --features four numeric values or --csv_path path/to/file.csv")
        label = predict_row(model, encoder, args.features)
        print(f"Predicted Species: {label}")

if __name__ == "__main__":
    main()

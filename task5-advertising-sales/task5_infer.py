import argparse, os, joblib
import pandas as pd
import joblib

def add_interaction(Xdf: pd.DataFrame):
    Xdf = Xdf.copy()
    Xdf["TVxRadio"] = Xdf["TV"] * Xdf["Radio"]
    return Xdf

ART_DIR = "artifacts"

parser = argparse.ArgumentParser(description="Predict Sales from ad spends.")
parser.add_argument("--tv", type=float, help="TV spend")
parser.add_argument("--radio", type=float, help="Radio spend")
parser.add_argument("--newspaper", type=float, help="Newspaper spend")
parser.add_argument("--csv", type=str, help="Optional: CSV path with columns TV,Radio,Newspaper")
args = parser.parse_args()

model_path = os.path.join(ART_DIR, "best_model.pkl")
assert os.path.exists(model_path), "❌ Train first: best_model.pkl not found."
model = joblib.load(model_path)

if args.csv:
    df = pd.read_csv(args.csv)
    preds = model.predict(df[["TV", "Radio", "Newspaper"]])
    df["Predicted_Sales"] = preds
    out = os.path.join(ART_DIR, "batch_predictions.csv")
    df.to_csv(out, index=False)
    print(f"✅ Saved batch predictions to {out}")
else:
    assert args.tv is not None and args.radio is not None and args.newspaper is not None, \
        "❌ Provide --tv --radio --newspaper or use --csv"
    row = pd.DataFrame([{"TV": args.tv, "Radio": args.radio, "Newspaper": args.newspaper}])
    pred = model.predict(row)[0]
    print(f"✅ Predicted Sales: {pred:.3f}")

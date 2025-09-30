import argparse, os, json, pathlib
from src.train import train_and_save
from src.forecast import forecast_years
from src.data_utils import load_dataset, add_features
from src.eda import plot_series

ROOT = pathlib.Path(__file__).parent.resolve()
OUT_DIR = ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
ARTIFACTS_DIR = ROOT / "artifacts"

def run(data_path: str, forecast_n: int = 0):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    metrics, df_raw = train_and_save(data_path, artifacts_dir=str(ARTIFACTS_DIR), target_col="headcount_ratio_national", nlags=2)

    from src.data_utils import load_dataset
    df = load_dataset(data_path)
    plot_series(df, "year", "headcount_ratio_national", "National headcount ratio over time", str(PLOTS_DIR / "headcount_national.png"))

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    forecasts_df = None
    if forecast_n and forecast_n > 0:
        forecasts_df = forecast_years(data_path, artifacts_dir=str(ARTIFACTS_DIR), years_ahead=forecast_n, target_col="headcount_ratio_national")
        forecasts_df.to_csv(OUT_DIR / "forecasts.csv", index=False)
        print("\nForecasts saved to outputs/forecasts.csv")
        print(forecasts_df.to_string(index=False))
    return df_raw, metrics, forecasts_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str((ROOT / "data" / "Pakistan_Poverty_Dataset_2000_2023.csv")), help="Path to CSV")
    ap.add_argument("--forecast", type=int, default=5, help="Years ahead to forecast")
    args = ap.parse_args()
    data_path = args.data if os.path.exists(args.data) else str(ROOT / "data" / "Pakistan_Poverty_Dataset_2000_2023.csv")
    run(data_path, forecast_n=args.forecast)

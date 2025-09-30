import numpy as np
import pandas as pd
import pickle, os
from .data_utils import load_dataset, add_features, design_matrix

def forecast_years(data_path: str, artifacts_dir: str, years_ahead: int = 5, target_col: str = "headcount_ratio_national"):
    with open(os.path.join(artifacts_dir, "model.pkl"), "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    nlags = bundle["nlags"]
    feat_names = bundle["features"]

    hist = load_dataset(data_path).copy()
    last_year = int(hist["year"].max())
    future_years = list(range(last_year+1, last_year+1+years_ahead))

    preds = []
    for y in future_years:
        new_row = {"year": y}
        for col in ["unemployment_rate_pct","population_millions","gdp_growth_rate_pct","inflation_rate_pct"]:
            if col in hist.columns:
                new_row[col] = float(hist[col].iloc[-1])
        df_tmp = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        df_tmp_feat = add_features(df_tmp, target_col=target_col, nlags=nlags)
        X_last, _ = design_matrix(df_tmp_feat, target_col=target_col, nlags=nlags)
        x = X_last.iloc[[-1]][feat_names]
        yhat = float(model.predict(x)[0])
        preds.append({"year": y, target_col: yhat})
        hist = pd.concat([hist, pd.DataFrame([{"year": y, target_col: yhat, **{k:new_row.get(k, np.nan) for k in new_row if k!='year'}}])], ignore_index=True)

    return pd.DataFrame(preds, columns=["year", target_col])

import math
import json, pickle, os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_utils import load_dataset, add_features, train_test_split_by_year, design_matrix

def train_and_save(data_path: str, artifacts_dir: str, target_col: str = "headcount_ratio_national", nlags: int = 2):
    df_raw = load_dataset(data_path)
    df_feat = add_features(df_raw, target_col=target_col, nlags=nlags)
    train_df, test_df = train_test_split_by_year(df_feat, test_years=4)
    Xtr, ytr = design_matrix(train_df, target_col=target_col, nlags=nlags)
    Xte, yte = design_matrix(test_df, target_col=target_col, nlags=nlags)

    alphas = np.logspace(-3, 3, 25)
    model = RidgeCV(alphas=alphas, cv=min(5, max(2, len(Xtr)//3)))
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    metrics = {
        "MAE": float(mean_absolute_error(yte, preds)),
        "RMSE": math.sqrt(mean_squared_error(yte, preds)),
        "R2": float(r2_score(yte, preds)),
        "alpha": float(model.alpha_),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte))
    }

    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "nlags": nlags,
            "features": list(Xtr.columns),
            "target": target_col,
        }, f)
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, df_raw

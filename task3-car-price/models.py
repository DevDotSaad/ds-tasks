from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

def eval_and_log(name, y_true, y_pred, save_plot_path=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE={mae:.0f} | RMSE={rmse:.0f} | R²={r2:.3f}")
    if save_plot_path:
        resid = y_true - y_pred
        plt.figure()
        plt.scatter(y_pred, resid, alpha=0.5, s=10)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted"); plt.ylabel("Residuals")
        plt.title(f"Residuals — {name}")
        plt.savefig(save_plot_path, dpi=120, bbox_inches="tight")
        plt.close()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def train_and_evaluate(X, y, preprocessor, ARTIFACTS_DIR):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}

    linreg = Pipeline([("prep", preprocessor), ("model", LinearRegression())])
    linreg.fit(X_train, y_train)
    results["LinearRegression"] = eval_and_log("LinearRegression", y_valid, linreg.predict(X_valid), f"{ARTIFACTS_DIR}/resid_linear.png")

    rf = Pipeline([("prep", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
    rf.fit(X_train, y_train)
    results["RandomForest"] = eval_and_log("RandomForest", y_valid, rf.predict(X_valid), f"{ARTIFACTS_DIR}/resid_rf.png")

    gbr = Pipeline([("prep", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])
    gbr.fit(X_train, y_train)
    results["GradientBoosting"] = eval_and_log("GradientBoosting", y_valid, gbr.predict(X_valid), f"{ARTIFACTS_DIR}/resid_gbr.png")

    dump(linreg, f"{ARTIFACTS_DIR}/model_linear.pkl")
    dump(rf, f"{ARTIFACTS_DIR}/model_rf.pkl")
    dump(gbr, f"{ARTIFACTS_DIR}/model_gbr.pkl")
    dump(preprocessor, f"{ARTIFACTS_DIR}/preprocessor.pkl")
    
    model = rf.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        feature_names = (preprocessor.named_transformers_["num"].feature_names_in_.tolist() +
                         list(preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out()))
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
        plt.figure()
        importances[::-1].plot(kind="barh")
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(f"{ARTIFACTS_DIR}/feature_importance.png", dpi=120, bbox_inches="tight")
        plt.close()

    return results

import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

DATA_PATH = os.path.join("data", "Advertising.csv")
assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)

sns.pairplot(df[["TV","Radio","Newspaper","Sales"]])
plt.savefig(os.path.join(ART_DIR, "pairplot.png"))
plt.close()

corr = df.corr(numeric_only=True)
plt.figure()
sns.heatmap(corr, annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(ART_DIR, "corr_heatmap.png"))
plt.close()

X = df[["TV","Radio","Newspaper"]].copy()
y = df["Sales"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def add_interaction(Xdf: pd.DataFrame):
    Xdf = Xdf.copy()
    Xdf["TVxRadio"] = Xdf["TV"] * Xdf["Radio"]
    return Xdf

interaction_adder = FunctionTransformer(add_interaction)

num_features_basic = ["TV","Radio","Newspaper"]
num_features_inter = ["TV","Radio","Newspaper","TVxRadio"]

basic_prep = ColumnTransformer(
    transformers=[("scale", StandardScaler(), num_features_basic)],
    remainder="drop"
)

inter_prep = ColumnTransformer(
    transformers=[("scale", StandardScaler(), num_features_inter)],
    remainder="drop"
)

models = {
    "linear_baseline": Pipeline([
        ("prep", basic_prep),
        ("lin", LinearRegression())
    ]),
    "linear_interaction": Pipeline([
        ("add_inter", interaction_adder),
        ("prep", inter_prep),
        ("lin", LinearRegression())
    ]),
    "ridge_cv": Pipeline([
        ("add_inter", interaction_adder),
        ("prep", inter_prep),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5))
    ]),
    "lasso_cv": Pipeline([
        ("add_inter", interaction_adder),
        ("prep", inter_prep),
        ("lasso", LassoCV(alphas=np.logspace(-3, 1, 25), cv=5, max_iter=10000, random_state=42))
    ]),
    "random_forest": Pipeline([
        ("add_inter", interaction_adder),
        ("prep", inter_prep),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42))
    ]),
    "gbr": Pipeline([
        ("add_inter", interaction_adder),
        ("prep", inter_prep),
        ("gbr", GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, random_state=42))
    ]),
}

def eval_and_plot(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    preds = pipe.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    r2 = r2_score(yte, preds)
    plt.scatter(preds, yte - preds)
    plt.axhline(0, ls="--")
    plt.title(f"Residuals: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.savefig(os.path.join(ART_DIR, f"resid_{name}.png"))
    plt.close()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

metrics, best_model, best_r2 = {}, None, -1
for name, pipe in models.items():
    m = eval_and_plot(name, pipe, X_train, y_train, X_test, y_test)
    metrics[name] = m
    if m["R2"] > best_r2:
        best_r2, best_model = m["R2"], pipe

import json, joblib
with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(best_model, os.path.join(ART_DIR, "best_model.pkl"))

preds = best_model.predict(X_test)
out_df = X_test.copy()
out_df["Actual"] = y_test
out_df["Predicted"] = preds
out_df.to_csv(os.path.join(ART_DIR, "predictions_sample.csv"), index=False)

print("✅ Training complete. Artifacts saved in artifacts/")

import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


REQUIRED_COLS = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']

def validate_and_load(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}.\n"
            "👉 Please download the exact Kaggle dataset used in the PDF:\n"
            "   kaggle datasets download -d saurabh00007/iriscsv -p data --unzip\n"
            "   and ensure 'Iris.csv' is placed at data/Iris.csv"
        )
    df = pd.read_csv(csv_path)
    
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")
    
    if df.isna().any().any():
        df = df.dropna().reset_index(drop=True)
    
    for c in REQUIRED_COLS[:-1]:
        df[c] = pd.to_numeric(df[c], errors='raise')
    df['Species'] = df['Species'].astype(str)
    return df

def get_models():
    models = {
        'logreg': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))]),
        'svc': Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True))]),
        'rf': Pipeline([('clf', RandomForestClassifier(random_state=0))]),
        'knn': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
    }
    return models

def cv_score(pipe: Pipeline, X, y, cv, metric='f1_macro'):
    return cross_val_score(pipe, X, y, cv=cv, scoring=metric)

def plot_confusion(cm, classes, out_path: Path):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/Iris.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    csv_path = Path(args.data_path)
    df = validate_and_load(csv_path)

    X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
    y_text = df['Species'].values

    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    models = get_models()

    cv_results = {}
    for name, pipe in models.items():
        f1_scores = cv_score(pipe, X_train, y_train, cv, metric='f1_macro')
        acc_scores = cv_score(pipe, X_train, y_train, cv, metric='accuracy')
        cv_results[name] = {
            'f1_macro_mean': float(np.mean(f1_scores)),
            'f1_macro_std': float(np.std(f1_scores)),
            'accuracy_mean': float(np.mean(acc_scores)),
            'accuracy_std': float(np.std(acc_scores)),
        }
        print(f"[CV] {name}: F1={np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f} | Acc={np.mean(acc_scores):.4f}±{np.std(acc_scores):.4f}")

    best_name = max(cv_results.items(), key=lambda kv: kv[1]['f1_macro_mean'])[0]
    print(f"Best baseline model: {best_name}")

    grids = {
        'svc': {
            'clf__C': [0.1, 1, 3, 10],
            'clf__gamma': ['scale', 0.1, 0.01],
            'clf__kernel': ['rbf', 'linear']
        },
        'rf': {
            'clf__n_estimators': [100, 200, 400],
            'clf__max_depth': [None, 3, 5, 8]
        },
        'logreg': {
            'clf__C': [0.1, 1.0, 3.0, 10.0],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs']
        },
        'knn': {
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance']
        }
    }

    best_pipe = models[best_name]
    param_grid = grids.get(best_name, {})
    if param_grid:
        print(f"Tuning {best_name} with GridSearchCV...")
        gs = GridSearchCV(best_pipe, param_grid, scoring='f1_macro', cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        best_params = gs.best_params_
        print("Best params:", best_params)
    else:
        best_pipe.fit(X_train, y_train)
        best_params = {}

    y_pred = best_pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)

    print("\\nHoldout Evaluation:")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
    print(report)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(best_pipe, artifacts_dir / "model.joblib")
    joblib.dump(le, artifacts_dir / "label_encoder.joblib")

    with open(artifacts_dir / "metrics.json", "w") as f:
        json.dump({
            "cv_results": cv_results,
            "best_model": best_name,
            "best_params": best_params,
            "holdout": {"accuracy": acc, "macro_f1": f1}
        }, f, indent=2)

    plot_confusion(cm, le.classes_, artifacts_dir / "confusion_matrix.png")
    print(f"Saved artifacts to: {artifacts_dir.resolve()}")

if __name__ == "__main__":
    main()

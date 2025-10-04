from pathlib import Path
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = Path("data/raw/creditcard.csv")
MODEL_DIR = Path("notebooks/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y


def main():
    X, y = load_data()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    params = dict(
        penalty="l2",
        C=0.1,                   # regularization strength
        solver="liblinear",      # good for small & imbalanced datasets
        class_weight="balanced", # adjust for fraud rarity
        max_iter=1000,
        random_state=42
    )

    model = LogisticRegression(**params)

    with mlflow.start_run(run_name="logreg_baseline_creditcard"):
        model.fit(X_train, y_train)

    for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = model.predict_proba(Xs)[:, 1]
        yhat = (prob >= 0.5).astype(int)

        roc = roc_auc_score(ys, prob)
        ap = average_precision_score(ys, prob)
        mlflow.log_metric(f"roc_auc_{split}", roc)
        mlflow.log_metric(f"ap_{split}", ap)

        if split == "test":
            print(f"\n=== {split.upper()} ===")
            print("ROC-AUC:", roc, "AP:", ap)
            print(classification_report(ys, yhat, digits=4))

    mlflow.log_params(params)

    model_path = MODEL_DIR / "logreg_creditcard.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(str(model_path))


def main():
    X, y = load_data()

    # stratified split to preserve imbalance
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    params = dict(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",  # critical for imbalance
        n_jobs=-1,
        random_state=42
    )
    clf = RandomForestClassifier(**params)
    with mlflow.start_run(run_name="rf_baseline_creditcard"):
        clf.fit(X_train, y_train)

        # Evaluate on val + test (ranking metrics better for imbalance)
        for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
            prob = clf.predict_proba(Xs)[:,1]
            yhat = (prob >= 0.5).astype(int)

            roc = roc_auc_score(ys, prob)
            ap  = average_precision_score(ys, prob)  # PR-AUC (AP)
            mlflow.log_metric(f"roc_auc_{split}", roc)
            mlflow.log_metric(f"ap_{split}", ap)

            if split == "test":
                print(f"\n=== {split.upper()} ===")
                print("ROC-AUC:", roc, "AP:", ap)
                print(classification_report(ys, yhat, digits=4))

        mlflow.log_params(params)
        # save model artifact
        model_path = MODEL_DIR / "rf_creditcard.pkl"
        joblib.dump(clf, model_path)
        mlflow.log_artifact(str(model_path))

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_feature_columns(data: pd.DataFrame) -> list:
    """Use project feature columns if present, otherwise infer numeric predictors."""
    preferred = [
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "is_weekend",
        "is_morning",
        "is_afternoon",
        "is_evening",
        "is_night",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "month_sin",
        "month_cos",
        "Latitude",
        "Longitude",
        "lat_grid",
        "lon_grid",
        "crime_density",
        "distance_to_center",
        "historical_crime_count",
    ]
    available = [c for c in preferred if c in data.columns]
    if available:
        return available

    excluded = {"is_high_risk", "risk_score", "Type", "Date", "Time", "Datetime"}
    numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in excluded]


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        # For models without predict_proba; SVC has it enabled here, so this is a safe fallback.
        y_score = model.decision_function(x_test)

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return result


def main():
    data_path = "./data/ml_enhanced_crime_data.csv"
    data = pd.read_csv(data_path)

    if "is_high_risk" not in data.columns:
        raise ValueError("Target column 'is_high_risk' not found in dataset")

    features = get_feature_columns(data)
    x = data[features]
    y = data["is_high_risk"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "SVM_RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results.append(evaluate_model(name, model, x_train, x_test, y_train, y_test))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)

    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))

    print("\n=== Confusion Matrices ===")
    for _, row in results_df.iterrows():
        print(f"\n{row['model']}: {row['confusion_matrix']}")

    results_df.to_csv("./models/simple_model_comparison_results.csv", index=False)
    print("\nSaved: ./models/simple_model_comparison_results.csv")


if __name__ == "__main__":
    main()

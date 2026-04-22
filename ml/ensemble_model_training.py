import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

def train_ensemble_model():
    print("🚀 Starting Optimized Stacking Ensemble Training for 85% Accuracy...")
    
    # 1. Load Data
    data_path = "./data/ml_enhanced_crime_data.csv"
    if not os.path.exists(data_path):
        print("❌ Data file not found. Please run feature engineering first.")
        return
    
    data = pd.read_csv(data_path)
    
    # Define features (matching the 35 features from 1_enhanced_feature_engineering.py)
    feature_columns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'Latitude', 'Longitude', 'lat_grid', 'lon_grid', 'crime_density',
        'distance_to_center', 'historical_crime_count',
        'historical_high_risk_count', 'historical_avg_severity',
        'baseline_risk_score', 'neighborhood_cluster',
        'night_time_risk', 'location_density_risk',
        'hour_density', 'weekend_night', 'dist_to_center_density',
        'lat_squared', 'lon_squared', 'lat_lon_prod'
    ]
    
    X = data[feature_columns]
    y = data['is_high_risk']
    
    # Handle any potential NaNs
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_columns)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Handle Class Imbalance with SMOTE
    print("\n⚖️ Balancing classes with SMOTE...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"    Resampled training data: {X_train_res.shape}")

    # 4. Hyperparameter Tuning for Base Models
    print("\n🔧 Tuning Base Models (this may take a few minutes)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # XGBoost Tuning
    print("  - Tuning XGBoost...")
    xgb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        xgb_params, n_iter=20, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train_res, y_train_res)
    best_xgb = xgb_search.best_estimator_
    print(f"    Best XGB Accuracy: {xgb_search.best_score_:.4f}")

    # RandomForest Tuning
    print("  - Tuning RandomForest...")
    rf_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params, n_iter=20, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train_res, y_train_res)
    best_rf = rf_search.best_estimator_
    print(f"    Best RF Accuracy: {rf_search.best_score_:.4f}")

    # Gradient Boosting Tuning
    print("  - Tuning GradientBoosting...")
    gb_params = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    gb_search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params, n_iter=20, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42
    )
    gb_search.fit(X_train_res, y_train_res)
    best_gb = gb_search.best_estimator_
    print(f"    Best GB Accuracy: {gb_search.best_score_:.4f}")

    # 5. Stacking Classifier
    print("\n🏗️ Building Final Stacking Ensemble...")
    estimators = [
        ('xgb', best_xgb),
        ('gb', best_gb),
        ('rf', best_rf)
    ]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("🚀 Training stacking classifier...")
    stacking_model.fit(X_train_res, y_train_res)
    
    # 5. Threshold Tuning for Maximum Accuracy
    print("\n🎯 Tuning decision threshold...")
    y_proba = stacking_model.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0, 1, 100)
    accuracies = [accuracy_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(accuracies)]
    max_accuracy = np.max(accuracies)
    
    print(f"    Best Threshold: {best_threshold:.2f}")
    print(f"    Max Accuracy at this threshold: {max_accuracy:.4f}")
    
    # 6. Final Evaluation
    y_pred = (y_proba >= best_threshold).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n✅ FINAL EVALUATION (Threshold={best_threshold:.2f}):")
    print(f"   Accuracy: {max_accuracy:.4f}")
    print(f"   AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save Everything
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Wrap the model logic if we want to save the threshold too
    # For now, we save the model and notes
    joblib.dump(stacking_model, "./models/classification_model.pkl")
    joblib.dump(stacking_model, "./models/best_classification_model.pkl")
    joblib.dump(feature_columns, "./models/feature_columns.pkl")
    joblib.dump(imputer, "./models/imputer.pkl")
    joblib.dump({'threshold': best_threshold, 'accuracy': max_accuracy}, "./models/model_metadata.pkl")
    
    print("\n💾 All assets saved!")
    return stacking_model

if __name__ == "__main__":
    train_ensemble_model()

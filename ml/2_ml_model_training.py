import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def train_crime_classification_models():
   
    
    print("🚀 Training ML Models for Crime Classification...")
    
    
    data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
    
   
    feature_columns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'Latitude', 'Longitude', 'lat_grid', 'lon_grid', 'crime_density',
        'distance_to_center'
    ]
    
    X = data[feature_columns]
    y_classification = data['is_high_risk']
    y_regression = data['risk_score']
    
    
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"🎯 High risk in training: {y_train_cls.mean():.2%}")
    
   
    gb_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
    )
    
    
    print(f"\n🔧 Training GradientBoosting Model...")
    gb_model.fit(X_train, y_train_cls)
    y_pred = gb_model.predict(X_test)
    y_proba = gb_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test_cls, y_pred)
    auc_score = roc_auc_score(y_test_cls, y_proba)
    cv_scores = cross_val_score(gb_model, X_train, y_train_cls, cv=5, scoring='roc_auc')
    
    results = {
        'GradientBoosting': {
            'accuracy': accuracy,
            'auc': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': gb_model
        }
    }
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ AUC: {auc_score:.4f}")
    print(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    best_model = gb_model
    best_model_name = "GradientBoosting"
    best_score = auc_score
    
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 10 Important Features:")
        print(feature_importance.head(10))
        
        
        feature_importance.to_csv("./models/feature_importance.csv", index=False)
    
    
    joblib.dump(best_model, "./models/best_classification_model.pkl")
    joblib.dump(best_model, "./models/classification_model.pkl")
    joblib.dump(scaler, "./models/feature_scaler.pkl")
    
    
    joblib.dump(feature_columns, "./models/feature_columns.pkl")
    
    print("\n💾 Models saved successfully!")
    
    return results, best_model, feature_columns

def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 {model_name} Performance Report:")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    results, gb_model, features = train_crime_classification_models()
    
    
    data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
    X = data[features]
    y = data['is_high_risk']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    evaluate_model_performance(gb_model, X_test, y_test, "GradientBoosting Model")
    
    print(f"\n🎉 ML Training Complete! GradientBoosting AUC: {results['GradientBoosting']['auc']:.4f}")
  
    
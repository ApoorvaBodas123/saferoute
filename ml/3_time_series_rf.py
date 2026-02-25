import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def prepare_time_series_data():
    
    
    print("🔄 Preparing time series data...")
    
  
    data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.sort_values('Datetime')
    
    
    temp_data = data.copy()
    temp_data['date'] = temp_data['Datetime'].dt.date
    temp_data['hour'] = temp_data['Datetime'].dt.hour
    
    hourly_crime = temp_data.groupby(['date', 'hour']).agg({
        'is_high_risk': ['sum', 'count'],
        'risk_score': 'mean',
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    
    hourly_crime.columns = ['date', 'hour', 'high_risk_count', 'total_crimes', 'avg_risk_score', 'avg_lat', 'avg_lon']
    
   
    hourly_crime['datetime'] = pd.to_datetime(hourly_crime['date'].astype(str) + ' ' + hourly_crime['hour'].astype(str) + ':00:00')
    hourly_crime = hourly_crime.set_index('datetime')
    
   
    full_timeline = pd.date_range(
        start=hourly_crime.index.min(),
        end=hourly_crime.index.max(),
        freq='H'
    )
    
   
    hourly_crime = hourly_crime.reindex(full_timeline)
    hourly_crime = hourly_crime.ffill().fillna(0)
    
   
    hourly_crime['hour'] = hourly_crime.index.hour
    hourly_crime['day_of_week'] = hourly_crime.index.dayofweek
    hourly_crime['month'] = hourly_crime.index.month
    hourly_crime['is_weekend'] = (hourly_crime['day_of_week'] >= 5).astype(int)
    
   
    hourly_crime['hour_sin'] = np.sin(2 * np.pi * hourly_crime['hour'] / 24)
    hourly_crime['hour_cos'] = np.cos(2 * np.pi * hourly_crime['hour'] / 24)
    hourly_crime['day_sin'] = np.sin(2 * np.pi * hourly_crime['day_of_week'] / 7)
    hourly_crime['day_cos'] = np.cos(2 * np.pi * hourly_crime['day_of_week'] / 7)
    
    
    for lag in [1, 2, 3, 6, 12, 24]:
        hourly_crime[f'high_risk_lag_{lag}'] = hourly_crime['high_risk_count'].shift(lag)
        hourly_crime[f'total_crime_lag_{lag}'] = hourly_crime['total_crimes'].shift(lag)
    
    
    for window in [3, 6, 12]:
        hourly_crime[f'high_risk_rolling_{window}'] = hourly_crime['high_risk_count'].rolling(window=window).mean()
        hourly_crime[f'risk_score_rolling_{window}'] = hourly_crime['avg_risk_score'].rolling(window=window).mean()
    
    
    hourly_crime = hourly_crime.dropna()
    
    print(f"✅ Time series data prepared: {len(hourly_crime)} hours")
    return hourly_crime

def train_time_series_model():

    
    print("🚀 Training Time Series Model...")
    
  
    ts_data = prepare_time_series_data()
    
    
    feature_columns = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'high_risk_lag_1', 'high_risk_lag_2', 'high_risk_lag_3', 'high_risk_lag_6', 'high_risk_lag_12', 'high_risk_lag_24',
        'total_crime_lag_1', 'total_crime_lag_2', 'total_crime_lag_3', 'total_crime_lag_6', 'total_crime_lag_12', 'total_crime_lag_24',
        'high_risk_rolling_3', 'high_risk_rolling_6', 'high_risk_rolling_12',
        'risk_score_rolling_3', 'risk_score_rolling_6', 'risk_score_rolling_12'
    ]
    
    X = ts_data[feature_columns]
    y = ts_data['high_risk_count']
    
   
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"📊 Training data shape: {X_train.shape}")
    print(f"📊 Test data shape: {X_test.shape}")
    
   
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n📊 Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 10 Important Features:")
    print(feature_importance.head(10))
    
   
    joblib.dump(model, "./models/time_series_rf_model.pkl")
    joblib.dump(feature_columns, "./models/time_series_features.pkl")
    feature_importance.to_csv("./models/time_series_feature_importance.csv", index=False)
    
    print("💾 Time series model saved successfully!")
    
    return model, feature_columns, ts_data

def predict_future_crime(model, last_data, hours_ahead=24):
    
    
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(hours_ahead):
        
        feature_columns = joblib.load("./models/time_series_features.pkl")
        X_pred = current_data[feature_columns].iloc[-1:].values
        
       
        pred = model.predict(X_pred)[0]
        predictions.append(max(0, pred))  
        
       
        current_data.loc[current_data.index[-1], 'high_risk_count'] = pred
    
    return np.array(predictions)

def create_crime_trend_analysis():
    
    
    print("📈 Analyzing crime trends...")
    
    data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
   
    hourly_avg = data.groupby(data['Datetime'].dt.hour)['risk_score'].mean()
    
    
    daily_avg = data.groupby(data['Datetime'].dt.dayofweek)['risk_score'].mean()
    
    monthly_avg = data.groupby(data['Datetime'].dt.month)['risk_score'].mean()
    
    
    risk_level_counts = data['is_high_risk'].value_counts()
    
    
    trend_data = {
        'hourly_pattern': hourly_avg.to_dict(),
        'daily_pattern': daily_avg.to_dict(),
        'monthly_pattern': monthly_avg.to_dict(),
        'risk_distribution': risk_level_counts.to_dict(),
        'high_risk_percentage': (data['is_high_risk'].mean() * 100)
    }
    
    joblib.dump(trend_data, "./models/crime_trend_analysis.pkl")
    
    print("✅ Crime trend analysis completed!")
    print(f"   High risk crimes: {trend_data['high_risk_percentage']:.2f}%")
    
    return trend_data

if __name__ == "__main__":
  
    ts_model, ts_features, ts_data = train_time_series_model()
    

    trend_analysis = create_crime_trend_analysis()
    
    print("\n🎉 Time Series Analysis Complete!")
    print("📊 Models saved: time_series_rf_model.pkl")
    print("📈 Trend analysis saved: crime_trend_analysis.pkl")

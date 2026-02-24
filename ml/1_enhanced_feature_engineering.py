import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def enhanced_feature_engineering():
    """Advanced feature engineering for ML models"""
    
    # Load data
    print("🔄 Loading and preprocessing data...")
    data = pd.read_csv("./data/SouthCrimeDetails.csv")
    
    # Basic cleaning
    data = data[['Type', 'Date', 'Time', 'Latitude', 'Longitude']]
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    # Convert coordinates to numeric
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
    data.dropna(inplace=True)
    
    # Enhanced datetime features
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')
    data = data.dropna(subset=['Datetime'])
    
    # Temporal features
    data['hour'] = data['Datetime'].dt.hour
    data['day_of_week'] = data['Datetime'].dt.dayofweek
    data['day_of_month'] = data['Datetime'].dt.day
    data['month'] = data['Datetime'].dt.month
    data['quarter'] = data['Datetime'].dt.quarter
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Time-based features
    data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] < 12)).astype(int)
    data['is_afternoon'] = ((data['hour'] >= 12) & (data['hour'] < 18)).astype(int)
    data['is_evening'] = ((data['hour'] >= 18) & (data['hour'] < 22)).astype(int)
    data['is_night'] = ((data['hour'] >= 22) | (data['hour'] < 6)).astype(int)
    
    # Cyclical encoding for temporal features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Spatial features - create grid and calculate density
    data['lat_grid'] = (data['Latitude'] // 0.01) * 0.01
    data['lon_grid'] = (data['Longitude'] // 0.01) * 0.01
    
    # Calculate crime density per grid cell
    grid_density = data.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='crime_density')
    data = data.merge(grid_density, on=['lat_grid', 'lon_grid'], how='left')
    
    # Distance to city center (Bangalore center: 12.9716, 77.5946)
    from math import radians, cos, sin, asin, sqrt
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
    
    data['distance_to_center'] = data.apply(
        lambda row: haversine_distance(row['Latitude'], row['Longitude'], 12.9716, 77.5946), 
        axis=1
    )
    
    # Crime type encoding
    le = LabelEncoder()
    data['crime_type_encoded'] = le.fit_transform(data['Type'])
    
    # Enhanced crime severity mapping
    crime_severity_map = {
        'MURDER': 10, 'RAPE': 9, 'KIDNAPPING': 8, 'ROBBERY': 7,
        'ASSAULT': 6, 'BURGLARY': 5, 'THEFT': 4, 'CHEATING': 3,
        'OTHERS': 2, 'CYBER CRIME': 1
    }
    data['crime_severity'] = data['Type'].str.upper().map(crime_severity_map).fillna(2)
    
    # Time-weighted severity
    time_weights = {
        'is_night': 1.5, 'is_evening': 1.2, 'is_afternoon': 1.0, 'is_morning': 0.8
    }
    for time_col, weight in time_weights.items():
        data.loc[data[time_col] == 1, 'time_weighted_severity'] = data.loc[data[time_col] == 1, 'crime_severity'] * weight
    
    data['time_weighted_severity'] = data['time_weighted_severity'].fillna(data['crime_severity'])
    
    # Historical crime count in area (rolling window)
    data_sorted = data.sort_values('Datetime')
    data_sorted['historical_crime_count'] = data_sorted.groupby(['lat_grid', 'lon_grid']).cumcount()
    
    # Target variable for classification (high risk vs low risk)
    data['is_high_risk'] = (data['time_weighted_severity'] >= 6).astype(int)
    
    # Target variable for regression (risk score)
    data['risk_score'] = data['time_weighted_severity'] * np.log1p(data['crime_density'])
    
    print("✅ Feature engineering completed!")
    print(f"📊 Dataset shape: {data.shape}")
    print(f"🎯 High risk crimes: {data['is_high_risk'].sum()} ({data['is_high_risk'].mean():.2%})")
    
    # Save processed data and encoders
    data.to_csv("./data/ml_enhanced_crime_data.csv", index=False)
    joblib.dump(le, "./models/crime_type_encoder.pkl")
    
    # Feature list for modeling
    feature_columns = [
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
        'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'Latitude', 'Longitude', 'lat_grid', 'lon_grid', 'crime_density',
        'distance_to_center', 'historical_crime_count'
    ]
    
    return data, feature_columns

if __name__ == "__main__":
    data, features = enhanced_feature_engineering()
    print(f"\n🔧 Features ready for ML: {len(features)}")
    print(f"📋 Feature list: {features}")

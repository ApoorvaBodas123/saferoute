import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')
def enhanced_feature_engineering():
    print("🔄 Loading and preprocessing data...")
    # 1. Load Data
    data = pd.read_csv("./data/SouthCrimeDetails.csv")
    
    # 2. Preprocessing
    # Keep essential columns and handle missing values
    data = data[['Type', 'Date', 'Time', 'Latitude', 'Longitude']]
    data.dropna(subset=['Date', 'Time', 'Latitude', 'Longitude'], inplace=True)
    data.drop_duplicates(inplace=True)
    
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
    data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    
    # Robust date parsing to handle mixed formats (DD/MM/YYYY, MM/DD/YYYY, DD/MM/YY)
    print("📅 Parsing dates (handling mixed formats)...")
    def robust_date_parse(date_str):
        date_str = str(date_str).strip()
        for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d/%m/%y', '%m/%d/%y', '%Y-%m-%d']:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return pd.NaT

    data['ParsedDate'] = data['Date'].apply(robust_date_parse)
    
    # Robust time parsing to handle 24h time with redundant AM/PM (e.g. 14:30PM)
    def robust_time_parse(time_str):
        time_str = str(time_str).strip().upper()
        # Try parsing as 24h by stripping AM/PM first
        clean_time = time_str.replace('AM', '').replace('PM', '').strip()
        for fmt in ['%H:%M', '%H.%M', '%H:%M:%S']:
            try:
                return datetime.strptime(clean_time, fmt).time()
            except:
                continue
        # If that fails, try with AM/PM properly for 12h formats
        for fmt in ['%I:%M %p', '%I:%M%p', '%I.%M %p', '%I.%M%p']:
            try:
                return datetime.strptime(time_str, fmt).time()
            except:
                continue
        return pd.NaT

    data['ParsedTime'] = data['Time'].apply(robust_time_parse)
    
    # Combine Date and Time
    def create_datetime(row):
        try:
            if pd.notnull(row['ParsedDate']) and pd.notnull(row['ParsedTime']):
                dt = datetime.combine(row['ParsedDate'], row['ParsedTime'])
                # Only keep reasonable years (2010-2026)
                if 2010 <= dt.year <= 2026:
                    return dt
        except:
            pass
        return pd.NaT

    data['Datetime'] = data.apply(create_datetime, axis=1)
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    
    initial_count = len(data)
    data = data.dropna(subset=['Datetime'])
    dropped_count = initial_count - len(data)
    print(f"✅ Date parsing complete. Retained {len(data)} records (Dropped {dropped_count} due to invalid format)")
    
    data['hour'] = data['Datetime'].dt.hour
    data['day_of_week'] = data['Datetime'].dt.dayofweek
    data['day_of_month'] = data['Datetime'].dt.day
    data['month'] = data['Datetime'].dt.month
    data['quarter'] = data['Datetime'].dt.quarter
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
   
    data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] < 12)).astype(int)
    data['is_afternoon'] = ((data['hour'] >= 12) & (data['hour'] < 18)).astype(int)
    data['is_evening'] = ((data['hour'] >= 18) & (data['hour'] < 22)).astype(int)
    data['is_night'] = ((data['hour'] >= 22) | (data['hour'] < 6)).astype(int)
    
   
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    
    data['lat_grid'] = (data['Latitude'] // 0.01) * 0.01
    data['lon_grid'] = (data['Longitude'] // 0.01) * 0.01
    
   
    grid_density = data.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='crime_density')
    data = data.merge(grid_density, on=['lat_grid', 'lon_grid'], how='left')
    
   
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
     
    le = LabelEncoder()
    data['crime_type_encoded'] = le.fit_transform(data['Type'])
       
    crime_severity_map = {
        'MURDER': 10, 'RAPE': 9, 'KIDNAPPING': 8, 'ROBBERY': 7,
        'ASSAULT': 6, 'BURGLARY': 5, 'THEFT': 4, 'CHEATING': 3,
        'OTHERS': 2, 'CYBER CRIME': 1
    }
    data['crime_severity'] = data['Type'].str.upper().map(crime_severity_map).fillna(2)
    
    
    time_weights = {
        'is_night': 1.5, 'is_evening': 1.2, 'is_afternoon': 1.0, 'is_morning': 0.8
    }
    for time_col, weight in time_weights.items():
        data.loc[data[time_col] == 1, 'time_weighted_severity'] = data.loc[data[time_col] == 1, 'crime_severity'] * weight
    
    data['time_weighted_severity'] = data['time_weighted_severity'].fillna(data['crime_severity'])
    
    
    data = data.sort_values('Datetime')
    data['historical_crime_count'] = data.groupby(['lat_grid', 'lon_grid']).cumcount()
    
    # Add historical high risk count and severity
    # We need to shift by 1 to avoid including the current event's risk in its own historical features
    data['is_high_risk'] = (data['time_weighted_severity'] >= 6).astype(int)
    data['historical_high_risk_count'] = data.groupby(['lat_grid', 'lon_grid'])['is_high_risk'].transform(lambda x: x.shift(1).fillna(0).cumsum())
    data['historical_avg_severity'] = data.groupby(['lat_grid', 'lon_grid'])['crime_severity'].transform(lambda x: x.shift(1).fillna(x.mean()).expanding().mean())
    
    data['risk_score'] = data['time_weighted_severity'] * np.log1p(data['crime_density'])
    
    # 11. Add Static Baseline Risk (from bangalore_risk_zones.json)
    try:
        if os.path.exists("./data/bangalore_risk_zones.json"):
            with open("./data/bangalore_risk_zones.json", 'r') as f:
                zones_data = json.load(f)
            zones_df = pd.DataFrame(zones_data)
            # Round to 0.01 to match JSON precision
            data['lat_grid_01'] = (data['Latitude'] // 0.01) * 0.01
            data['lon_grid_01'] = (data['Longitude'] // 0.01) * 0.01
            
            data = data.merge(
                zones_df[['lat_grid', 'lon_grid', 'risk_score']], 
                left_on=['lat_grid_01', 'lon_grid_01'], 
                right_on=['lat_grid', 'lon_grid'], 
                how='left',
                suffixes=('', '_baseline')
            )
            data['baseline_risk_score'] = data['risk_score_baseline'].fillna(data['risk_score_baseline'].mean())
            # Cleanup temp join columns
            data.drop(['lat_grid_01', 'lon_grid_01', 'lat_grid_baseline', 'lon_grid_baseline', 'risk_score_baseline'], axis=1, inplace=True, errors='ignore')
            print("✅ Integrated static risk zones baseline")
    except Exception as e:
        print(f"⚠️ Could not integrate baseline: {e}")
        data['baseline_risk_score'] = 1.0 # Fallback
        
    # 12. Spatial Clustering (Neighborhood Hotspots)
    print("📍 Clustering locations into hotspots...")
    coords = data[['Latitude', 'Longitude']]
    n_clusters = 25
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['neighborhood_cluster'] = kmeans.fit_transform(coords).argmin(axis=1)
    joblib.dump(kmeans, "./models/spatial_kmeans.pkl")
    
    # 13. Interaction Features
    data['night_time_risk'] = data['is_night'] * data['historical_avg_severity']
    data['location_density_risk'] = data['crime_density'] * data['historical_high_risk_count']
    
    # New High-Impact Interactions
    data['hour_density'] = data['hour'] * data['crime_density']
    data['weekend_night'] = data['is_weekend'] * data['is_night']
    data['dist_to_center_density'] = data['distance_to_center'] * data['crime_density']
    
    # 14. Spatial Gradient Features (Polynomial)
    data['lat_squared'] = data['Latitude'] ** 2
    data['lon_squared'] = data['Longitude'] ** 2
    data['lat_lon_prod'] = data['Latitude'] * data['Longitude']
    
    print("✅ Feature engineering completed!")
    print(f"📊 Dataset shape: {data.shape}")
    print(f"🎯 High risk crimes: {data['is_high_risk'].sum()} ({data['is_high_risk'].mean():.2%})")
    
    
    data.to_csv("./data/ml_enhanced_crime_data.csv", index=False)
    joblib.dump(le, "./models/crime_type_encoder.pkl")
    
   
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
    
    return data, feature_columns

if __name__ == "__main__":
    data, features = enhanced_feature_engineering()
    print(f"\n🔧 Features ready for ML: {len(features)}")
    print(f"📋 Feature list: {features}")
 

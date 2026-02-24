import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DynamicRiskScorer:
    """Dynamic risk scoring system with online learning capabilities"""
    
    def __init__(self):
        self.load_models()
        self.risk_history = []
        self.model_performance_log = []
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            self.classification_model = joblib.load("./models/ensemble_model.pkl")
            self.time_series_model = joblib.load("./models/time_series_rf_model.pkl")
            self.scaler = joblib.load("./models/feature_scaler.pkl")
            self.feature_columns = joblib.load("./models/feature_columns.pkl")
            self.ts_features = joblib.load("./models/time_series_features.pkl")
            self.trend_analysis = joblib.load("./models/crime_trend_analysis.pkl")
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            
    def calculate_dynamic_risk_score(self, latitude, longitude, prediction_time=None):
        """Calculate dynamic risk score for a specific location and time"""
        
        if prediction_time is None:
            prediction_time = datetime.now()
            
        # Extract temporal features
        hour = prediction_time.hour
        day_of_week = prediction_time.weekday()
        day_of_month = prediction_time.day
        month = prediction_time.month
        quarter = (month - 1) // 3 + 1
        
        # Create feature dictionary
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'quarter': quarter,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_morning': 1 if 6 <= hour < 12 else 0,
            'is_afternoon': 1 if 12 <= hour < 18 else 0,
            'is_evening': 1 if 18 <= hour < 22 else 0,
            'is_night': 1 if hour >= 22 or hour < 6 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'month_sin': np.sin(2 * np.pi * month / 12),
            'month_cos': np.cos(2 * np.pi * month / 12),
            'Latitude': latitude,
            'Longitude': longitude,
            'lat_grid': (latitude // 0.01) * 0.01,
            'lon_grid': (longitude // 0.01) * 0.01,
            'distance_to_center': self._calculate_distance_to_center(latitude, longitude),
            'crime_density': self._estimate_crime_density(latitude, longitude)
        }
        
        # Create feature array
        feature_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Make prediction
        risk_probability = self.classification_model.predict_proba(feature_array)[0, 1]
        
        # Apply temporal adjustments based on trend analysis
        temporal_multiplier = self._get_temporal_multiplier(hour, day_of_week, month)
        
        # Calculate final risk score
        base_risk_score = risk_probability * 10  # Scale to 0-10
        final_risk_score = base_risk_score * temporal_multiplier
        
        # Add time series prediction for future risk
        future_risk = self._predict_future_risk_trend(prediction_time)
        
        return {
            'current_risk_score': min(10, final_risk_score),
            'risk_probability': risk_probability,
            'temporal_multiplier': temporal_multiplier,
            'future_risk_trend': future_risk,
            'risk_level': self._categorize_risk(final_risk_score),
            'features_used': features
        }
    
    def _calculate_distance_to_center(self, lat, lon):
        """Calculate distance to Bangalore center"""
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
        
        return haversine_distance(lat, lon, 12.9716, 77.5946)
    
    def _estimate_crime_density(self, lat, lon):
        """Estimate crime density for a location"""
        # Load historical data for density estimation
        try:
            data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
            
            # Find nearby crimes within 1km
            nearby_crimes = data[
                (np.abs(data['Latitude'] - lat) < 0.01) & 
                (np.abs(data['Longitude'] - lon) < 0.01)
            ]
            
            if len(nearby_crimes) > 0:
                return len(nearby_crimes) / 10  # Normalize
            else:
                return 1.0  # Default low density
        except:
            return 1.0
    
    def _get_temporal_multiplier(self, hour, day_of_week, month):
        """Get temporal risk multiplier based on historical patterns"""
        
        # Base multipliers from trend analysis
        hourly_pattern = self.trend_analysis['hourly_pattern']
        daily_pattern = self.trend_analysis['daily_pattern']
        monthly_pattern = self.trend_analysis['monthly_pattern']
        
        # Get average risk for each time period
        avg_hourly_risk = np.mean(list(hourly_pattern.values()))
        avg_daily_risk = np.mean(list(daily_pattern.values()))
        avg_monthly_risk = np.mean(list(monthly_pattern.values()))
        
        # Calculate multipliers
        hour_multiplier = hourly_pattern.get(hour, avg_hourly_risk) / avg_hourly_risk
        day_multiplier = daily_pattern.get(day_of_week, avg_daily_risk) / avg_daily_risk
        month_multiplier = monthly_pattern.get(month, avg_monthly_risk) / avg_monthly_risk
        
        # Combine multipliers
        combined_multiplier = (hour_multiplier + day_multiplier + month_multiplier) / 3
        
        return max(0.5, min(2.0, combined_multiplier))  # Clamp between 0.5 and 2.0
    
    def _predict_future_risk_trend(self, current_time):
        """Predict risk trend for next 24 hours"""
        try:
            # This is a simplified version - in practice, you'd use the time series model
            hours = list(range(24))
            trend_scores = []
            
            for hour_ahead in range(1, 25):
                future_time = current_time + timedelta(hours=hour_ahead)
                hour = future_time.hour
                
                # Use historical patterns to predict trend
                hourly_pattern = self.trend_analysis['hourly_pattern']
                avg_risk = np.mean(list(hourly_pattern.values()))
                hour_risk = hourly_pattern.get(hour, avg_risk)
                
                trend_scores.append(hour_risk / avg_risk)
            
            return {
                'next_24_hours': trend_scores,
                'peak_risk_hour': hours[np.argmax(trend_scores)],
                'lowest_risk_hour': hours[np.argmin(trend_scores)],
                'trend_direction': 'increasing' if trend_scores[12] > trend_scores[0] else 'decreasing'
            }
        except:
            return {'next_24_hours': [1.0] * 24, 'peak_risk_hour': 0, 'lowest_risk_hour': 0, 'trend_direction': 'stable'}
    
    def _categorize_risk(self, risk_score):
        """Categorize risk level"""
        if risk_score <= 3:
            return 'Low'
        elif risk_score <= 6:
            return 'Medium'
        elif risk_score <= 8:
            return 'High'
        else:
            return 'Very High'
    
    def update_with_new_data(self, new_crime_data):
        """Update models with new crime data (online learning)"""
        
        print("🔄 Updating models with new data...")
        
        try:
            # Process new data
            new_data = self._process_new_crime_data(new_crime_data)
            
            if len(new_data) > 0:
                # Update classification model with partial fit
                X_new = new_data[self.feature_columns]
                y_new = new_data['is_high_risk']
                
                # For models that support partial_fit
                if hasattr(self.classification_model, 'partial_fit'):
                    self.classification_model.partial_fit(X_new, y_new)
                    print("✅ Classification model updated")
                
                # Log the update
                self.risk_history.append({
                    'timestamp': datetime.now(),
                    'new_records': len(new_data),
                    'avg_risk_score': new_data['risk_score'].mean()
                })
                
                # Save updated model
                joblib.dump(self.classification_model, "./models/ensemble_model.pkl")
                
                return True
            else:
                print("⚠️ No new data to process")
                return False
                
        except Exception as e:
            print(f"❌ Error updating models: {e}")
            return False
    
    def _process_new_crime_data(self, new_data):
        """Process new crime data for model updates"""
        # This would implement the same feature engineering as in step 1
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame()
    
    def get_risk_heatmap_data(self, bounds=None):
        """Generate risk heatmap data for visualization"""
        
        print("🔥 Generating risk heatmap...")
        
        # Generate grid points
        if bounds is None:
            # Default Bangalore bounds
            lat_min, lat_max = 12.8, 13.1
            lon_min, lon_max = 77.4, 77.8
        else:
            lat_min, lat_max, lon_min, lon_max = bounds
        
        # Create grid
        grid_points = []
        risk_scores = []
        
        lat_step = 0.01  # ~1km resolution
        lon_step = 0.01
        
        for lat in np.arange(lat_min, lat_max, lat_step):
            for lon in np.arange(lon_min, lon_max, lon_step):
                # Calculate risk for this point
                risk_result = self.calculate_dynamic_risk_score(lat, lon)
                
                grid_points.append([lat, lon])
                risk_scores.append(risk_result['current_risk_score'])
        
        heatmap_data = {
            'coordinates': grid_points,
            'risk_scores': risk_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        return heatmap_data
    
    def evaluate_model_performance(self):
        """Evaluate current model performance"""
        
        if len(self.risk_history) < 2:
            return "Insufficient data for evaluation"
        
        # Calculate performance metrics
        recent_updates = self.risk_history[-10:]  # Last 10 updates
        
        avg_new_records = np.mean([update['new_records'] for update in recent_updates])
        avg_risk_trend = np.mean([update['avg_risk_score'] for update in recent_updates])
        
        performance_report = {
            'total_updates': len(self.risk_history),
            'recent_avg_new_records': avg_new_records,
            'recent_avg_risk_score': avg_risk_trend,
            'last_update': self.risk_history[-1]['timestamp'],
            'model_status': 'Active' if len(self.risk_history) > 0 else 'No updates'
        }
        
        return performance_report

def initialize_dynamic_risk_system():
    """Initialize the dynamic risk scoring system"""
    
    print("🚀 Initializing Dynamic Risk Scoring System...")
    
    risk_scorer = DynamicRiskScorer()
    
    # Test the system with a sample location
    test_result = risk_scorer.calculate_dynamic_risk_score(12.9716, 77.5946)
    
    print(f"✅ System initialized successfully!")
    print(f"📍 Test location (Bangalore Center): {test_result['risk_level']} Risk")
    print(f"📊 Risk Score: {test_result['current_risk_score']:.2f}")
    
    return risk_scorer

if __name__ == "__main__":
    # Initialize and test the dynamic risk system
    risk_system = initialize_dynamic_risk_system()
    
    # Generate sample heatmap data
    heatmap_data = risk_system.get_risk_heatmap_data()
    print(f"🔥 Generated heatmap with {len(heatmap_data['coordinates'])} points")
    
    # Save system configuration
    joblib.dump(risk_system, "./models/dynamic_risk_system.pkl")
    
    print("\n🎉 Dynamic Risk Scoring System Complete!")
    print("💾 System saved: dynamic_risk_system.pkl")

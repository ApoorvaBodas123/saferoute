from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


try:
    risk_system = joblib.load("./models/dynamic_risk_system.pkl")
    route_optimizer = joblib.load("./models/ml_route_optimizer.pkl")
    classification_model = joblib.load("./models/ensemble_model.pkl")
    feature_columns = joblib.load("./models/feature_columns.pkl")
    trend_analysis = joblib.load("./models/crime_trend_analysis.pkl")
    print("✅ ML models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    risk_system = None
    route_optimizer = None

@app.route('/api/risk-score', methods=['POST'])
def get_risk_score():
    
    try:
        data = request.get_json()
        latitude = data['latitude']
        longitude = data['longitude']
        prediction_time = datetime.fromisoformat(data.get('prediction_time', datetime.now().isoformat()))
        
        if risk_system:
            result = risk_system.calculate_dynamic_risk_score(latitude, longitude, prediction_time)
        else:
            
            result = calculate_fallback_risk_score(latitude, longitude, prediction_time)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    
    try:
        data = request.get_json()
        start_lat = data['start_latitude']
        start_lon = data['start_longitude']
        end_lat = data['end_latitude']
        end_lon = data['end_longitude']
        strategy = data.get('strategy', 'balanced')
        
        if route_optimizer:
            result = route_optimizer.optimize_route(start_lat, start_lon, end_lat, end_lon, strategy)
        else:
            
            result = calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crime-trends', methods=['GET'])
def get_crime_trends():
   
    try:
        if trend_analysis:
            return jsonify(trend_analysis)
        else:
            
            return jsonify(get_fallback_trends())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route-update', methods=['POST'])
def get_route_update():
    
    try:
        data = request.get_json()
        route_coordinates = data['route_coordinates']
        current_position_index = data['current_position_index']
        
        if route_optimizer:
            result = route_optimizer.get_real_time_route_update(route_coordinates, current_position_index)
        else:
           
            result = get_fallback_route_update(route_coordinates, current_position_index)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
  
    return jsonify({
        'status': 'healthy',
        'ml_models_loaded': {
            'risk_system': risk_system is not None,
            'route_optimizer': route_optimizer is not None,
            'classification_model': classification_model is not None
        },
        'timestamp': datetime.now().isoformat()
    })


def calculate_fallback_risk_score(latitude, longitude, prediction_time):
    
    hour = prediction_time.hour
    is_night = hour >= 22 or hour < 6
    is_evening = hour >= 18 and hour < 22
    
    base_risk = 3.0
    
    if is_night:
        temporal_multiplier = 1.5
    elif is_evening:
        temporal_multiplier = 1.2
    else:
        temporal_multiplier = 1.0
    
    
    distance_to_center = haversine_distance(latitude, longitude, 12.9716, 77.5946)
    if distance_to_center < 5:
        spatial_multiplier = 1.3
    elif distance_to_center < 10:
        spatial_multiplier = 1.1
    else:
        spatial_multiplier = 1.0
    
    final_risk_score = base_risk * temporal_multiplier * spatial_multiplier
    final_risk_score = min(10, final_risk_score)
    
    if final_risk_score <= 3:
        risk_level = 'Low'
    elif final_risk_score <= 6:
        risk_level = 'Medium'
    elif final_risk_score <= 8:
        risk_level = 'High'
    else:
        risk_level = 'Very High'
    
    return {
        'current_risk_score': final_risk_score,
        'risk_probability': final_risk_score / 10,
        'risk_level': risk_level,
        'temporal_multiplier': temporal_multiplier,
        'is_fallback': True
    }

def calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy):
   
    route_coordinates = [[start_lat, start_lon], [end_lat, end_lon]]
    distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    estimated_time = (distance / 30) * 60  # minutes
    
    
    mid_lat = (start_lat + end_lat) / 2
    mid_lon = (start_lon + end_lon) / 2
    risk_result = calculate_fallback_risk_score(mid_lat, mid_lon, datetime.now())
    
    return {
        'route_coordinates': route_coordinates,
        'strategy': f'fallback_{strategy}',
        'distance_km': distance,
        'avg_risk_score': risk_result['current_risk_score'],
        'risk_level': risk_result['risk_level'],
        'estimated_time_minutes': estimated_time,
        'high_risk_segments': [],
        'is_fallback': True
    }

def get_fallback_trends():
   
    return {
        'hourly_pattern': {
            str(h): 1.0 + 0.5 * np.sin(2 * np.pi * h / 24) for h in range(24)
        },
        'daily_pattern': {
            str(d): 1.0 + 0.2 * np.sin(2 * np.pi * d / 7) for d in range(7)
        },
        'monthly_pattern': {
            str(m): 1.0 + 0.3 * np.sin(2 * np.pi * m / 12) for m in range(1, 13)
        },
        'is_fallback': True
    }

def get_fallback_route_update(route_coordinates, current_position_index):
    
    if current_position_index >= len(route_coordinates):
        return {'error': 'Invalid position index'}
    
    remaining_route = route_coordinates[current_position_index:]
    total_risk = 0
    
    for coord in remaining_route:
        risk_result = calculate_fallback_risk_score(coord[0], coord[1], datetime.now())
        total_risk += risk_result['current_risk_score']
    
    return {
        'remaining_segments': [
            {
                'coordinates': coord,
                'current_risk_score': calculate_fallback_risk_score(coord[0], coord[1], datetime.now())['current_risk_score'],
                'risk_level': calculate_fallback_risk_score(coord[0], coord[1], datetime.now())['risk_level']
            }
            for coord in remaining_route
        ],
        'avg_remaining_risk': total_risk / len(remaining_route),
        'segments_remaining': len(remaining_route),
        'update_timestamp': datetime.now().isoformat(),
        'is_fallback': True
    }

def haversine_distance(lat1, lon1, lat2, lon2):
    
    import math
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * math.asin(math.sqrt(a)) * 6371  # Earth radius in km

if __name__ == '__main__':
    print("🚀 Starting ML API Server...")
    print("📊 Available endpoints:")
    print("   POST /api/risk-score - Get dynamic risk score")
    print("   POST /api/optimize-route - Get optimized route")
    print("   GET /api/crime-trends - Get crime trends")
    print("   POST /api/route-update - Get route updates")
    print("   GET /api/health - Health check")
    
    app.run(host='0.0.0.0', port=8000, debug=True)

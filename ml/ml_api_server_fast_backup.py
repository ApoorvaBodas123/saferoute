from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime, timedelta
import hashlib
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)
cache = {}
CACHE_DURATION = 300  
def get_cache_key(data):
    data_str = str(sorted(data.items()))
    return hashlib.md5(data_str.encode()).hexdigest()
def get_from_cache(key):
    if key in cache:
        cached_data, timestamp = cache[key]
        if datetime.now() - timestamp < timedelta(seconds=CACHE_DURATION):
            return cached_data
        else:
            del cache[key]
    return None
def set_cache(key, data):
    cache[key] = (data, datetime.now())
    if len(cache) > 100:
        oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
        del cache[oldest_key]
print("🚀 Starting ML API Server (Fast Mode)...")
@app.route('/api/risk-score', methods=['POST'])
def get_risk_score():
    try:
        data = request.get_json()
        latitude = data['latitude']
        longitude = data['longitude']
        prediction_time = datetime.fromisoformat(data.get('prediction_time', datetime.now().isoformat()))
        result = calculate_fallback_risk_score(latitude, longitude, prediction_time)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    try:
        data = request.get_json()
        cache_key = get_cache_key(data)
        cached_result = get_from_cache(cache_key)
        if cached_result:
            return jsonify(cached_result)
        start_lat = data['start_latitude']
        start_lon = data['start_longitude']
        end_lat = data['end_latitude']
        end_lon = data['end_longitude']
        strategy = data.get('strategy', 'balanced')
        result = calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy)
        set_cache(cache_key, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/crime-trends', methods=['GET'])
def get_crime_trends():
    try:
        return jsonify(get_fallback_trends())
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/route-update', methods=['POST'])
def get_route_update():
    try:
        data = request.get_json()
        route_coordinates = data['route_coordinates']
        current_position_index = data['current_position_index']
        result = get_fallback_route_update(route_coordinates, current_position_index)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'mode': 'fallback',
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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from realistic_routes import get_realistic_route
def calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy):
    try:
        result = get_realistic_route(start_lat, start_lon, end_lat, end_lon, strategy)
        return {
            "route_coordinates": result["route_coordinates"],
            "total_distance": result["total_distance"],
            "estimated_duration": result["estimated_duration"],
            "total_risk_score": result["total_risk_score"],
            "risk_level": "Low" if result["total_risk_score"] < 3 else "Medium" if result["total_risk_score"] < 6 else "High",
            "strategy": strategy,
            "route_type": result["route_type"],
            "roads_used": result.get("roads_used", []),
            "success": True
        }
    except Exception as e:
        print(f"Error in realistic route calculation: {e}")
        return calculate_simple_route(start_lat, start_lon, end_lat, end_lon, strategy)
def calculate_simple_route(start_lat, start_lon, end_lat, end_lon, strategy):
    base_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    if strategy == "safest":
        route_coordinates = [
            [start_lat, start_lon],
            [start_lat + 0.01, start_lon],
            [start_lat + 0.01, start_lon + 0.015],
            [start_lat + 0.02, start_lon + 0.015],
            [(start_lat + end_lat) / 2 + 0.01, (start_lon + end_lon) / 2 + 0.005],
            [end_lat - 0.01, end_lon - 0.005],
            [end_lat - 0.005, end_lon],
            [end_lat, end_lon]
        ]
        distance = base_distance * 1.3
        risk_score = 2.5
        risk_level = "Low"
        estimated_time = (distance / 22) * 60
    elif strategy == "fastest":
        route_coordinates = [
            [start_lat, start_lon],
            [start_lat + 0.005, start_lon + 0.008],
            [(start_lat + end_lat) / 2, (start_lon + end_lon) / 2],
            [end_lat - 0.003, end_lon - 0.002],
            [end_lat, end_lon]
        ]
        distance = base_distance * 1.1
        risk_score = 6.5
        risk_level = "High"
        estimated_time = (distance / 30) * 60
    else:  
        route_coordinates = [
            [start_lat, start_lon],
            [start_lat + 0.008, start_lon + 0.01],
            [(start_lat + end_lat) / 2 + 0.002, (start_lon + end_lon) / 2 + 0.003],
            [end_lat - 0.007, end_lon - 0.004],
            [end_lat, end_lon]
        ]
        distance = base_distance * 1.2
        risk_score = 4.0
        risk_level = "Medium"
        estimated_time = (distance / 25) * 60
    return {
        "route_coordinates": route_coordinates,
        "total_distance": distance,
        "estimated_duration": estimated_time,
        "total_risk_score": risk_score,
            "risk_level": risk_level,
            "strategy": strategy,
            "route_type": "simple_fallback",
            "roads_used": [],
            "success": True
        }
def _identify_high_risk_segments(route_coordinates, base_risk):
    high_risk_segments = []
    for i in range(len(route_coordinates) - 1):
        segment_risk = base_risk + (i * 0.3)  
        if segment_risk > 6:
            high_risk_segments.append({
                'segment_index': i,
                'coordinates': route_coordinates[i],
                'risk_score': segment_risk
            })
    return high_risk_segments
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
    return 2 * math.asin(math.sqrt(a)) * 6371  
if __name__ == '__main__':
    print("📊 Available endpoints:")
    print("   POST /api/risk-score - Get dynamic risk score")
    print("   POST /api/optimize-route - Get optimized route")
    print("   GET /api/crime-trends - Get crime trends")
    print("   POST /api/route-update - Get route updates")
    print("   GET /api/health - Health check")
    print("🚀 Server starting on http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)

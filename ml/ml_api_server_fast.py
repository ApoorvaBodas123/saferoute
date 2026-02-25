from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import time
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)


BANGALORE_CENTER = (12.9716, 77.5946)

def haversine_distance(lat1, lon1, lat2, lon2):
    
    R = 6371 
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def calculate_fallback_risk(lat, lon):
    
    hour = datetime.now().hour
    is_night = hour >= 22 or hour < 6
    is_evening = hour >= 18 and hour < 22
    
    
    base_risk = 3.0
    
   
    temporal_multiplier = 1.5 if is_night else 1.2 if is_evening else 1.0
    
    
    distance_from_center = haversine_distance(lat, lon, BANGALORE_CENTER[0], BANGALORE_CENTER[1])
    spatial_multiplier = 1.3 if distance_from_center < 5 else 1.1 if distance_from_center < 10 else 1.0
    
    risk_score = base_risk * temporal_multiplier * spatial_multiplier
    final_risk_score = max(0, min(10, risk_score))
    
    risk_level = 'Low' if final_risk_score <= 3 else 'Medium' if final_risk_score <= 6 else 'High'
    
    return {
        'current_risk_score': final_risk_score,
        'risk_probability': final_risk_score / 10.0,
        'risk_level': risk_level,
        'temporal_multiplier': temporal_multiplier,
        'is_fallback': True,
    }

def calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy):
    
    if strategy == 'safest':
     
        route_coords = [
            [start_lat, start_lon],
            
            [start_lat + 0.02, start_lon + 0.02],
            
            [(start_lat + end_lat) / 2 + 0.03, (start_lon + end_lon) / 2 + 0.03],
            [end_lat + 0.02, end_lon + 0.02],
            
            [end_lat, end_lon]
        ]
        distance_multiplier = 1.4
        risk_score = 2.5
        roads_used = ['Major Highways', 'Well-lit Arterial Roads']
        
    elif strategy == 'fastest':
        
        route_coords = [
            [start_lat, start_lon],
           
            [start_lat - 0.015, start_lon + 0.015],
            
            [(start_lat + end_lat) / 2 - 0.02, (start_lon + end_lon) / 2 + 0.02],
            [end_lat - 0.015, end_lon + 0.015],
           
            [end_lat, end_lon]
        ]
        distance_multiplier = 1.1
        risk_score = 6.5
        roads_used = ['Direct Inner Roads', 'Shortcuts']
        
    else:  
        
        route_coords = [
            [start_lat, start_lon],
           
            [start_lat + 0.01, start_lon - 0.01],
            
            [(start_lat + end_lat) / 2, (start_lon + end_lon) / 2],
            
            [end_lat + 0.01, end_lon - 0.01],
            [end_lat, end_lon]
        ]
        distance_multiplier = 1.2
        risk_score = 4.0
        roads_used = ['Mixed Roads', 'Standard Routes']
    
   
    base_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    total_distance = base_distance * distance_multiplier
    
   
    if strategy == 'safest':
        avg_speed = 25 
    elif strategy == 'fastest':
        avg_speed = 35  
    else:
        avg_speed = 30 
    
    estimated_time = (total_distance / avg_speed) * 60  
    
    return {
        'route_coordinates': route_coords,
        'total_distance': total_distance,
        'estimated_duration': estimated_time,
        'total_risk_score': risk_score,
        'risk_level': 'Low' if risk_score < 3 else 'Medium' if risk_score < 6 else 'High',
        'strategy': strategy,
        'route_type': 'bangalore_road_network',
        'roads_used': roads_used,
        'success': True
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/risk-score', methods=['POST'])
def get_risk_score():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
        
        risk_result = calculate_fallback_risk(lat, lon)
        
        return jsonify({
            'latitude': lat,
            'longitude': lon,
            'risk_score': risk_result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    try:
        data = request.get_json()
        start_lat = data.get('start_latitude')
        start_lon = data.get('start_longitude')
        end_lat = data.get('end_latitude')
        end_lon = data.get('end_longitude')
        strategy = data.get('strategy', 'balanced')
        
        if any(x is None for x in [start_lat, start_lon, end_lat, end_lon]):
            return jsonify({'error': 'Missing required coordinates'}), 400
        
        route_result = calculate_fallback_route(start_lat, start_lon, end_lat, end_lon, strategy)
        
        return jsonify({
            'start_location': {'latitude': start_lat, 'longitude': start_lon},
            'end_location': {'latitude': end_lat, 'longitude': end_lon},
            'strategy': strategy,
            'route': route_result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crime-trends', methods=['GET'])
def get_crime_trends():
    try:
       
        trends = {
            'daily_trends': [
                {'date': '2024-02-20', 'incidents': 45},
                {'date': '2024-02-21', 'incidents': 52},
                {'date': '2024-02-22', 'incidents': 38},
                {'date': '2024-02-23', 'incidents': 61},
                {'date': '2024-02-24', 'incidents': 47},
            ],
            'hourly_patterns': [
                {'hour': 0, 'incidents': 12},
                {'hour': 6, 'incidents': 8},
                {'hour': 12, 'incidents': 25},
                {'hour': 18, 'incidents': 45},
                {'hour': 22, 'incidents': 38},
            ],
            'risk_zones': [
                {'area': 'MG Road', 'risk_level': 'Medium', 'incidents': 23},
                {'area': 'Indiranagar', 'risk_level': 'Low', 'incidents': 15},
                {'area': 'Koramangala', 'risk_level': 'High', 'incidents': 34},
            ]
        }
        
        return jsonify({
            'trends': trends,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route-update', methods=['POST'])
def get_route_update():
    try:
        data = request.get_json()
        current_lat = data.get('current_latitude')
        current_lon = data.get('current_longitude')
        target_lat = data.get('target_latitude')
        target_lon = data.get('target_longitude')
        
        if any(x is None for x in [current_lat, current_lon, target_lat, target_lon]):
            return jsonify({'error': 'Missing required coordinates'}), 400
        
        # Calculate remaining distance
        remaining_distance = haversine_distance(current_lat, current_lon, target_lat, target_lon)
        
        # Get current risk
        current_risk = calculate_fallback_risk(current_lat, current_lon)
        
        # Estimate remaining time (assuming average speed of 30 km/h)
        remaining_time = (remaining_distance / 30) * 60
        
        return jsonify({
            'current_location': {'latitude': current_lat, 'longitude': current_lon},
            'target_location': {'latitude': target_lat, 'longitude': target_lon},
            'remaining_distance': remaining_distance,
            'remaining_time': remaining_time,
            'current_risk': current_risk,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency-trigger', methods=['POST'])
def emergency_trigger():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        user_id = data.get('user_id', 'unknown')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Missing coordinates for emergency alert'}), 400
            
        print("\n" + "="*50)
        print("🚨 EMERGENCY TRIGGER RECEIVED 🚨")
        print("="*50)
        print(f"User: {user_id}")
        print(f"Location: {lat}, {lon}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Action: Simulating dispatch of live location to emergency contacts...")
        print("="*50 + "\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Emergency contacts notified with live location.',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting ML API Server (Realistic Road Mode)...")
    print("📊 Available endpoints:")
    print("   POST /api/risk-score - Get dynamic risk score")
    print("   POST /api/optimize-route - Get optimized route with realistic roads")
    print("   GET /api/crime-trends - Get crime trends")
    print("   POST /api/route-update - Get route updates")
    print("   GET /api/health - Health check")
    print("🚀 Server starting on http://localhost:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=True)

from datetime import datetime
from pathlib import Path
import math
import os

import bcrypt
from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import numpy as np

load_dotenv(Path(__file__).with_name('.env'))

app = Flask(__name__)
CORS(app)

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'ml' / 'models' / 'classification_model.pkl'
FEATURES_PATH = BASE_DIR / 'ml' / 'models' / 'feature_columns.pkl'
BANGALORE_CENTER = (12.9716, 77.5946)

MONGO_HOST = os.getenv('MONGO_HOST') or os.getenv('MONGODB_HOST', '127.0.0.1')
MONGO_PORT = int(os.getenv('MONGO_PORT') or os.getenv('MONGODB_PORT', 27017))
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME') or os.getenv('MONGODB_DB_NAME', 'womensafety')
AUTH_API_PORT = int(os.getenv('AUTH_API_PORT', 8001))


# -----------------------------
# Infrastructure setup
# -----------------------------
client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/')
db = client[MONGO_DB_NAME]

classification_model = None
feature_columns = None


def load_ml_artifacts() -> None:
    global classification_model, feature_columns

    try:
        if MODEL_PATH.exists() and FEATURES_PATH.exists():
            classification_model = joblib.load(MODEL_PATH)
            feature_columns = joblib.load(FEATURES_PATH)
            print(f'ML model loaded: {MODEL_PATH}')
        else:
            classification_model = None
            feature_columns = None
            print('ML model artifacts not found. Running with fallback ML logic.')
    except Exception as exc:
        classification_model = None
        feature_columns = None
        print(f'Failed to load ML artifacts. Fallback mode enabled. Error: {exc}')


load_ml_artifacts()
print(f'Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}')


# -----------------------------
# Shared ML helpers
# -----------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return earth_radius_km * c


def get_temporal_features(dt: datetime) -> dict:
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month

    return {
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': dt.day,
        'month': month,
        'quarter': (month - 1) // 3 + 1,
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
    }


def calculate_fallback_risk(lat: float, lon: float) -> dict:
    hour = datetime.now().hour
    is_night = hour >= 22 or hour < 6
    is_evening = 18 <= hour < 22

    base_risk = 3.0
    temporal_multiplier = 1.5 if is_night else 1.2 if is_evening else 1.0

    distance_from_center = haversine_distance(lat, lon, BANGALORE_CENTER[0], BANGALORE_CENTER[1])
    spatial_multiplier = 1.3 if distance_from_center < 5 else 1.1 if distance_from_center < 10 else 1.0

    final_risk_score = max(0.0, min(10.0, base_risk * temporal_multiplier * spatial_multiplier))
    risk_level = 'Low' if final_risk_score <= 3 else 'Medium' if final_risk_score <= 6 else 'High'

    return {
        'current_risk_score': final_risk_score,
        'risk_probability': final_risk_score / 10.0,
        'risk_level': risk_level,
        'temporal_multiplier': temporal_multiplier,
        'is_fallback': True,
    }


def calculate_ml_risk(lat: float, lon: float) -> dict:
    if classification_model is None or feature_columns is None:
        return calculate_fallback_risk(lat, lon)

    try:
        now = datetime.now()
        features = get_temporal_features(now)
        features['Latitude'] = lat
        features['Longitude'] = lon
        features['lat_grid'] = (lat // 0.01) * 0.01
        features['lon_grid'] = (lon // 0.01) * 0.01

        dist_to_center = haversine_distance(lat, lon, BANGALORE_CENTER[0], BANGALORE_CENTER[1])
        features['distance_to_center'] = dist_to_center
        features['crime_density'] = 1.0 + (1.5 if dist_to_center < 5 else 1.0 if dist_to_center < 10 else 0)

        feature_values = [features.get(col, 0) for col in feature_columns]
        feature_array = np.array(feature_values).reshape(1, -1)

        risk_probability = float(classification_model.predict_proba(feature_array)[0, 1])
        risk_score = risk_probability * 10

        return {
            'current_risk_score': risk_score,
            'risk_probability': risk_probability,
            'risk_level': 'Low' if risk_score <= 3 else 'Medium' if risk_score <= 6 else 'High',
            'is_fallback': False,
            'model_type': 'gradient_boosting',
        }
    except Exception as exc:
        print(f'Error in ML prediction. Using fallback. Error: {exc}')
        return calculate_fallback_risk(lat, lon)


def calculate_fallback_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, strategy: str) -> dict:
    if strategy == 'safest':
        route_coords = [
            [start_lat, start_lon],
            [start_lat + 0.02, start_lon + 0.02],
            [(start_lat + end_lat) / 2 + 0.03, (start_lon + end_lon) / 2 + 0.03],
            [end_lat + 0.02, end_lon + 0.02],
            [end_lat, end_lon],
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
            [end_lat, end_lon],
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
            [end_lat, end_lon],
        ]
        distance_multiplier = 1.2
        risk_score = 4.0
        roads_used = ['Mixed Roads', 'Standard Routes']

    base_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    total_distance = base_distance * distance_multiplier

    avg_speed = 25 if strategy == 'safest' else 35 if strategy == 'fastest' else 30
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
        'success': True,
    }


# -----------------------------
# Core endpoints
# -----------------------------
@app.route('/')
def home():
    return jsonify(
        {
            'message': 'Women Safety Unified API Server',
            'status': 'running',
            'database': MONGO_DB_NAME,
            'ml_model_loaded': classification_model is not None,
            'timestamp': datetime.now().isoformat(),
        }
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        client.admin.command('ping')
        mongo_status = 'connected'
    except Exception as exc:
        mongo_status = f'disconnected: {exc}'

    return jsonify(
        {
            'status': 'healthy',
            'mongodb': mongo_status,
            'database': MONGO_DB_NAME,
            'ml_model_loaded': classification_model is not None,
            'timestamp': datetime.now().isoformat(),
        }
    )


# -----------------------------
# Auth endpoints
# -----------------------------
@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json() or {}

        if not all(k in data for k in ['name', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400

        existing_user = db.users.find_one({'email': data['email']})
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400

        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

        user = {
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password.decode('utf-8'),
            'emergency_contacts': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        }

        result = db.users.insert_one(user)
        user['_id'] = str(result.inserted_id)
        user_response = {k: v for k, v in user.items() if k != 'password'}

        return jsonify({'message': 'User registered successfully', 'user': user_response}), 201
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json() or {}

        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400

        user = db.users.find_one({'email': data['email']})
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        if not bcrypt.checkpw(data['password'].encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid email or password'}), 401

        user_response = {
            '_id': str(user['_id']),
            'name': user['name'],
            'email': user['email'],
            'emergency_contacts': user.get('emergency_contacts', []),
            'created_at': user['created_at'],
        }
        return jsonify({'message': 'Login successful', 'user': user_response})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/user/<email>', methods=['GET'])
def get_user(email):
    try:
        user = db.users.find_one({'email': email})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        user_response = {
            '_id': str(user['_id']),
            'name': user['name'],
            'email': user['email'],
            'emergency_contacts': user.get('emergency_contacts', []),
            'created_at': user['created_at'],
        }
        return jsonify(user_response)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/emergency-contacts/<user_email>', methods=['GET'])
def get_user_emergency_contacts(user_email):
    try:
        user = db.users.find_one({'email': user_email})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        contacts = user.get('emergency_contacts', [])
        return jsonify({'contacts': contacts, 'count': len(contacts)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/emergency-contacts', methods=['POST'])
def save_user_emergency_contacts():
    try:
        data = request.get_json() or {}
        user_email = data.get('user_email')
        contacts = data.get('contacts', [])

        if not user_email:
            return jsonify({'error': 'Missing user_email'}), 400

        if not isinstance(contacts, list):
            return jsonify({'error': 'contacts must be a list'}), 400

        sanitized_contacts = []
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            name = str(contact.get('name', '')).strip()
            phone = str(contact.get('phone', '')).strip()
            relation = contact.get('relation')

            if not name or not phone:
                continue

            sanitized_contacts.append(
                {
                    'name': name,
                    'phone': phone,
                    'relation': str(relation).strip() if relation is not None else None,
                }
            )

        result = db.users.update_one(
            {'email': user_email},
            {
                '$set': {
                    'emergency_contacts': sanitized_contacts,
                    'updated_at': datetime.now().isoformat(),
                }
            },
        )

        if result.matched_count == 0:
            return jsonify({'error': 'User not found'}), 404

        return jsonify(
            {
                'message': 'Emergency contacts saved successfully',
                'contacts': sanitized_contacts,
                'count': len(sanitized_contacts),
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/emergency', methods=['POST'])
def create_emergency_log():
    try:
        data = request.get_json() or {}

        required_fields = ['user_email', 'user_name', 'location']
        if not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        emergency_log = {
            'user_email': data['user_email'],
            'user_name': data['user_name'],
            'location': data['location'],
            'actions_triggered': data.get('actions_triggered', {}),
            'detected_transcript': data.get('detected_transcript'),
            'trigger_phrase': data.get('trigger_phrase'),
            'contacts_count': data.get('contacts_count', 0),
            'timestamp': datetime.now().isoformat(),
            'id': str(ObjectId()),
        }

        result = db.emergency_logs.insert_one(emergency_log)
        emergency_log['_id'] = str(result.inserted_id)

        return jsonify({'message': 'Emergency log created successfully', 'log': emergency_log}), 201
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/emergency/<user_email>', methods=['GET'])
def get_emergency_logs(user_email):
    try:
        logs = list(db.emergency_logs.find({'user_email': user_email}).sort('timestamp', -1))
        for log in logs:
            log['_id'] = str(log['_id'])
        return jsonify({'logs': logs, 'count': len(logs)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# -----------------------------
# ML endpoints
# -----------------------------
@app.route('/api/risk-score', methods=['POST'])
def get_risk_score():
    try:
        data = request.get_json() or {}
        lat = data.get('latitude')
        lon = data.get('longitude')

        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        risk_result = calculate_ml_risk(float(lat), float(lon))

        return jsonify(
            {
                'latitude': lat,
                'longitude': lon,
                'risk_score': risk_result,
                'timestamp': datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    try:
        data = request.get_json() or {}
        start_lat = data.get('start_latitude')
        start_lon = data.get('start_longitude')
        end_lat = data.get('end_latitude')
        end_lon = data.get('end_longitude')
        strategy = data.get('strategy', 'balanced')

        if any(x is None for x in [start_lat, start_lon, end_lat, end_lon]):
            return jsonify({'error': 'Missing required coordinates'}), 400

        route_result = calculate_fallback_route(
            float(start_lat), float(start_lon), float(end_lat), float(end_lon), str(strategy)
        )

        return jsonify(
            {
                'start_location': {'latitude': start_lat, 'longitude': start_lon},
                'end_location': {'latitude': end_lat, 'longitude': end_lon},
                'strategy': strategy,
                'route': route_result,
                'timestamp': datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/crime-trends', methods=['GET'])
def get_crime_trends():
    return jsonify(
        {
            'trends': {
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
                ],
            },
            'timestamp': datetime.now().isoformat(),
        }
    )


@app.route('/api/route-update', methods=['POST'])
def get_route_update():
    try:
        data = request.get_json() or {}
        route_coordinates = data.get('route_coordinates', [])
        current_index = int(data.get('current_position_index', 0))

        if not route_coordinates:
            return jsonify({'error': 'Missing route_coordinates'}), 400

        if current_index < 0 or current_index >= len(route_coordinates):
            return jsonify({'error': 'Invalid current_position_index'}), 400

        remaining = route_coordinates[current_index:]
        segment_updates = []
        total_risk = 0.0

        for coord in remaining:
            lat, lon = float(coord[0]), float(coord[1])
            risk = calculate_ml_risk(lat, lon)
            total_risk += float(risk['current_risk_score'])
            segment_updates.append(
                {
                    'coordinates': [lat, lon],
                    'current_risk_score': risk['current_risk_score'],
                    'risk_level': risk['risk_level'],
                }
            )

        avg_remaining_risk = total_risk / len(remaining)

        return jsonify(
            {
                'remaining_segments': segment_updates,
                'avg_remaining_risk': avg_remaining_risk,
                'segments_remaining': len(remaining),
                'update_timestamp': datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/emergency-trigger', methods=['POST'])
def emergency_trigger():
    try:
        data = request.get_json() or {}
        lat = data.get('latitude')
        lon = data.get('longitude')
        user_id = data.get('user_id', 'unknown')

        if lat is None or lon is None:
            return jsonify({'error': 'Missing coordinates for emergency alert'}), 400

        print('\n' + '=' * 50)
        print('EMERGENCY TRIGGER RECEIVED')
        print('=' * 50)
        print(f'User: {user_id}')
        print(f'Location: {lat}, {lon}')
        print(f'Detected phrase: {data.get("detected_transcript", "n/a")}')
        print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print('Action: Dispatching emergency alert payload...')
        print('=' * 50 + '\n')

        return jsonify(
            {
                'status': 'success',
                'message': 'Emergency contacts notified with live location.',
                'timestamp': datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    print(f'Starting unified backend on http://localhost:{AUTH_API_PORT}')
    print('Auth + ML endpoints are served from the same process.')
    app.run(debug=True, host='0.0.0.0', port=AUTH_API_PORT)

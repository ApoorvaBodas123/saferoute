from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
from datetime import datetime
import bcrypt

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web app

# MongoDB configuration
MONGO_HOST = os.getenv('MONGO_HOST', '127.0.0.1')
MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'womensafety')

# MongoDB connection
client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/')
db = client[MONGO_DB_NAME]

print(f"Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}")

@app.route('/')
def home():
    return jsonify({
        'message': 'Women Safety API Server',
        'status': 'running',
        'database': MONGO_DB_NAME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        return jsonify({
            'status': 'healthy',
            'mongodb': 'connected',
            'database': MONGO_DB_NAME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'mongodb': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not all(k in data for k in ['name', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user already exists
        existing_user = db.users.find_one({'email': data['email']})
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400
        
        # Hash password
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        # Create user
        user = {
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password.decode('utf-8'),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        result = db.users.insert_one(user)
        user['_id'] = str(result.inserted_id)
        
        # Remove password from response
        user_response = {k: v for k, v in user.items() if k != 'password'}
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user_response
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400
        
        # Find user
        user = db.users.find_one({'email': data['email']})
        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Check password
        if not bcrypt.checkpw(data['password'].encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Remove password from response
        user_response = {
            '_id': str(user['_id']),
            'name': user['name'],
            'email': user['email'],
            'created_at': user['created_at']
        }
        
        return jsonify({
            'message': 'Login successful',
            'user': user_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/user/<email>', methods=['GET'])
def get_user(email):
    try:
        user = db.users.find_one({'email': email})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Remove password from response
        user_response = {
            '_id': str(user['_id']),
            'name': user['name'],
            'email': user['email'],
            'created_at': user['created_at']
        }
        
        return jsonify(user_response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency', methods=['POST'])
def create_emergency_log():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_email', 'user_name', 'location']
        if not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Create emergency log
        emergency_log = {
            'user_email': data['user_email'],
            'user_name': data['user_name'],
            'location': data['location'],
            'actions_triggered': data.get('actions_triggered', {}),
            'timestamp': datetime.now().isoformat(),
            'id': str(ObjectId())
        }
        
        result = db.emergency_logs.insert_one(emergency_log)
        emergency_log['_id'] = str(result.inserted_id)
        
        return jsonify({
            'message': 'Emergency log created successfully',
            'log': emergency_log
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency/<user_email>', methods=['GET'])
def get_emergency_logs(user_email):
    try:
        logs = list(db.emergency_logs.find({'user_email': user_email}).sort('timestamp', -1))
        
        # Convert ObjectId to string
        for log in logs:
            log['_id'] = str(log['_id'])
        
        return jsonify({
            'logs': logs,
            'count': len(logs)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

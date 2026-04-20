import 'package:mongo_dart/mongo_dart.dart';
import 'dart:convert';
import 'dart:html' as html show window;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class MongoDbService {
  static Db? _db;
  static const String _usersCollection = 'users';
  static const String _emergencyCollection = 'emergency_logs';
  static bool _isWebPlatform = false;
  static bool _connectionAttempted = false;
  static bool _isConnected = false;
  
  // Initialize environment variables
  static Future<void> loadEnv() async {
    try {
      await dotenv.load(fileName: ".env");
      print('Environment variables loaded successfully');
      print('MongoDB Host: ${dotenv.env['MONGODB_HOST']}');
      print('MongoDB Port: ${dotenv.env['MONGODB_PORT']}');
      print('MongoDB DB: ${dotenv.env['MONGODB_DB_NAME']}');
    } catch (e) {
      print('Could not load .env file: $e');
      print('Using default MongoDB connection settings');
    }
  }
  
  // Get connection string from environment variables
  static String get _connectionString {
    final host = dotenv.env['MONGODB_HOST'] ?? '127.0.0.1';
    final port = dotenv.env['MONGODB_PORT'] ?? '27017';
    final dbName = dotenv.env['MONGODB_DB_NAME'] ?? 'womensafety';
    return 'mongodb://$host:$port/$dbName';
  }
  
  static String get _dbName {
    return dotenv.env['MONGODB_DB_NAME'] ?? 'womensafety';
  }

  // Check if running on web platform
  static bool get _isWeb {
    if (!_connectionAttempted) {
      // More reliable web platform detection
      _isWebPlatform = identical(0, 0.0) && !Uri.base.toString().startsWith('file://');
      _connectionAttempted = true;
      print('Platform detection: ${_isWebPlatform ? "Web" : "Native/Mobile"}');
    }
    return _isWebPlatform;
  }

  // Initialize MongoDB connection
  static Future<void> init() async {
    // Load environment variables first
    await loadEnv();
    
    if (_isWeb) {
      print('MongoDB: Web platform detected - using local storage fallback');
      _isConnected = false;
      return;
    }

    if (_isConnected && _db != null) {
      print('MongoDB: Already connected');
      return;
    }

    try {
      print('MongoDB: Attempting to connect to $_connectionString...');
      _db = await Db.create(_connectionString);
      await _db!.open();
      _isConnected = true;
      print('MongoDB connected successfully to ${_dbName} database');
    } catch (e) {
      _isConnected = false;
      print('MongoDB connection failed: $e');
      print('MongoDB: Falling back to local storage for web compatibility');
    }
  }

  // Save user to MongoDB
  static Future<bool> saveUser(Map<String, dynamic> user) async {
    try {
      if (_isWeb) {
        return await _saveUserToLocalStorage(user);
      }
      
      if (_db == null || !_isConnected) await init();
      
      if (!_isConnected) {
        print('MongoDB not available - using local storage fallback');
        return await _saveUserToLocalStorage(user);
      }
      
      final collection = _db!.collection(_usersCollection);
      await collection.insertOne(user);
      print('User saved to MongoDB: ${user['email']}');
      return true;
    } catch (e) {
      print('Error saving user to MongoDB: $e');
      print('Attempting local storage fallback...');
      return await _saveUserToLocalStorage(user);
    }
  }

  // Save user to local storage (web fallback)
  static Future<bool> _saveUserToLocalStorage(Map<String, dynamic> user) async {
    try {
      final users = await _getUsersFromLocalStorage();
      users[user['email']] = user;
      await _saveUsersToLocalStorage(users);
      print('User saved to local storage: ${user['email']}');
      return true;
    } catch (e) {
      print('Error saving user to local storage: $e');
      return false;
    }
  }

  // Find user by email
  static Future<Map<String, dynamic>?> findUserByEmail(String email) async {
    try {
      if (_isWeb) {
        return await _findUserInLocalStorage(email);
      }
      
      if (_db == null || !_isConnected) await init();
      
      if (!_isConnected) {
        print('MongoDB not available - using local storage fallback');
        return await _findUserInLocalStorage(email);
      }
      
      final collection = _db!.collection(_usersCollection);
      final user = await collection.findOne(where.eq('email', email));
      return user;
    } catch (e) {
      print('Error finding user in MongoDB: $e');
      print('Attempting local storage fallback...');
      return await _findUserInLocalStorage(email);
    }
  }

  // Find user in local storage (web fallback)
  static Future<Map<String, dynamic>?> _findUserInLocalStorage(String email) async {
    try {
      final users = await _getUsersFromLocalStorage();
      final user = users[email];
      if (user != null) {
        print('User found in local storage: $email');
      }
      return user;
    } catch (e) {
      print('Error finding user in local storage: $e');
      return null;
    }
  }

  // Save emergency log
  static Future<bool> saveEmergencyLog(Map<String, dynamic> emergencyData) async {
    try {
      if (_isWeb) {
        return await _saveEmergencyLogToLocalStorage(emergencyData);
      }
      
      if (_db == null || !_isConnected) await init();
      
      if (!_isConnected) {
        print('MongoDB not available - using local storage fallback');
        return await _saveEmergencyLogToLocalStorage(emergencyData);
      }
      
      final collection = _db!.collection(_emergencyCollection);
      final logEntry = {
        ...emergencyData,
        'timestamp': DateTime.now().toIso8601String(),
        'id': DateTime.now().millisecondsSinceEpoch.toString(),
      };
      await collection.insertOne(logEntry);
      print('Emergency log saved to MongoDB');
      return true;
    } catch (e) {
      print('Error saving emergency log to MongoDB: $e');
      print('Attempting local storage fallback...');
      return await _saveEmergencyLogToLocalStorage(emergencyData);
    }
  }

  // Save emergency log to local storage (web fallback)
  static Future<bool> _saveEmergencyLogToLocalStorage(Map<String, dynamic> emergencyData) async {
    try {
      final logs = await _getEmergencyLogsFromLocalStorage();
      final logEntry = {
        ...emergencyData,
        'timestamp': DateTime.now().toIso8601String(),
        'id': DateTime.now().millisecondsSinceEpoch.toString(),
      };
      logs.add(logEntry);
      await _saveEmergencyLogsToLocalStorage(logs);
      print('Emergency log saved to local storage');
      return true;
    } catch (e) {
      print('Error saving emergency log to local storage: $e');
      return false;
    }
  }

  // Get emergency logs for a user
  static Future<List<Map<String, dynamic>>> getEmergencyLogs(String userEmail) async {
    try {
      if (_isWeb) {
        return await _getEmergencyLogsFromLocalStorageForUser(userEmail);
      }
      
      if (_db == null || !_isConnected) await init();
      
      if (!_isConnected) {
        print('MongoDB not available - using local storage fallback');
        return await _getEmergencyLogsFromLocalStorageForUser(userEmail);
      }
      
      final collection = _db!.collection(_emergencyCollection);
      final logs = await collection.find(where.eq('user_email', userEmail)).toList();
      return logs;
    } catch (e) {
      print('Error fetching emergency logs from MongoDB: $e');
      print('Attempting local storage fallback...');
      return await _getEmergencyLogsFromLocalStorageForUser(userEmail);
    }
  }

  // Get emergency logs from local storage for a user (web fallback)
  static Future<List<Map<String, dynamic>>> _getEmergencyLogsFromLocalStorageForUser(String userEmail) async {
    try {
      final logs = await _getEmergencyLogsFromLocalStorage();
      final userLogs = logs.where((log) => log['user_email'] == userEmail).toList();
      print('Found ${userLogs.length} emergency logs for $userEmail in local storage');
      return userLogs;
    } catch (e) {
      print('Error fetching emergency logs from local storage: $e');
      return [];
    }
  }

  // Local storage helper methods
  static Future<Map<String, dynamic>> _getUsersFromLocalStorage() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final usersJson = prefs.getString('local_users') ?? '{}';
      return Map<String, dynamic>.from(jsonDecode(usersJson));
    } catch (e) {
      print('Error reading users from local storage: $e');
      return {};
    }
  }

  static Future<void> _saveUsersToLocalStorage(Map<String, dynamic> users) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('local_users', jsonEncode(users));
      print('Saved ${users.length} users to local storage');
    } catch (e) {
      print('Error saving users to local storage: $e');
    }
  }

  static Future<List<Map<String, dynamic>>> _getEmergencyLogsFromLocalStorage() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final logsJson = prefs.getString('local_emergency_logs') ?? '[]';
      return List<Map<String, dynamic>>.from(jsonDecode(logsJson));
    } catch (e) {
      print('Error reading emergency logs from local storage: $e');
      return [];
    }
  }

  static Future<void> _saveEmergencyLogsToLocalStorage(List<Map<String, dynamic>> logs) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('local_emergency_logs', jsonEncode(logs));
      print('Saved ${logs.length} emergency logs to local storage');
    } catch (e) {
      print('Error saving emergency logs to local storage: $e');
    }
  }

  // Close MongoDB connection
  static Future<void> close() async {
    if (_db != null && _isConnected) {
      await _db!.close();
      _isConnected = false;
      print('MongoDB connection closed');
    }
  }
}

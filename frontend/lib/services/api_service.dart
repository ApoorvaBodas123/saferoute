import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class ApiService {
  static String get _baseUrl {
    // In production, this would come from environment variables
    return dotenv.env['API_BASE_URL'] ?? 'http://localhost:8001';
  }

  // Health check
  static Future<Map<String, dynamic>> healthCheck() async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/api/health'));
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Health check failed: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Health check error: $e');
    }
  }

  // Register user
  static Future<Map<String, dynamic>> register(Map<String, dynamic> userData) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/register'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(userData),
      );

      final responseData = jsonDecode(response.body);
      
      if (response.statusCode == 201) {
        return responseData;
      } else {
        throw Exception(responseData['error'] ?? 'Registration failed');
      }
    } catch (e) {
      throw Exception('Registration error: $e');
    }
  }

  // Login user
  static Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'email': email, 'password': password}),
      );

      final responseData = jsonDecode(response.body);
      
      if (response.statusCode == 200) {
        return responseData;
      } else {
        throw Exception(responseData['error'] ?? 'Login failed');
      }
    } catch (e) {
      throw Exception('Login error: $e');
    }
  }

  // Get user by email
  static Future<Map<String, dynamic>?> getUserByEmail(String email) async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/api/user/$email'));
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else if (response.statusCode == 404) {
        return null;
      } else {
        throw Exception('Failed to get user: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Get user error: $e');
    }
  }

  // Create emergency log
  static Future<Map<String, dynamic>> createEmergencyLog(Map<String, dynamic> emergencyData) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/emergency'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(emergencyData),
      );

      final responseData = jsonDecode(response.body);
      
      if (response.statusCode == 201) {
        return responseData;
      } else {
        throw Exception(responseData['error'] ?? 'Failed to create emergency log');
      }
    } catch (e) {
      throw Exception('Emergency log error: $e');
    }
  }

  // Get emergency logs for user
  static Future<List<Map<String, dynamic>>> getEmergencyLogs(String userEmail) async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/api/emergency/$userEmail'));
      
      if (response.statusCode == 200) {
        final responseData = jsonDecode(response.body);
        return List<Map<String, dynamic>>.from(responseData['logs']);
      } else {
        throw Exception('Failed to get emergency logs: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Get emergency logs error: $e');
    }
  }

  static Future<List<Map<String, dynamic>>> getEmergencyContacts(String userEmail) async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/api/emergency-contacts/$userEmail'));

      final responseData = jsonDecode(response.body);
      if (response.statusCode == 200) {
        return List<Map<String, dynamic>>.from(responseData['contacts'] ?? const []);
      }

      throw Exception(responseData['error'] ?? 'Failed to get emergency contacts');
    } catch (e) {
      throw Exception('Get emergency contacts error: $e');
    }
  }

  static Future<List<Map<String, dynamic>>> saveEmergencyContacts(
    String userEmail,
    List<Map<String, dynamic>> contacts,
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/emergency-contacts'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_email': userEmail, 'contacts': contacts}),
      );

      final responseData = jsonDecode(response.body);
      if (response.statusCode == 200) {
        return List<Map<String, dynamic>>.from(responseData['contacts'] ?? const []);
      }

      throw Exception(responseData['error'] ?? 'Failed to save emergency contacts');
    } catch (e) {
      throw Exception('Save emergency contacts error: $e');
    }
  }
}

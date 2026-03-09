import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class AuthService {
  static const String _userKey = 'current_user';

  // Check if a user is currently logged in
  static Future<bool> isLoggedIn() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.containsKey(_userKey);
  }

  // Get current user details
  static Future<Map<String, dynamic>?> getCurrentUser() async {
    final prefs = await SharedPreferences.getInstance();
    final userStr = prefs.getString(_userKey);
    if (userStr != null) {
      return jsonDecode(userStr);
    }
    return null;
  }

  // Login a user
  static Future<bool> login(String email, String password) async {
    final prefs = await SharedPreferences.getInstance();
    // Simulate API delay
    await Future.delayed(const Duration(seconds: 1));

    // For local mock, we accept any matching pattern from saved "db"
    // To keep it simple, if no specific DB, we just log them in if not empty
    if (email.isNotEmpty && password.isNotEmpty) {
      // Create a mock user profile
      final user = {
        'name': email.split('@')[0], // Extract name from email as fallback
        'email': email,
      };
      
      // Look for specifically registered user data by email prefix key
      final registeredUserStr = prefs.getString('user_$email');
      if (registeredUserStr != null) {
        final registeredUser = jsonDecode(registeredUserStr);
        // Basic password check
        if (registeredUser['password'] == password) {
           await prefs.setString(_userKey, jsonEncode(registeredUser));
           return true;
        } else {
          return false; // Wrong password
        }
      }

      // If not strictly registered but valid, let's allow login for demo purposes
      await prefs.setString(_userKey, jsonEncode(user));
      return true;
    }
    return false;
  }

  // Register a new user
  static Future<bool> register(String name, String email, String password) async {
    final prefs = await SharedPreferences.getInstance();
    // Simulate API delay
    await Future.delayed(const Duration(seconds: 1));

    if (name.isNotEmpty && email.isNotEmpty && password.isNotEmpty) {
      final user = {
        'name': name,
        'email': email,
        'password': password, // Storing password for mock login validation
      };
      
      // Save specific user data
      await prefs.setString('user_$email', jsonEncode(user));
      // Log them in immediately
      await prefs.setString(_userKey, jsonEncode({'name': name, 'email': email}));
      return true;
    }
    return false;
  }

  // Logout the current user
  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_userKey);
  }
}

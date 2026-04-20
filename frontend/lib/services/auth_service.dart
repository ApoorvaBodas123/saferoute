import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'api_service.dart';

class AuthService {
  static const String _userKey = 'current_user';

  // Email validation
  static String? validateEmail(String email) {
    if (email.isEmpty) return 'Email is required';
    
    // Check for @gmail.com specifically
    if (!email.endsWith('@gmail.com')) {
      return 'Email must be a @gmail.com address';
    }
    
    // Basic email format validation
    final emailRegex = RegExp(r'^[a-zA-Z0-9._%+-]+@gmail\.com$');
    if (!emailRegex.hasMatch(email)) {
      return 'Invalid email format';
    }
    
    return null;
  }

  // Password validation
  static String? validatePassword(String password) {
    if (password.isEmpty) return 'Password is required';
    if (password.length < 8) return 'Password must be at least 8 characters';
    
    // Check for uppercase
    if (!password.contains(RegExp(r'[A-Z]'))) {
      return 'Password must contain at least one uppercase letter';
    }
    
    // Check for lowercase
    if (!password.contains(RegExp(r'[a-z]'))) {
      return 'Password must contain at least one lowercase letter';
    }
    
    // Check for special symbols
    if (!password.contains(RegExp(r'[!@#$%^&*(),.?":{}|<>]'))) {
      return 'Password must contain at least one special symbol';
    }
    
    return null;
  }

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
  static Future<String?> login(String email, String password) async {
    // Validate inputs
    final emailError = validateEmail(email);
    if (emailError != null) return emailError;
    
    final passwordError = validatePassword(password);
    if (passwordError != null) return passwordError;
    
    try {
      final response = await ApiService.login(email, password);
      
      if (response['message'] == 'Login successful') {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_userKey, jsonEncode(response['user']));
        return null; // Success
      } else {
        return 'Login failed';
      }
    } catch (e) {
      return e.toString().replaceAll('Exception: ', '');
    }
  }

  // Register a new user
  static Future<String?> register(String name, String email, String password) async {
    // Validate inputs
    final emailError = validateEmail(email);
    if (emailError != null) return emailError;
    
    final passwordError = validatePassword(password);
    if (passwordError != null) return passwordError;
    
    try {
      final userData = {
        'name': name,
        'email': email,
        'password': password,
      };
      
      final response = await ApiService.register(userData);
      
      if (response['message'] == 'User registered successfully') {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_userKey, jsonEncode(response['user']));
        return null; // Success
      } else {
        return 'Registration failed';
      }
    } catch (e) {
      return e.toString().replaceAll('Exception: ', '');
    }
  }

  // Logout the current user
  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_userKey);
  }
}

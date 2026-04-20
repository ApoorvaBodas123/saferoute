import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'mongodb_service.dart';

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
    
    final prefs = await SharedPreferences.getInstance();
    // Simulate API delay
    await Future.delayed(const Duration(seconds: 1));

    // Look for user in MongoDB
    final registeredUser = await MongoDbService.findUserByEmail(email);
    if (registeredUser != null) {
      // Basic password check
      if (registeredUser['password'] == password) {
         await prefs.setString(_userKey, jsonEncode(registeredUser));
         return null; // Success
      } else {
         return 'Invalid email or password'; // Wrong password
      }
    }

    return 'Email not registered. Please register first.';
  }

  // Register a new user
  static Future<String?> register(String name, String email, String password) async {
    // Validate inputs
    final emailError = validateEmail(email);
    if (emailError != null) return emailError;
    
    final passwordError = validatePassword(password);
    if (passwordError != null) return passwordError;
    
    final prefs = await SharedPreferences.getInstance();
    // Simulate API delay
    await Future.delayed(const Duration(seconds: 1));

    // Check if user already exists in MongoDB
    final existingUser = await MongoDbService.findUserByEmail(email);
    if (existingUser != null) {
      return 'Email already registered. Please use a different email.';
    }

    if (name.isNotEmpty && email.isNotEmpty && password.isNotEmpty) {
      final user = {
        'name': name,
        'email': email,
        'password': password,
        'created_at': DateTime.now().toIso8601String(),
      };
      
      // Save user to MongoDB
      final success = await MongoDbService.saveUser(user);
      if (success) {
        // Log them in immediately
        await prefs.setString(_userKey, jsonEncode({'name': name, 'email': email}));
        return null; // Success
      } else {
        return 'Registration failed. Please try again.';
      }
    }
    return 'Registration failed. Please try again.';
  }

  // Logout the current user
  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_userKey);
  }
}

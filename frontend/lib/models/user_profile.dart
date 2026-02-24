import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class UserProfile {
  final String name;
  final String email;
  final SafetyLevel riskTolerance;
  final List<String> preferredAreas;
  final List<String> avoidedAreas;
  final TimeOfDay usualTravelStart;
  final TimeOfDay usualTravelEnd;

  UserProfile({
    required this.name,
    required this.email,
    this.riskTolerance = SafetyLevel.moderate,
    this.preferredAreas = const [],
    this.avoidedAreas = const [],
    this.usualTravelStart = const TimeOfDay(hour: 9, minute: 0),
    this.usualTravelEnd = const TimeOfDay(hour: 17, minute: 0),
  });

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'email': email,
      'riskTolerance': riskTolerance.toString(),
      'preferredAreas': preferredAreas,
      'avoidedAreas': avoidedAreas,
      'usualTravelStart': '${usualTravelStart.hour}:${usualTravelStart.minute}',
      'usualTravelEnd': '${usualTravelEnd.hour}:${usualTravelEnd.minute}',
    };
  }

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      name: json['name'] ?? '',
      email: json['email'] ?? '',
      riskTolerance: SafetyLevel.values.firstWhere(
        (e) => e.toString() == json['riskTolerance'],
        orElse: () => SafetyLevel.moderate,
      ),
      preferredAreas: List<String>.from(json['preferredAreas'] ?? []),
      avoidedAreas: List<String>.from(json['avoidedAreas'] ?? []),
      usualTravelStart: _parseTime(json['usualTravelStart'] ?? '09:00'),
      usualTravelEnd: _parseTime(json['usualTravelEnd'] ?? '17:00'),
    );
  }

  static TimeOfDay _parseTime(String timeString) {
    final parts = timeString.split(':');
    return TimeOfDay(
      hour: int.parse(parts[0]),
      minute: int.parse(parts[1]),
    );
  }

  static Future<void> saveProfile(UserProfile profile) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_profile', json.encode(profile.toJson()));
  }

  static Future<UserProfile?> loadProfile() async {
    final prefs = await SharedPreferences.getInstance();
    final profileJson = prefs.getString('user_profile');
    
    if (profileJson != null) {
      return UserProfile.fromJson(json.decode(profileJson));
    }
    return null;
  }
}

enum SafetyLevel {
  conservative,
  moderate,
  adventurous,
}

extension SafetyLevelExtension on SafetyLevel {
  String get displayName {
    switch (this) {
      case SafetyLevel.conservative:
        return 'Conservative (Very Safe)';
      case SafetyLevel.moderate:
        return 'Moderate (Balanced)';
      case SafetyLevel.adventurous:
        return 'Adventurous (Fastest)';
    }
  }

  double get riskMultiplier {
    switch (this) {
      case SafetyLevel.conservative:
        return 0.7;  // Prefer safer routes
      case SafetyLevel.moderate:
        return 1.0;  // Balanced approach
      case SafetyLevel.adventurous:
        return 1.3;  // Prefer faster routes
    }
  }
}

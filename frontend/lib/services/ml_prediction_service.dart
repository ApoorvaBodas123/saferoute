import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:math' as math;

class MLPredictionService {
  static const String _baseUrl = 'http://localhost:8000'; // ML API endpoint
  
  // Get dynamic risk score for a location
  static Future<Map<String, dynamic>> getDynamicRiskScore(
    double latitude, 
    double longitude, {
    DateTime? predictionTime
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/risk-score'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'latitude': latitude,
          'longitude': longitude,
          'prediction_time': predictionTime?.toIso8601String() ?? DateTime.now().toIso8601String(),
        }),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final Map<String, dynamic> result = jsonDecode(response.body);
        final Map<String, dynamic> riskData = result['risk_score'];
        riskData['is_fallback'] = false;
        return riskData;
      } else {
        throw Exception('Failed to get risk score: ${response.statusCode}');
      }
    } catch (e) {
      print('ML API Error: $e');
      // Fallback to static risk calculation
      return _calculateFallbackRiskScore(latitude, longitude);
    }
  }

  // Get ML-optimized route
  static Future<Map<String, dynamic>> getOptimizedRoute(
    double startLat,
    double startLon,
    double endLat,
    double endLon, {
    String strategy = 'balanced'
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/optimize-route'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'start_latitude': startLat,
          'start_longitude': startLon,
          'end_latitude': endLat,
          'end_longitude': endLon,
          'strategy': strategy,
        }),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        final Map<String, dynamic> result = jsonDecode(response.body);
        final Map<String, dynamic> routeData = result['route'];
        routeData['is_fallback'] = false;
        return routeData;
      } else {
        throw Exception('Failed to get optimized route: ${response.statusCode}');
      }
    } catch (e) {
      print('ML API Error: $e');
      // Fallback to basic route calculation
      return _calculateFallbackRoute(startLat, startLon, endLat, endLon, strategy);
    }
  }

  // Get crime trend analysis
  static Future<Map<String, dynamic>> getCrimeTrends() async {
    try {
      final response = await http.get(
        Uri.parse('$_baseUrl/api/crime-trends'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to get crime trends: ${response.statusCode}');
      }
    } catch (e) {
      // Fallback to static trends
      return _getFallbackTrends();
    }
  }

  // Get real-time route updates
  static Future<Map<String, dynamic>> getRouteUpdate(
    List<List<double>> routeCoordinates,
    int currentPositionIndex
  ) async {
    try {
      final response = await http.post(
        Uri.parse('$_baseUrl/api/route-update'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'route_coordinates': routeCoordinates,
          'current_position_index': currentPositionIndex,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to get route update: ${response.statusCode}');
      }
    } catch (e) {
      // Fallback to basic update
      return _getFallbackRouteUpdate(routeCoordinates, currentPositionIndex);
    }
  }

  // Fallback methods when ML API is unavailable
  static Map<String, dynamic> _calculateFallbackRiskScore(double latitude, double longitude) {
    final hour = DateTime.now().hour;
    final isNight = hour >= 22 || hour < 6;
    final isEvening = hour >= 18 && hour < 22;
    
    // Base risk calculation
    double baseRisk = 3.0;
    
    // Time-based adjustments
    if (isNight) {
      baseRisk *= 1.5;
    } else if (isEvening) {
      baseRisk *= 1.2;
    }
    
    // Distance from center (simplified)
    final distanceFromCenter = _calculateDistance(latitude, longitude, 12.9716, 77.5946);
    if (distanceFromCenter < 5) {
      baseRisk *= 1.3;
    } else if (distanceFromCenter < 10) {
      baseRisk *= 1.1;
    }
    
    final riskScore = baseRisk.clamp(0.0, 10.0);
    String riskLevel;
    if (riskScore <= 3) {
      riskLevel = 'Low';
    } else if (riskScore <= 6) {
      riskLevel = 'Medium';
    } else if (riskScore <= 8) {
      riskLevel = 'High';
    } else {
      riskLevel = 'Very High';
    }
    
    return {
      'current_risk_score': riskScore,
      'risk_probability': riskScore / 10.0,
      'risk_level': riskLevel,
      'temporal_multiplier': isNight ? 1.5 : (isEvening ? 1.2 : 1.0),
      'is_fallback': true,
    };
  }

  static Map<String, dynamic> _calculateFallbackRoute(
    double startLat, double startLon, double endLat, double endLon, String strategy
  ) {
    // Simple straight-line route with risk assessment
    final routeCoordinates = [
      [startLat, startLon],
      [endLat, endLon]
    ];
    
    // Calculate basic metrics
    final distance = _calculateDistance(startLat, startLon, endLat, endLon);
    final estimatedTime = (distance / 30) * 60; // Assuming 30 km/h average speed
    
    // Simple risk assessment
    final midLat = (startLat + endLat) / 2;
    final midLon = (startLon + endLon) / 2;
    final riskResult = _calculateFallbackRiskScore(midLat, midLon);
    
    return {
      'route_coordinates': routeCoordinates,
      'strategy': 'fallback',
      'distance_km': distance,
      'avg_risk_score': riskResult['current_risk_score'],
      'risk_level': riskResult['risk_level'],
      'estimated_time_minutes': estimatedTime,
      'high_risk_segments': [],
      'is_fallback': true,
    };
  }

  static Map<String, dynamic> _getFallbackTrends() {
    return {
      'hourly_pattern': {
        '0': 1.2, '1': 1.3, '2': 1.4, '3': 1.5, '4': 1.4, '5': 1.3,
        '6': 0.8, '7': 0.7, '8': 0.6, '9': 0.5, '10': 0.6, '11': 0.7,
        '12': 0.8, '13': 0.9, '14': 1.0, '15': 1.1, '16': 1.2, '17': 1.3,
        '18': 1.4, '19': 1.5, '20': 1.4, '21': 1.3, '22': 1.4, '23': 1.3
      },
      'daily_pattern': {
        '0': 1.1, '1': 1.0, '2': 1.0, '3': 1.1, '4': 1.2, '5': 1.3, '6': 1.4
      },
      'monthly_pattern': {
        '1': 0.9, '2': 0.8, '3': 0.9, '4': 1.0, '5': 1.1, '6': 1.2,
        '7': 1.3, '8': 1.3, '9': 1.2, '10': 1.1, '11': 1.0, '12': 0.9
      },
      'is_fallback': true,
    };
  }

  static Map<String, dynamic> _getFallbackRouteUpdate(
    List<List<double>> routeCoordinates, int currentPositionIndex
  ) {
    if (currentPositionIndex >= routeCoordinates.length) {
      return {'error': 'Invalid position index'};
    }
    
    final remainingRoute = routeCoordinates.skip(currentPositionIndex).toList();
    final totalRisk = remainingRoute.fold(0.0, (sum, coord) {
      final riskResult = _calculateFallbackRiskScore(coord[0], coord[1]);
      return sum + (riskResult['current_risk_score'] as double);
    });
    
    return {
      'remaining_segments': remainingRoute.asMap().entries.map((entry) => {
        'coordinates': entry.value,
        'current_risk_score': _calculateFallbackRiskScore(entry.value[0], entry.value[1])['current_risk_score'],
        'risk_level': _calculateFallbackRiskScore(entry.value[0], entry.value[1])['risk_level'],
      }).toList(),
      'avg_remaining_risk': totalRisk / remainingRoute.length,
      'segments_remaining': remainingRoute.length,
      'update_timestamp': DateTime.now().toIso8601String(),
      'is_fallback': true,
    };
  }

  // Utility method to calculate distance between two points
  static double _calculateDistance(double lat1, double lon1, double lat2, double lon2) {
    const double earthRadius = 6371; // in kilometers
    
    final double dLat = _toRadians(lat2 - lat1);
    final double dLon = _toRadians(lon2 - lon1);
    
    final double a = (dLat / 2).sin() * (dLat / 2).sin() +
        lat1.toRadians().cos() * lat2.toRadians().cos() *
        (dLon / 2).sin() * (dLon / 2).sin();
    final double c = 2 * a.sqrt().asin();
    
    return earthRadius * c;
  }

  static double _toRadians(double degrees) {
    return degrees * (3.14159265359 / 180);
  }
}

// Extension methods for math operations
extension DoubleExtension on double {
  double toRadians() => this * (3.14159265359 / 180);
  double sin() => math.sin(this);
  double cos() => math.cos(this);
  double sqrt() => math.sqrt(this);
  double asin() => math.asin(this);
}

import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class RouteHistory {
  final String id;
  final List<List<double>> coordinates;
  final String strategy;
  final double distance;
  final double riskScore;
  final String riskLevel;
  final DateTime timestamp;
  final int durationMinutes;

  RouteHistory({
    required this.id,
    required this.coordinates,
    required this.strategy,
    required this.distance,
    required this.riskScore,
    required this.riskLevel,
    required this.timestamp,
    required this.durationMinutes,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'coordinates': coordinates,
      'strategy': strategy,
      'distance': distance,
      'riskScore': riskScore,
      'riskLevel': riskLevel,
      'timestamp': timestamp.toIso8601String(),
      'durationMinutes': durationMinutes,
    };
  }

  factory RouteHistory.fromJson(Map<String, dynamic> json) {
    return RouteHistory(
      id: json['id'],
      coordinates: List<List<double>>.from(json['coordinates']),
      strategy: json['strategy'],
      distance: json['distance'].toDouble(),
      riskScore: json['riskScore'].toDouble(),
      riskLevel: json['riskLevel'],
      timestamp: DateTime.parse(json['timestamp']),
      durationMinutes: json['durationMinutes'],
    );
  }
}

class RouteAnalytics {
  static Future<List<RouteHistory>> getRouteHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final historyJson = prefs.getStringList('route_history') ?? [];
    
    return historyJson
        .map((json) => RouteHistory.fromJson(jsonDecode(json)))
        .toList()
      ..sort((a, b) => b.timestamp.compareTo(a.timestamp)); // Most recent first
  }

  static Future<void> saveRoute(RouteHistory route) async {
    final prefs = await SharedPreferences.getInstance();
    final history = await getRouteHistory();
    
    history.insert(0, route); // Add to beginning
    
    // Keep only last 50 routes
    if (history.length > 50) {
      history.removeRange(50, history.length);
    }
    
    final historyJson = history.map((r) => json.encode(r.toJson())).toList();
    await prefs.setStringList('route_history', historyJson);
  }

  static Map<String, dynamic> getSafetyStats(List<RouteHistory> routes) {
    if (routes.isEmpty) {
      return {
        'totalRoutes': 0,
        'avgRiskScore': 0.0,
        'totalDistance': 0.0,
        'safestRoute': null,
        'mostUsedStrategy': 'balanced',
      };
    }

    final totalDistance = routes.fold<double>(0, (sum, r) => sum + r.distance);
    final avgRiskScore = routes.fold<double>(0, (sum, r) => sum + r.riskScore) / routes.length;
    
    final strategyCounts = <String, int>{};
    for (final route in routes) {
      strategyCounts[route.strategy] = (strategyCounts[route.strategy] ?? 0) + 1;
    }
    
    final mostUsedStrategy = strategyCounts.entries
        .reduce((a, b) => a.value > b.value ? a : b)
        .key;
    
    final safestRoute = routes.reduce((a, b) => a.riskScore < b.riskScore ? a : b);

    return {
      'totalRoutes': routes.length,
      'avgRiskScore': avgRiskScore,
      'totalDistance': totalDistance,
      'safestRoute': safestRoute,
      'mostUsedStrategy': mostUsedStrategy,
      'strategiesUsed': strategyCounts.keys.toList(),
    };
  }
}

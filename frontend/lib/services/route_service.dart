import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';

class RouteService {
  static Future<List<List<LatLng>>> getRoutes(
      LatLng start, LatLng end) async {
    final url =
        'https://router.project-osrm.org/route/v1/driving/'
        '${start.longitude},${start.latitude};'
        '${end.longitude},${end.latitude}'
        '?overview=full&geometries=geojson&alternatives=true';

    final response = await http.get(Uri.parse(url));
    final data = json.decode(response.body);

    List<List<LatLng>> routes = [];

    for (final route in data['routes']) {
      final coords = route['geometry']['coordinates'];

      routes.add(
        coords
            .map<LatLng>((c) => LatLng(c[1], c[0]))
            .toList(),
      );
    }

    return routes;
  }

  static Future<List<LatLng>> getSnappedRoute(List<LatLng> waypoints) async {
    if (waypoints.isEmpty) return [];
    if (waypoints.length == 1) return waypoints;
    
    // OSRM demo server commonly limits to 100 waypoints. 
    // Take up to 25 evenly spaced waypoints to be safe and fast.
    List<LatLng> sampledWaypoints = [];
    if (waypoints.length <= 25) {
      sampledWaypoints = waypoints;
    } else {
      for (int i = 0; i < 25; i++) {
        int index = (i * (waypoints.length - 1) / 24).round();
        sampledWaypoints.add(waypoints[index]);
      }
    }

    final coordsString = sampledWaypoints
        .map((p) => '${p.longitude},${p.latitude}')
        .join(';');

    final url = 'https://router.project-osrm.org/route/v1/driving/'
        '$coordsString'
        '?overview=full&geometries=geojson';

    try {
      final response = await http.get(Uri.parse(url));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['routes'] != null && data['routes'].isNotEmpty) {
          final coords = data['routes'][0]['geometry']['coordinates'];
          return coords.map<LatLng>((c) => LatLng(c[1], c[0])).toList();
        }
      }
    } catch (e) {
      print('OSRM Routing Error: $e');
    }
    
    // Fallback to original waypoints if OSRM fails
    return waypoints;
  }
}

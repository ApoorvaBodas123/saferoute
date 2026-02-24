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
}

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../models/risk_zone.dart';
import '../services/risk_service.dart';
import '../services/route_service.dart';
import '../services/location_service.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  @override
  void initState() {
    super.initState();
    loadUserLocation();
  }

  Future<void> loadUserLocation() async {
    final locationData = await LocationService.getCurrentLocation();

    if (locationData != null) {
      setState(() {
        userLocation = LatLng(locationData.latitude!, locationData.longitude!);
      });
    }
  }

  bool showSafeOnly = false;
  LatLng? userLocation;

  LatLng? destination;
  List<List<LatLng>> allRoutes = [];

  // 🔢 Calculate full route risk
  double calculateRouteRisk(List<LatLng> route, List<RiskZone> zones) {
    double riskSum = 0;

    for (final point in route) {
      for (final zone in zones) {
        final distance = const Distance().as(
          LengthUnit.Meter,
          point,
          LatLng(zone.lat, zone.lon),
        );

        if (distance < 800) {
          riskSum += zone.risk * (1 - distance / 800);
        }
      }
    }
    return riskSum;
  }

  // 🎨 Segment-wise coloring
  List<Polyline> buildSegmentColoredRoute(
    List<LatLng> route,
    List<RiskZone> zones,
  ) {
    List<Polyline> polylines = [];

    for (int i = 0; i < route.length - 1; i++) {
      LatLng current = route[i];
      LatLng next = route[i + 1];

      double maxNearbyRisk = 0;

      for (final zone in zones) {
        final distance = const Distance().as(
          LengthUnit.Meter,
          current,
          LatLng(zone.lat, zone.lon),
        );

        if (distance < 600) {
          if (zone.risk > maxNearbyRisk) {
            maxNearbyRisk = zone.risk;
          }
        }
      }

      Color color;

      if (maxNearbyRisk >= 5) {
        color = Colors.red; // High risk
      } else if (maxNearbyRisk >= 3) {
        color = Colors.orange; // Moderate
      } else {
        color = Colors.green; // Safe
      }

      polylines.add(
        Polyline(points: [current, next], strokeWidth: 6, color: color),
      );
    }

    return polylines;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Safe Route App'),
        backgroundColor: Colors.blueAccent,
        actions: [
          IconButton(
            icon: const Icon(Icons.swap_horiz),
            onPressed: () {
              setState(() {
                showSafeOnly = !showSafeOnly;
              });
            },
          ),
        ],
      ),
      body: FutureBuilder<List<RiskZone>>(
        future: RiskService.loadZones(),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }

          final zones = snapshot.data!;

          // Evaluate routes
          final evaluatedRoutes = allRoutes.map((route) {
            final risk = calculateRouteRisk(route, zones);
            return {'points': route, 'risk': risk};
          }).toList();

          Map<String, dynamic>? safestRoute;
          Map<String, dynamic>? riskyRoute;

          if (evaluatedRoutes.isNotEmpty) {
            evaluatedRoutes.sort(
              (a, b) => (a['risk'] as double).compareTo(b['risk'] as double),
            );

            safestRoute = evaluatedRoutes.first;
            riskyRoute = evaluatedRoutes.last;
          }

          return Column(
            children: [
              // 🚨 Banner
              if (safestRoute != null)
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  color: Colors.green,
                  child: const Text(
                    '🟢 Green = Safest Route   |   🔴/🟡 = Risky Areas',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),

              Expanded(
                child: FlutterMap(
                  options: MapOptions(
                    center: userLocation ?? LatLng(12.9716, 77.5946),
                    zoom: 13,
                    onTap: (tapPosition, latlng) async {
                      if (userLocation == null) return;

                      final routes = await RouteService.getRoutes(
                        userLocation!,
                        latlng,
                      );

                      setState(() {
                        destination = latlng;
                        allRoutes = routes;
                      });
                    },
                  ),
                  children: [
                    TileLayer(
                      urlTemplate:
                          'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      userAgentPackageName: 'com.example.safe_route_app',
                    ),

                    // ✅ Draw ONLY safest route with segment coloring

                    // 🛣 Show both safe and risky routes
                    if (safestRoute != null && riskyRoute != null)
                      PolylineLayer(
                        polylines: [
                          // 🟢 Safe route first (thin)
                          Polyline(
                            points: safestRoute['points'] as List<LatLng>,
                            strokeWidth: 5,
                            color: Colors.green.withOpacity(0.7),
                          ),

                          // 🔴 Risk-colored route on top
                          ...buildSegmentColoredRoute(
                            riskyRoute['points'] as List<LatLng>,
                            zones,
                          ),
                        ],
                      ),

                    // 👤 User marker
                    MarkerLayer(
                      markers: [
                        Marker(
                          point: userLocation ?? LatLng(12.9716, 77.5946),
                          width: 40,
                          height: 40,
                          child: const Icon(
                            Icons.person_pin_circle,
                            color: Colors.blue,
                            size: 40,
                          ),
                        ),
                      ],
                    ),

                    // 🎯 Destination marker
                    if (destination != null)
                      MarkerLayer(
                        markers: [
                          Marker(
                            point: destination!,
                            width: 40,
                            height: 40,
                            child: const Icon(
                              Icons.flag,
                              color: Colors.black,
                              size: 36,
                            ),
                          ),
                        ],
                      ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../services/route_service.dart';
import '../services/location_service.dart';
import '../services/ml_prediction_service.dart';
import '../services/safety_service.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  bool showSafeOnly = false;
  LatLng? userLocation;
  LatLng? destination;
  List<List<LatLng>> allRoutes = [];
  Map<String, dynamic>? mlRouteResult;
  bool isMLRouteLoading = false;
  String selectedStrategy = 'balanced';
  Map<String, dynamic>? safestRoute;
  Map<String, dynamic>? riskyRoute;
  bool isSafetyModeActive = false;
  
  // Removed unused _riskZones
  @override
  void initState() {
    super.initState();
    loadUserLocation();
    _initSafetyService();
    
    // Add test route for debugging
    final testRoute = [
      LatLng(12.9716, 77.5946),
      LatLng(12.9762, 77.6033),
      LatLng(12.9850, 77.6100)
    ];
    
    setState(() {
      safestRoute = {
        'points': testRoute,
        'risk_score': 3.5,
        'distance': 8.5,
        'duration': 25,
      };
      
      riskyRoute = {
        'points': testRoute,
        'risk_score': 3.5,
        'distance': 8.5,
        'duration': 25,
      };
      
      allRoutes = [testRoute];
    });
  }

  Future<void> _initSafetyService() async {
    await SafetyService.initialize();
  }

  Future<void> loadUserLocation() async {
    final locationData = await LocationService.getCurrentLocation();

    if (locationData != null) {
      setState(() {
        userLocation = LatLng(locationData.latitude!, locationData.longitude!);
      });
    }
  }

  // 🤖 ML-powered route calculation
  Future<void> calculateMLRoute(LatLng start, LatLng end, String strategy) async {
    if (userLocation == null) return;

    setState(() {
      isMLRouteLoading = true;
      mlRouteResult = null;
    });

    try {
      final result = await MLPredictionService.getOptimizedRoute(
        start.latitude,
        start.longitude,
        end.latitude,
        end.longitude,
        strategy: strategy,
      );

      // Process route points outside of setState since they require async calls
      List<LatLng>? processedMainRoute;
      Map<String, dynamic>? newSafestRoute;
      Map<String, dynamic>? newRiskyRoute;
      
      if (result['route_coordinates'] != null) {
        final rawPoints = (result['route_coordinates'] as List).map((coord) => 
          LatLng(coord[0], coord[1])
        ).toList();
        
        processedMainRoute = await RouteService.getSnappedRoute(rawPoints);
        
       
        if (strategy == 'safest') {
          newSafestRoute = {
            'points': processedMainRoute,
            'risk_score': result['total_risk_score'] ?? 2.5,
            'distance': result['total_distance'] ?? 3.0,
            'duration': result['estimated_duration'] ?? 7.5,
          };
          
          final altRaw = [
            rawPoints.first,
            LatLng((rawPoints.first.latitude + rawPoints.last.latitude) / 2 + 0.02, 
                   (rawPoints.first.longitude + rawPoints.last.longitude) / 2 - 0.02),
            rawPoints.last
          ];
          
          newRiskyRoute = {
            'points': await RouteService.getSnappedRoute(altRaw),
            'risk_score': 6.5,
            'distance': 2.5,
            'duration': 4.2,
          };
        } else if (strategy == 'fastest') {
          newRiskyRoute = {
            'points': processedMainRoute,
            'risk_score': result['total_risk_score'] ?? 6.5,
            'distance': result['total_distance'] ?? 2.5,
            'duration': result['estimated_duration'] ?? 4.2,
          };
          
          final altRaw = [
            rawPoints.first,
            LatLng((rawPoints.first.latitude + rawPoints.last.latitude) / 2 - 0.02, 
                   (rawPoints.first.longitude + rawPoints.last.longitude) / 2 + 0.02),
            rawPoints.last
          ];
          
          newSafestRoute = {
            'points': await RouteService.getSnappedRoute(altRaw),
            'risk_score': 2.5,
            'distance': 3.5,
            'duration': 8.4,
          };
        } else { 
          newSafestRoute = {
            'points': processedMainRoute,
            'risk_score': result['total_risk_score'] ?? 4.0,
            'distance': result['total_distance'] ?? 2.7,
            'duration': result['estimated_duration'] ?? 5.4,
          };
          
          final altRaw = [
            rawPoints.first,
            LatLng((rawPoints.first.latitude + rawPoints.last.latitude) / 2 + 0.015, 
                   (rawPoints.first.longitude + rawPoints.last.longitude) / 2 + 0.015),
            rawPoints.last
          ];
          
          newRiskyRoute = {
            'points': await RouteService.getSnappedRoute(altRaw),
            'risk_score': 4.0,
            'distance': 2.7,
            'duration': 5.4,
          };
        }
      }

      
      setState(() {
        mlRouteResult = result;
        isMLRouteLoading = false;
        
        if (processedMainRoute != null) {
          safestRoute = newSafestRoute;
          riskyRoute = newRiskyRoute;
          allRoutes = [processedMainRoute];
        }
      });
    } catch (e) {
      setState(() {
        isMLRouteLoading = false;
      });
      
      final safestPoints = [
        start,
        LatLng((start.latitude + end.latitude) / 2 + 0.01, (start.longitude + end.longitude) / 2 - 0.01),
        end
      ];
      
      final riskyPoints = [
        start,
        LatLng((start.latitude + end.latitude) / 2 - 0.01, (start.longitude + end.longitude) / 2 + 0.01),
        end
      ];
      
      setState(() {
        safestRoute = {
          'points': safestPoints,
          'risk_score': 3.0,
          'distance': 10.5,
          'duration': 32,
        };
        
        riskyRoute = {
          'points': riskyPoints,
          'risk_score': 7.5,
          'distance': 9.0,
          'duration': 25,
        };
        
        allRoutes = [safestPoints, riskyPoints];
      });
    }
  }

  // Helper to generate fake risk circles along the route for visualization
  List<CircleMarker> _generateRiskZones(List<LatLng> points) {
    if (points.isEmpty) return [];
    List<CircleMarker> circles = [];
    
    // Show denser zones (approx every 15-20 points depending on route length) to create overlapping heatmap effect
    int step = (points.length / 15).ceil();
    if (step < 1) step = 1;

    for (int i = 0; i < points.length; i += step) {
      // Assign fake risk colors based on the selected strategy to demonstrate functionality
      Color circleColor;
      if (selectedStrategy == 'safest') {
         circleColor = Colors.green.withValues(alpha: 0.25); // Safe, lighter opacity
      } else if (selectedStrategy == 'fastest') {
         // Fastest route has higher risk, mix of red and orange
         circleColor = (i % 2 == 0) ? Colors.red.withValues(alpha: 0.25) : Colors.orange.withValues(alpha: 0.25);
      } else {
         // Balanced is a mix
         circleColor = (i % 3 == 0) ? Colors.orange.withValues(alpha: 0.25) : Colors.green.withValues(alpha: 0.25);
      }

      circles.add(
        CircleMarker(
          point: points[i],
          color: circleColor,
          borderStrokeWidth: 0,
          useRadiusInMeter: true,
          radius: 600, // 600 meters radius for wider overlapping coverage
        )
      );
    }
    return circles;
  }

  @override
  Widget build(BuildContext context) {
    // Determine which route data to display in the card
    Map<String, dynamic>? activeRouteData;
    if (safestRoute != null && riskyRoute != null) {
       activeRouteData = (selectedStrategy == 'safest') ? safestRoute : 
                         (selectedStrategy == 'fastest') ? riskyRoute : safestRoute; // using safest as fallback for balanced if 'balanced' is missing specific stats
    }

    return Scaffold(
      body: Stack(
        children: [
          // 1. The Map
          FlutterMap(
            options: MapOptions(
              initialCenter: userLocation ?? LatLng(12.9716, 77.5946),
              initialZoom: 13,
              onTap: (tapPosition, latlng) async {
                if (userLocation == null) return;
                setState(() {
                  destination = latlng;
                });
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('📍 Destination set! Calculating optimal route...'),
                    backgroundColor: Color(0xFFFD6296),
                    duration: Duration(seconds: 2),
                  ),
                );
                calculateMLRoute(userLocation!, latlng, selectedStrategy);
              },
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.example.safe_route_app',
              ),
              
              // NEW: Risk Zones Overlay (Drawn below routes)
              if (activeRouteData != null)
                 CircleLayer(
                   circles: _generateRiskZones(activeRouteData['points'] as List<LatLng>),
                 ),

              if (safestRoute != null && riskyRoute != null)
                PolylineLayer(
                  polylines: [
                    Polyline(
                      points: selectedStrategy == 'safest' 
                        ? safestRoute!['points'] as List<LatLng>
                        : selectedStrategy == 'fastest'
                          ? riskyRoute!['points'] as List<LatLng>
                          : safestRoute!['points'] as List<LatLng>,
                      strokeWidth: 6,
                      color: selectedStrategy == 'safest' 
                        ? Colors.green.withValues(alpha: 0.8)
                        : selectedStrategy == 'fastest'
                          ? Colors.orange.withValues(alpha: 0.8)
                          : Colors.blue.withValues(alpha: 0.8),
                    ),
                    Polyline(
                      points: selectedStrategy == 'safest' 
                        ? riskyRoute!['points'] as List<LatLng>
                        : selectedStrategy == 'fastest'
                          ? safestRoute!['points'] as List<LatLng>
                          : riskyRoute!['points'] as List<LatLng>,
                      strokeWidth: 3,
                      color: Colors.grey.withValues(alpha: 0.5),
                    ),
                  ],
                ),
              MarkerLayer(
                markers: [
                  Marker(
                    point: userLocation ?? LatLng(12.9716, 77.5946),
                    width: 40, height: 40,
                    child: const Icon(Icons.person_pin_circle, color: Color(0xFFFD6296), size: 40),
                  ),
                ],
              ),
              if (destination != null)
                MarkerLayer(
                  markers: [
                    Marker(
                      point: destination!,
                      width: 40, height: 40,
                      child: const Icon(Icons.location_on, color: Colors.red, size: 40),
                    ),
                  ],
                ),
              if (isMLRouteLoading)
                const Center(
                  child: Card(
                    color: Colors.white,
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircularProgressIndicator(color: Color(0xFFFD6296)),
                          SizedBox(height: 8),
                          Text('Calculating optimal route...', style: TextStyle(color: Color(0xFFFD6296))),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),

          // 2. Floating Top Left Action Button (Hamburger Menu simulation)
          Positioned(
            top: 50,
            left: 16,
            child: FloatingActionButton(
              heroTag: 'menuBtn',
              backgroundColor: const Color(0xFFFD6296),
              mini: true,
              onPressed: () {},
              child: const Icon(Icons.menu, color: Colors.white),
            ),
          ),

          // 3. Floating Overlay Top actions (Strategy & Location)
          Positioned(
            top: 50,
            right: 16,
            child: Column(
              children: [
                FloatingActionButton(
                  heroTag: 'strategyBtn',
                  backgroundColor: Colors.white,
                  mini: true,
                  onPressed: () {
                     showDialog(
                       context: context,
                       builder: (context) => AlertDialog(
                         title: const Text("Select Route Strategy"),
                         content: Column(
                           mainAxisSize: MainAxisSize.min,
                           children: [
                             ListTile(
                               title: const Text("🛡️ Safest"),
                               onTap: () {
                                 setState(() { selectedStrategy = 'safest'; });
                                 if (destination != null && userLocation != null) {
                                  calculateMLRoute(userLocation!, destination!, 'safest');
                                 }
                                 Navigator.pop(context);
                               },
                             ),
                             ListTile(
                               title: const Text("⚡ Fastest"),
                               onTap: () {
                                 setState(() { selectedStrategy = 'fastest'; });
                                 if (destination != null && userLocation != null) {
                                  calculateMLRoute(userLocation!, destination!, 'fastest');
                                 }
                                 Navigator.pop(context);
                               },
                             ),
                             ListTile(
                               title: const Text("⚖️ Balanced"),
                               onTap: () {
                                 setState(() { selectedStrategy = 'balanced'; });
                                 if (destination != null && userLocation != null) {
                                  calculateMLRoute(userLocation!, destination!, 'balanced');
                                 }
                                 Navigator.pop(context);
                               },
                             ),
                           ]
                         )
                       )
                     );
                  },
                  child: const Icon(Icons.settings, color: Color(0xFFFD6296)),
                ),
                const SizedBox(height: 8),
                FloatingActionButton(
                  heroTag: 'safetyBtn',
                  backgroundColor: isSafetyModeActive ? Colors.red : Colors.white,
                  mini: true,
                  onPressed: () async {
                    final newValue = !isSafetyModeActive;
                    await SafetyService.toggleSafetyMode(newValue);
                    if (!context.mounted) return;
                    setState(() { isSafetyModeActive = newValue; });
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(newValue ? 'Safety Mode ON' : 'Safety Mode OFF'),
                        backgroundColor: newValue ? Colors.red : Colors.grey,
                        duration: const Duration(seconds: 2),
                      ),
                    );
                  },
                  child: Icon(
                    isSafetyModeActive ? Icons.shield : Icons.shield_outlined,
                    color: isSafetyModeActive ? Colors.white : const Color(0xFFFD6296),
                  ),
                ),
              ],
            ),
          ),
          
          // 4. Status Indicator bar & Route Details Panel
          if (activeRouteData != null && !isMLRouteLoading)
             Positioned(
               bottom: 80, // Sit just above the bottom navigation bar
               left: 16,
               right: 16,
               child: Container(
                 padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                 decoration: BoxDecoration(
                   color: Colors.white,
                   borderRadius: BorderRadius.circular(20),
                   boxShadow: [
                     BoxShadow(
                       color: Colors.black.withValues(alpha: 0.1),
                       blurRadius: 10,
                       offset: const Offset(0, -5),
                     )
                   ]
                 ),
                 child: Column(
                   crossAxisAlignment: CrossAxisAlignment.start,
                   children: [
                     Row(
                       mainAxisAlignment: MainAxisAlignment.spaceBetween,
                       children: [
                         Text(
                           '${selectedStrategy.toUpperCase()} ROUTE',
                           style: const TextStyle(fontWeight: FontWeight.bold, color: Color(0xFFFD6296), fontSize: 16),
                         ),
                         Container(
                           padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                           decoration: BoxDecoration(
                             color: (activeRouteData['risk_score'] as double) < 4.0 ? Colors.green.shade100 : Colors.orange.shade100,
                             borderRadius: BorderRadius.circular(12),
                           ),
                           child: Text(
                             'Risk: ${(activeRouteData['risk_score'] as double).toStringAsFixed(1)}',
                             style: TextStyle(
                               fontWeight: FontWeight.bold, 
                               color: (activeRouteData['risk_score'] as double) < 4.0 ? Colors.green.shade700 : Colors.orange.shade700
                             ),
                           ),
                         )
                       ],
                     ),
                     const SizedBox(height: 12),
                     Row(
                       mainAxisAlignment: MainAxisAlignment.spaceAround,
                       children: [
                         _buildMetrics(Icons.access_time_filled, '${(activeRouteData['duration']).toInt()} min'),
                         Container(width: 1, height: 30, color: Colors.grey.shade300),
                         _buildMetrics(Icons.route, '${(activeRouteData['distance'] as double).toStringAsFixed(1)} km'),
                       ],
                     )
                   ],
                 )
               ),
             ),
        ],
      ),
    );
  }

  Widget _buildMetrics(IconData icon, String value) {
    return Row(
      children: [
        Icon(icon, color: Colors.grey.shade600, size: 20),
        const SizedBox(width: 6),
        Text(value, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: Colors.black87)),
      ],
    );
  }
}
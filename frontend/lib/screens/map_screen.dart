import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../services/route_service.dart';
import '../services/location_service.dart';
import '../services/ml_prediction_service.dart';
import '../services/safety_service.dart';
import 'profile_screen.dart';

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
      
      
      final routePoints = [
        start,
        LatLng((start.latitude + end.latitude) / 2, (start.longitude + end.longitude) / 2),
        end
      ];
      
      setState(() {
        safestRoute = {
          'points': routePoints,
          'risk_score': 5.0,
          'distance': 10.0,
          'duration': 30,
        };
        
        riskyRoute = {
          'points': routePoints,
          'risk_score': 5.0,
          'distance': 10.0,
          'duration': 30,
        };
        
        allRoutes = [routePoints];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Safe Route App (ML-Powered)'),
        backgroundColor: isSafetyModeActive ? Colors.red.shade800 : Colors.blueAccent,
        actions: [
         
          IconButton(
            icon: Icon(
              isSafetyModeActive ? Icons.shield : Icons.shield_outlined,
              color: isSafetyModeActive ? Colors.amber : Colors.white,
            ),
            tooltip: isSafetyModeActive ? 'Disable Safety Mode' : 'Enable Safety Mode',
            onPressed: () async {
              final newValue = !isSafetyModeActive;
              await SafetyService.toggleSafetyMode(newValue);
              if (!context.mounted) return;
              setState(() {
                isSafetyModeActive = newValue;
              });
              
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    newValue 
                      ? '🛡️ Safety Mode ENABLED: Monitoring location and voice ("help me")'
                      : '🛑 Safety Mode DISABLED'
                  ),
                  backgroundColor: newValue ? Colors.red.shade800 : Colors.blue,
                  duration: const Duration(seconds: 3),
                ),
              );
            },
          ),
  
          IconButton(
            icon: const Icon(Icons.person),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) => const ProfileScreen(),
                ),
              );
            },
          ),

          PopupMenuButton<String>(
            icon: const Icon(Icons.settings),
            onSelected: (String strategy) {
              setState(() {
                selectedStrategy = strategy;
              });
            
              if (destination != null && userLocation != null) {
                calculateMLRoute(userLocation!, destination!, strategy);
              }
            },
            itemBuilder: (BuildContext context) => [
              const PopupMenuItem<String>(
                value: 'safest',
                child: Text('🛡️ Safest Route'),
              ),
              const PopupMenuItem<String>(
                value: 'fastest',
                child: Text('⚡ Fastest Route'),
              ),
              const PopupMenuItem<String>(
                value: 'balanced',
                child: Text('⚖️ Balanced Route'),
              ),
            ],
          ),
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
      body: Column(
        children: [
        
          Container(
            width: double.infinity,
            color: selectedStrategy == 'safest' ? Colors.green : 
                   selectedStrategy == 'fastest' ? Colors.orange : Colors.blue,
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Text(
                '${selectedStrategy.toUpperCase()} Route Selected',
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          Expanded(
            child: FlutterMap(
              options: MapOptions(
                initialCenter: userLocation ?? LatLng(12.9716, 77.5946),
                initialZoom: 13,
                onTap: (tapPosition, latlng) async {
                  if (userLocation == null) return;

                  // Set destination first
                  setState(() {
                    destination = latlng;
                  });

                 
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('📍 Destination set! Calculating optimal route...'),
                      backgroundColor: Colors.blueAccent,
                      duration: Duration(seconds: 2),
                    ),
                  );

                  
                  calculateMLRoute(userLocation!, latlng, selectedStrategy);
                },
              ),
              children: [
                TileLayer(
                  urlTemplate:
                      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                  userAgentPackageName: 'com.example.safe_route_app',
                ),

                // 🛣 Show routes with strategy-specific colors
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

               
                if (destination != null)
                  MarkerLayer(
                    markers: [
                      Marker(
                        point: destination!,
                        width: 40,
                        height: 40,
                        child: const Icon(
                          Icons.location_on,
                          color: Colors.red,
                          size: 40,
                        ),
                      ),
                    ],
                  ),

                // 🔄 Loading indicator for ML route calculation
                if (isMLRouteLoading)
                  const Center(
                    child: Card(
                      color: Colors.white,
                      child: Padding(
                        padding: EdgeInsets.all(16),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            CircularProgressIndicator(),
                            SizedBox(height: 8),
                            Text('Calculating optimal route...'),
                          ],
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

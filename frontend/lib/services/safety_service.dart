import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:location/location.dart';
import 'package:latlong2/latlong.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'ml_prediction_service.dart';
import 'emergency_service.dart';

class SafetyService {
  static final Location _location = Location();
  static final SpeechToText _speechToText = SpeechToText();
  static final FlutterLocalNotificationsPlugin _notificationsPlugin = FlutterLocalNotificationsPlugin();
  
  // Emergency popup overlay
  static OverlayEntry? _emergencyOverlay;
  static bool _isEmergencyOverlayShowing = false;
  static BuildContext? _context;
  
  static bool _isActive = false;
  static bool _isListening = false;
  static StreamSubscription<LocationData>? _locationSubscription;
  static LatLng? _currentLocation;
  
  // Emergency cooldown to prevent multiple triggers
  static DateTime? _lastEmergencyTrigger;
  static const Duration _emergencyCooldown = Duration(minutes: 1);
  
  // The secret phrase to trigger the SOS sequence
  static const String secretPhrase = "help me";
  // The threshold above which a location is considered "unsafe"
  static const double highRiskThreshold = 7.0;

  static bool get isActive => _isActive;

  static Future<void> initialize() async {
    // Initialize notifications
    const AndroidInitializationSettings initializationSettingsAndroid =
        AndroidInitializationSettings('@mipmap/ic_launcher');
    const DarwinInitializationSettings initializationSettingsIOS = DarwinInitializationSettings();
    const InitializationSettings initializationSettings = InitializationSettings(
      android: initializationSettingsAndroid,
      iOS: initializationSettingsIOS,
    );
    await _notificationsPlugin.initialize(
      settings: initializationSettings,
      onDidReceiveNotificationResponse: (details) {},
    );
    
    // Initialize speech recognition
    await _speechToText.initialize(
      onError: (val) => print('🎙️ Speech Error: ${val.errorMsg}'),
      onStatus: (val) {
        if (val == 'done' && _isActive) {
          // Restart listening loop if it stops while safety mode is active
          startListeningLoop();
        }
      },
    );
  }

  static Future<void> toggleSafetyMode(bool activate) async {
    _isActive = activate;
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('safety_mode_active', activate);

    if (activate) {
      print('Safety Mode ENABLED');
      await _startLocationTracking();
      await startListeningLoop();
      _showNotification(
        'Safety Mode Active', 
        'Monitoring your surroundings and listening for emergency phrase.'
      );
    } else {
      print(' Safety Mode DISABLED');
      print('🛑 Safety Mode DISABLED');
      await _stopLocationTracking();
      await _stopListeningLoop();
      _showNotification(
        'Safety Mode Disabled', 
        'Background monitoring has been stopped.'
      );
    }
  }

  static Future<void> _startLocationTracking() async {
    // Request permissions
    bool serviceEnabled = await _location.serviceEnabled();
    if (!serviceEnabled) {
      serviceEnabled = await _location.requestService();
      if (!serviceEnabled) return;
    }

    PermissionStatus permissionGranted = await _location.hasPermission();
    if (permissionGranted == PermissionStatus.denied) {
      permissionGranted = await _location.requestPermission();
      if (permissionGranted != PermissionStatus.granted) return;
    }

    // Configure background tracking
    try {
      await _location.enableBackgroundMode(enable: true);
    } catch (e) {
      print('Background location mode not supported on this platform: $e');
    }
    
    // Only fetch risk score every ~100 meters or 30 seconds to save battery/API calls
    await _location.changeSettings(
      accuracy: LocationAccuracy.high,
      interval: 30000,
      distanceFilter: 100,
    );

    _locationSubscription = _location.onLocationChanged.listen((LocationData loc) async {
      if (loc.latitude != null && loc.longitude != null && _isActive) {
        _currentLocation = LatLng(loc.latitude!, loc.longitude!);
        
        try {
          // Check risk silently in background
          final riskData = await MLPredictionService.getDynamicRiskScore(
            loc.latitude!, 
            loc.longitude!
          );
          
          final riskScore = riskData['current_risk_score'] as double;
          
          if (riskScore >= highRiskThreshold) {
            _showNotification(
              '⚠️ High Risk Area Detected', 
              'Risk Score: ${riskScore.toStringAsFixed(1)}. Stay alert and keep moving.'
            );
          }
        } catch (e) {
          print('Error checking background risk: $e');
        }
      }
    });
  }

  static Future<void> _stopLocationTracking() async {
    await _locationSubscription?.cancel();
    _locationSubscription = null;
    try {
      await _location.enableBackgroundMode(enable: false);
    } catch (e) {
      // Background mode not supported on all platforms via location package
    }
  }

  static Future<void> startListeningLoop() async {
    // Reset listening state to allow new listening session
    _isListening = false;
    
    if (!_isActive) return;
    
    // Force initialize speech
    bool initialized = await _speechToText.initialize();
    
    if (_speechToText.isAvailable) {
      _isListening = true;
      try {
        await _speechToText.listen(
          onResult: (result) async {
            final recognizedWords = result.recognizedWords.toLowerCase();
            
            if (recognizedWords.contains(secretPhrase)) {
              // Check cooldown to prevent multiple triggers
              final now = DateTime.now();
              if (_lastEmergencyTrigger != null && 
                  now.difference(_lastEmergencyTrigger!) < _emergencyCooldown) {
                return;
              }
              
              _lastEmergencyTrigger = now;
              _stopListeningLoop();
              _showEmergencyDetectedFeedback();
              
              if (_currentLocation != null) {
                await EmergencyService.triggerEmergencySequence(_currentLocation!);
              } else {
                await EmergencyService.triggerEmergencySequence(const LatLng(12.9716, 77.5946));
              }
            }
          },
          listenFor: const Duration(seconds: 30),
          pauseFor: const Duration(seconds: 2),
          listenOptions: SpeechListenOptions(
            partialResults: true,
            cancelOnError: false,
            listenMode: ListenMode.dictation,
          ),
        );
      } catch (e) {
        _isListening = false;
      }
    }
  }

  static Future<void> _stopListeningLoop() async {
    if (_isListening) {
      await _speechToText.stop();
      _isListening = false;
    }
  }

  
  static Future<void> _showEmergencyDetectedFeedback() async {
    // 1. Show emergency popup
    _showEmergencyPopup();
    
    // 2. Immediate vibration feedback
    await _triggerVibration();
    
    // 3. Show emergency notification
    await _showEmergencyNotification();
    
    // 4. Flash the screen briefly (subtle indicator)
    await _flashScreenIndicator();
  }

  static void setContext(BuildContext context) {
    _context = context;
  }

  static void _showEmergencyPopup() {
    if (_isEmergencyOverlayShowing || _context == null) return;
    _isEmergencyOverlayShowing = true;
    
    final overlay = Overlay.of(_context!);
    _emergencyOverlay = OverlayEntry(
      builder: (context) => Positioned(
        top: 100,
        left: 20,
        right: 20,
        child: Material(
          elevation: 8,
          borderRadius: BorderRadius.circular(12),
          color: Colors.red,
          child: Container(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.emergency, color: Colors.white, size: 32),
                    SizedBox(width: 12),
                    Text(
                      'EMERGENCY!',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                const Text(
                  'Emergency phrase detected\nLocation sent to emergency contacts',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: () => hideEmergencyPopup(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.red,
                  ),
                  child: const Text('OK'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
    
    overlay.insert(_emergencyOverlay!);
    print("EMERGENCY: Showing emergency popup overlay");
  }

  static void hideEmergencyPopup() {
    _emergencyOverlay?.remove();
    _emergencyOverlay = null;
    _isEmergencyOverlayShowing = false;
    print("EMERGENCY: Emergency popup hidden");
  }

  static Future<void> _triggerVibration() async {
    try {
      // Vibrate pattern: short-long-short to indicate emergency detected
      // Note: This would require the 'vibration' package to be added
      print("Vibration: Emergency phrase detected!");
    } catch (e) {
      print("Vibration not available: $e");
    }
  }

  static Future<void> _showEmergencyNotification() async {
    // Handle web compatibility for vibration pattern
    AndroidNotificationDetails androidPlatformChannelSpecifics;
    
    try {
      androidPlatformChannelSpecifics = AndroidNotificationDetails(
        'emergency_detected', 
        'Emergency Detected',
        channelDescription: 'Emergency phrase detection alerts',
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'Emergency phrase detected',
        sound: RawResourceAndroidNotificationSound('notification'),
        enableVibration: true,
        playSound: true,
        vibrationPattern: Int64List.fromList([0, 200, 100, 200]),
      );
    } catch (e) {
      // Fallback for web where Int64List is not supported
      androidPlatformChannelSpecifics = AndroidNotificationDetails(
        'emergency_detected', 
        'Emergency Detected',
        channelDescription: 'Emergency phrase detection alerts',
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'Emergency phrase detected',
        sound: RawResourceAndroidNotificationSound('notification'),
        enableVibration: false, // Disable vibration on web
        playSound: true,
      );
    }
    final NotificationDetails platformChannelSpecifics =
        NotificationDetails(android: androidPlatformChannelSpecifics);
        
    await _notificationsPlugin.show(
      id: 999, // Unique ID for emergency detection
      title: 'EMERGENCY DETECTED',
      body: 'Emergency phrase recognized. Location sent to emergency contacts.',
      notificationDetails: platformChannelSpecifics,
    );
  }

  static Future<void> _flashScreenIndicator() async {
    // This would require additional UI integration to flash the screen
    // For now, we'll just log it
    print("Screen flash: Emergency detected indicator");
  }

  static Future<void> _showNotification(String title, String body) async {
    const AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
      'safety_alerts', 
      'Safety Alerts',
      channelDescription: 'Important safety notifications and alerts',
      importance: Importance.max,
      priority: Priority.high,
      ticker: 'ticker',
    );
    const NotificationDetails platformChannelSpecifics =
        NotificationDetails(android: androidPlatformChannelSpecifics);
        
    await _notificationsPlugin.show(
      id: 0,
      title: title,
      body: body,
      notificationDetails: platformChannelSpecifics,
    );
  }
}

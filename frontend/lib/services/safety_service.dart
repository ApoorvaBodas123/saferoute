import 'dart:async';
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
  
  static bool _isActive = false;
  static bool _isListening = false;
  static StreamSubscription<LocationData>? _locationSubscription;
  static LatLng? _currentLocation;
  
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
          _startListeningLoop();
        }
      },
    );
  }

  static Future<void> toggleSafetyMode(bool activate) async {
    _isActive = activate;
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('safety_mode_active', activate);

    if (activate) {
      print('🛡️ Safety Mode ENABLED');
      await _startLocationTracking();
      await _startListeningLoop();
      _showNotification(
        'Safety Mode Active', 
        'Monitoring your surroundings and listening for emergency phrase.'
      );
    } else {
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

  static Future<void> _startListeningLoop() async {
    if (!_isActive || _isListening) return;
    
    if (_speechToText.isAvailable) {
      _isListening = true;
      try {
        await _speechToText.listen(
          onResult: (result) {
            final recognizedWords = result.recognizedWords.toLowerCase();
            print("🎙️ Heard: \$recognizedWords");
            
            if (recognizedWords.contains(secretPhrase)) {
              print("🚨 Secret phrase detected!");
              _stopListeningLoop();
              if (_currentLocation != null) {
                EmergencyService.triggerEmergencySequence(_currentLocation!);
              } else {
                // Trigger without precise location if necessary
                 EmergencyService.triggerEmergencySequence(const LatLng(0,0));
              }
            }
          },
          listenFor: const Duration(seconds: 30),
          pauseFor: const Duration(seconds: 5),
          listenOptions: SpeechListenOptions(
            partialResults: true,
            cancelOnError: true,
            listenMode: ListenMode.dictation,
          ),
        );
      } catch (e) {
        print("🎙️ Error starting listener: \$e");
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

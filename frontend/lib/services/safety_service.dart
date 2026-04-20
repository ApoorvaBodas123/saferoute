import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:latlong2/latlong.dart';
import 'package:location/location.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'emergency_service.dart';
import 'ml_prediction_service.dart';
import 'safety_settings_service.dart';

class EmergencyDetectionEvent {
  const EmergencyDetectionEvent({
    required this.transcript,
    required this.triggerPhrase,
    required this.timestamp,
  });

  final String transcript;
  final String triggerPhrase;
  final DateTime timestamp;
}

class SafetyService {
  static final Location _location = Location();
  static final SpeechToText _speechToText = SpeechToText();
  static final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  static final StreamController<EmergencyDetectionEvent> _eventController =
      StreamController<EmergencyDetectionEvent>.broadcast();

  static bool _isActive = false;
  static bool _isListening = false;
  static bool _speechReady = false;
  static bool _isEmergencyFlowActive = false;
  static Timer? _restartTimer;
  static StreamSubscription<LocationData>? _locationSubscription;
  static LatLng? _currentLocation;
  static String _secretPhrase = SafetySettingsService.defaultSecretPhrase;

  static const double highRiskThreshold = 7.0;
  static const Duration _emergencyCooldown = Duration(seconds: 20);
  static DateTime? _lastEmergencyTrigger;
  static const List<String> _fallbackEmergencyPhrases = <String>[
    'help me',
    'did you feed the dog',
    'save me',
    'emergency',
  ];

  static bool get isActive => _isActive;
  static bool get isListening => _isListening;
  static String get secretPhrase => _secretPhrase;
  static Stream<EmergencyDetectionEvent> get emergencyEvents =>
      _eventController.stream;

  static Future<void> initialize() async {
    const androidInit = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosInit = DarwinInitializationSettings();
    const settings = InitializationSettings(android: androidInit, iOS: iosInit);

    await _notificationsPlugin.initialize(
      settings: settings,
      onDidReceiveNotificationResponse: (_) {},
    );

    _secretPhrase = await SafetySettingsService.getSecretPhrase();
    await _initializeSpeechEngine();
  }

  static Future<void> refreshSettings() async {
    _secretPhrase = await SafetySettingsService.getSecretPhrase();
  }

  static Future<void> _initializeSpeechEngine() async {
    _speechReady = await _speechToText.initialize(
      onError: (val) => print('Speech error: ${val.errorMsg}'),
      onStatus: (status) {
        if (status == 'done' || status == 'notListening') {
          _isListening = false;
        }
        if ((status == 'done' || status == 'notListening') && _isActive) {
          _restartTimer?.cancel();
          _restartTimer = Timer(const Duration(milliseconds: 300), () {
            startListeningLoop();
          });
        }
      },
    );

    if (!_speechReady) {
      print('Speech recognition unavailable in this session.');
    }
  }

  static Future<void> toggleSafetyMode(bool activate) async {
    _isActive = activate;

    if (activate) {
      await refreshSettings();
      if (!_speechReady) {
        await _initializeSpeechEngine();
      }
      await _startLocationTracking();
      await startListeningLoop();
      await _showNotification(
        'Safety Mode Active',
        'Monitoring surroundings and listening for your secret phrase.',
      );
    } else {
      await _stopLocationTracking();
      await _stopListeningLoop();
      await _showNotification(
        'Safety Mode Disabled',
        'Background monitoring has been stopped.',
      );
    }
  }

  static Future<void> _startLocationTracking() async {
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

    try {
      await _location.enableBackgroundMode(enable: true);
    } catch (_) {}

    await _location.changeSettings(
      accuracy: LocationAccuracy.high,
      interval: 30000,
      distanceFilter: 100,
    );

    _locationSubscription =
        _location.onLocationChanged.listen((LocationData loc) async {
      if (loc.latitude == null || loc.longitude == null || !_isActive) return;

      _currentLocation = LatLng(loc.latitude!, loc.longitude!);

      try {
        final riskData =
            await MLPredictionService.getDynamicRiskScore(loc.latitude!, loc.longitude!);
        final riskScore = (riskData['current_risk_score'] as num?)?.toDouble() ?? 0;

        if (riskScore >= highRiskThreshold) {
          await _showNotification(
            'High Risk Area Detected',
            'Risk score ${riskScore.toStringAsFixed(1)}. Stay alert.',
          );
        }
      } catch (e) {
        print('Background risk check failed: $e');
      }
    });
  }

  static Future<void> _stopLocationTracking() async {
    await _locationSubscription?.cancel();
    _locationSubscription = null;

    try {
      await _location.enableBackgroundMode(enable: false);
    } catch (_) {}
  }

  static Future<void> startListeningLoop() async {
    if (!_isActive || _isListening || !_speechReady || _isEmergencyFlowActive) {
      return;
    }
    if (!_speechToText.isAvailable) return;
    if (_speechToText.isListening) {
      _isListening = true;
      return;
    }

    _isListening = true;

    try {
      await _speechToText.listen(
        onResult: (result) {
          final transcript = result.recognizedWords.toLowerCase().trim();
          if (transcript.isEmpty) return;

          print('Heard transcript: $transcript');

          if (_isEmergencyFlowActive) return;

          if (_containsEmergencyPhrase(transcript)) {
            final now = DateTime.now();
            if (_lastEmergencyTrigger != null &&
                now.difference(_lastEmergencyTrigger!) < _emergencyCooldown) {
              return;
            }

            _lastEmergencyTrigger = now;
            _isEmergencyFlowActive = true;

            _eventController.add(
              EmergencyDetectionEvent(
                transcript: transcript,
                triggerPhrase: _secretPhrase,
                timestamp: now,
              ),
            );

            _showEmergencyDetectedFeedback();
            _stopListeningLoop();

            final location = _currentLocation ?? const LatLng(12.9716, 77.5946);
            EmergencyService.triggerEmergencySequence(
              location,
              detectedTranscript: transcript,
              triggerPhrase: _secretPhrase,
            ).whenComplete(() {
              _isEmergencyFlowActive = false;
              if (_isActive) {
                startListeningLoop();
              }
            });
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
      print('Listener start failed: $e');
      _isListening = _speechToText.isListening;

      // Web speech engines can throw if restart races with browser internals.
      if (_isActive && !_speechToText.isListening) {
        _restartTimer?.cancel();
        _restartTimer = Timer(const Duration(seconds: 1), () {
          startListeningLoop();
        });
      }
    }
  }

  static bool _containsEmergencyPhrase(String transcript) {
    final normalizedTranscript = transcript
        .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();

    if (normalizedTranscript.isEmpty) return false;

    final candidatePhrases = <String>{
      _secretPhrase,
      ..._fallbackEmergencyPhrases,
    }
        .map(
          (p) => p
              .toLowerCase()
              .replaceAll(RegExp(r'[^a-z0-9\s]'), ' ')
              .replaceAll(RegExp(r'\s+'), ' ')
              .trim(),
        )
        .where((p) => p.isNotEmpty)
        .toList();

    for (final phrase in candidatePhrases) {
      if (normalizedTranscript.contains(phrase)) {
        return true;
      }
    }

    return false;
  }

  static Future<void> _stopListeningLoop() async {
    if (_isListening) {
      await _speechToText.stop();
      _isListening = false;
    }
    _restartTimer?.cancel();
  }

  static Future<void> _showEmergencyDetectedFeedback() async {
    await _showEmergencyNotification();
  }

  static Future<void> _showEmergencyNotification() async {
    AndroidNotificationDetails androidNotification;
    if (kIsWeb) {
      androidNotification = const AndroidNotificationDetails(
        'emergency_detected',
        'Emergency Detected',
        channelDescription: 'Emergency phrase detection alerts',
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'Emergency phrase detected',
        enableVibration: false,
        playSound: true,
      );
    } else {
      androidNotification = AndroidNotificationDetails(
        'emergency_detected',
        'Emergency Detected',
        channelDescription: 'Emergency phrase detection alerts',
        importance: Importance.max,
        priority: Priority.high,
        ticker: 'Emergency phrase detected',
        enableVibration: true,
        playSound: true,
        vibrationPattern: Int64List.fromList(<int>[0, 200, 100, 200]),
      );
    }

    final details = NotificationDetails(android: androidNotification);

    await _notificationsPlugin.show(
      id: 999,
      title: 'Emergency Phrase Detected',
      body: 'Triggering SOS sequence now.',
      notificationDetails: details,
    );
  }

  static Future<void> _showNotification(String title, String body) async {
    const androidNotification = AndroidNotificationDetails(
      'safety_alerts',
      'Safety Alerts',
      channelDescription: 'Important safety notifications and alerts',
      importance: Importance.max,
      priority: Priority.high,
      ticker: 'safety-alert',
    );
    const details = NotificationDetails(android: androidNotification);

    await _notificationsPlugin.show(
      id: 0,
      title: title,
      body: body,
      notificationDetails: details,
    );
  }
}

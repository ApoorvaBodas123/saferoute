import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_callkit_incoming/entities/call_kit_params.dart';
import 'package:flutter_callkit_incoming/flutter_callkit_incoming.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import 'package:latlong2/latlong.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:uuid/uuid.dart';

import 'api_service.dart';
import 'auth_service.dart';
import 'safety_settings_service.dart';

class EmergencyService {
  static final _audioRecorder = AudioRecorder();
  static final FlutterLocalNotificationsPlugin _notificationsPlugin =
      FlutterLocalNotificationsPlugin();

  static bool _isLocationShared = false;
  static bool _isAudioRecording = false;
  static bool _isFakeCallTriggered = false;
  static String? _audioFilePath;
  static String? _lastLocationShareTime;
  static bool _isEmergencyInProgress = false;

  static bool get isLocationShared => _isLocationShared;
  static bool get isAudioRecording => _isAudioRecording;
  static bool get isFakeCallTriggered => _isFakeCallTriggered;
  static String? get audioFilePath => _audioFilePath;
  static String? get lastLocationShareTime => _lastLocationShareTime;
  static String get _mlBaseUrl =>
      dotenv.env['ML_API_BASE_URL'] ?? 'http://localhost:8000';

  static Future<void> triggerEmergencySequence(
    LatLng currentLocation, {
    required String detectedTranscript,
    required String triggerPhrase,
  }) async {
    if (_isEmergencyInProgress) {
      return;
    }

    _isEmergencyInProgress = true;
    _isLocationShared = false;
    _isAudioRecording = false;
    _isFakeCallTriggered = false;

    try {
      final contacts = await SafetySettingsService.getEmergencyContacts();
      final aiCoverConversation = _buildAiCoverConversation();

      await _sendEmergencyAlert(
        currentLocation,
        detectedTranscript,
        triggerPhrase,
        contacts,
      );
      await _startHiddenAudioRecording();
      await _triggerFakeCall(aiCoverConversation);
      await _showEmergencyStatusNotification(aiCoverConversation);

      final currentUser = await AuthService.getCurrentUser();
      if (currentUser != null) {
        final emergencyLog = {
          'user_email': currentUser['email'],
          'user_name': currentUser['name'],
          'location': {
            'latitude': currentLocation.latitude,
            'longitude': currentLocation.longitude,
          },
          'actions_triggered': {
            'location_shared': _isLocationShared,
            'audio_recording': _isAudioRecording,
            'fake_call': _isFakeCallTriggered,
          },
          'detected_transcript': detectedTranscript,
          'trigger_phrase': triggerPhrase,
          'contacts_count': contacts.length,
          'ai_cover_conversation': aiCoverConversation,
        };

        try {
          await ApiService.createEmergencyLog(emergencyLog);
        } catch (e) {
          print('Failed to save emergency log: $e');
        }
      }
    } finally {
      _isEmergencyInProgress = false;
    }
  }

  static List<String> _buildAiCoverConversation() {
    // Lightweight generated conversation text used as a believable cover call flow.
    return <String>[
      'Hey, I am outside your place. Are you still coming?',
      'I brought your charger and the documents you asked for.',
      'Share your exact location, I will pick you up in five minutes.',
      'Stay on the line with me until you reach the main road.',
    ];
  }

  static Future<void> _showEmergencyStatusNotification(List<String> aiScript) async {
    const androidDetails = AndroidNotificationDetails(
      'emergency_status',
      'Emergency Status',
      channelDescription: 'Shows status of emergency actions',
      importance: Importance.max,
      priority: Priority.high,
      ticker: 'Emergency actions triggered',
    );
    const details = NotificationDetails(android: androidDetails);

    final summary = [
      'Location Shared: ${_isLocationShared ? 'YES' : 'NO'}',
      'Audio Recording: ${_isAudioRecording ? 'ACTIVE' : 'UNAVAILABLE'}',
      'Fake Call: ${_isFakeCallTriggered ? 'TRIGGERED' : 'UNAVAILABLE'}',
      'Cover Script: ${aiScript.first}',
    ].join(' | ');

    await _notificationsPlugin.show(
      id: 888,
      title: 'Emergency Status Report',
      body: summary,
      notificationDetails: details,
    );
  }

  static String getEmergencyStatus() {
    return '''
Emergency Status Report
=====================
Location Shared: ${_isLocationShared ? 'YES' : 'NO'}
Audio Recording: ${_isAudioRecording ? 'ACTIVE' : 'STOPPED'}
Fake Call: ${_isFakeCallTriggered ? 'TRIGGERED' : 'NOT TRIGGERED'}
Audio File Path: ${_audioFilePath ?? 'Not available'}
Last Location Share: ${_lastLocationShareTime ?? 'Never'}
''';
  }

  static Future<void> _sendEmergencyAlert(
    LatLng location,
    String detectedTranscript,
    String triggerPhrase,
    List<EmergencyContact> contacts,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse('$_mlBaseUrl/api/emergency-trigger'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'latitude': location.latitude,
              'longitude': location.longitude,
              'user_id': 'guardian_user_01',
              'timestamp': DateTime.now().toIso8601String(),
              'detected_transcript': detectedTranscript,
              'trigger_phrase': triggerPhrase,
              'emergency_contacts': contacts.map((c) => c.toJson()).toList(),
            }),
          )
          .timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        _isLocationShared = true;
        _lastLocationShareTime = DateTime.now().toIso8601String();
      } else {
        _isLocationShared = false;
      }
    } catch (e) {
      _isLocationShared = false;
      print('Error sharing emergency location: $e');
    }
  }

  static Future<void> _startHiddenAudioRecording() async {
    if (kIsWeb) {
      // This plugin path is unsupported on web.
      _isAudioRecording = false;
      return;
    }

    try {
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        _isAudioRecording = false;
        return;
      }

      if (await _audioRecorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        final fileName =
            'emergency_audio_${DateTime.now().millisecondsSinceEpoch}.m4a';
        final filePath = '${dir.path}/$fileName';

        await _audioRecorder.start(
          const RecordConfig(encoder: AudioEncoder.aacLc),
          path: filePath,
        );

        _isAudioRecording = true;
        _audioFilePath = filePath;

        Future.delayed(const Duration(minutes: 2), () async {
          if (await _audioRecorder.isRecording()) {
            await _audioRecorder.stop();
            _isAudioRecording = false;
          }
        });
      }
    } catch (e) {
      _isAudioRecording = false;
      print('Error starting audio recording: $e');
    }
  }

  static Future<void> _triggerFakeCall(List<String> aiScript) async {
    final isMobile = !kIsWeb &&
        (defaultTargetPlatform == TargetPlatform.android ||
            defaultTargetPlatform == TargetPlatform.iOS);

    if (!isMobile) {
      _isFakeCallTriggered = false;
      return;
    }

    try {
      await FlutterCallkitIncoming.showCallkitIncoming(
        CallKitParams(
          id: const Uuid().v4(),
          nameCaller: 'Aunt Priya',
          appName: 'SafeRoute',
          avatar: '',
          handle: 'Incoming',
          type: 0,
          textAccept: 'Accept',
          textDecline: 'Decline',
          duration: 30000,
          extra: <String, dynamic>{
            'coverConversation': aiScript,
          },
        ),
      );

      _isFakeCallTriggered = true;
    } catch (e) {
      _isFakeCallTriggered = false;
      print('Error triggering fake call: $e');
    }
  }
}

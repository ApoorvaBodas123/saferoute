import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_callkit_incoming/flutter_callkit_incoming.dart';
import 'package:flutter_callkit_incoming/entities/call_kit_params.dart';
import 'package:uuid/uuid.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'api_service.dart';
import 'package:latlong2/latlong.dart';
import 'auth_service.dart';

class EmergencyService {
  static const String _baseUrl = 'http://localhost:8000';
  static final _audioRecorder = AudioRecorder();
  static final FlutterLocalNotificationsPlugin _notificationsPlugin = FlutterLocalNotificationsPlugin();
  
  // Emergency status tracking
  static bool _isLocationShared = false;
  static bool _isAudioRecording = false;
  static bool _isFakeCallTriggered = false;
  static String? _audioFilePath;
  static String? _lastLocationShareTime;
  
  // Getters for status checking
  static bool get isLocationShared => _isLocationShared;
  static bool get isAudioRecording => _isAudioRecording;
  static bool get isFakeCallTriggered => _isFakeCallTriggered;
  static String? get audioFilePath => _audioFilePath;
  static String? get lastLocationShareTime => _lastLocationShareTime;
  
  /// Triggers the full emergency sequence discreetly
  static Future<void> triggerEmergencySequence(LatLng currentLocation) async {
    print('SECRET PHRASE DETECTED -> INITIATING SOS SEQUENCE');
    
    // Reset status trackers
    _isLocationShared = false;
    _isAudioRecording = false;
    _isFakeCallTriggered = false;
    
    // 1. Send live location to backend
    await _sendEmergencyAlert(currentLocation);
    
    // 2. Start recording ambient audio
    await _startHiddenAudioRecording();
    
    // 3. Initiate fake incoming call
    await _triggerFakeCall();
    
    // 4. Show comprehensive status notification
    await _showEmergencyStatusNotification();
    
    // 5. Save emergency log to MongoDB via API
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
      };
      
      try {
        await ApiService.createEmergencyLog(emergencyLog);
        print('Emergency log saved to MongoDB via API');
      } catch (e) {
        print('Failed to save emergency log: $e');
      }
    }
  }

  static Future<void> _showEmergencyStatusNotification() async {
    const AndroidNotificationDetails androidPlatformChannelSpecifics =
        AndroidNotificationDetails(
      'emergency_status', 
      'Emergency Status',
      channelDescription: 'Shows status of emergency actions',
      importance: Importance.max,
      priority: Priority.high,
      ticker: 'Emergency actions triggered',
    );
    const NotificationDetails platformChannelSpecifics =
        NotificationDetails(android: androidPlatformChannelSpecifics);
        
    String statusText = '''
Emergency Actions Status:
? Location Shared: ${_isLocationShared ? 'YES' : 'NO'}
? Audio Recording: ${_isAudioRecording ? 'ACTIVE' : 'FAILED'}
? Fake Call: ${_isFakeCallTriggered ? 'TRIGGERED' : 'FAILED'}
${_audioFilePath != null ? '? Audio File: $_audioFilePath' : ''}
${_lastLocationShareTime != null ? '? Location Time: $_lastLocationShareTime' : ''}
    ''';
        
    await _notificationsPlugin.show(
      id: 888,
      title: 'EMERGENCY STATUS REPORT',
      body: statusText.trim(),
      notificationDetails: platformChannelSpecifics,
    );
  }

  /// Get current emergency status as a formatted string
  static String getEmergencyStatus() {
    return '''
Emergency Status Report:
=====================
Location Shared: ${_isLocationShared ? 'YES' : 'NO'}
Audio Recording: ${_isAudioRecording ? 'ACTIVE' : 'STOPPED'}
Fake Call: ${_isFakeCallTriggered ? 'TRIGGERED' : 'NOT TRIGGERED'}
Audio File Path: ${_audioFilePath ?? 'Not available'}
Last Location Share: ${_lastLocationShareTime ?? 'Never'}
    ''';
  }

  static Future<void> _sendEmergencyAlert(LatLng location) async {
    try {
      print('SENDING EMERGENCY LOCATION...');
      print('   Location: ${location.latitude}, ${location.longitude}');
      print('   User ID: guardian_user_01');
      print('   Time: ${DateTime.now().toIso8601String()}');
      
      final response = await http.post(
        Uri.parse('$_baseUrl/api/emergency-trigger'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'latitude': location.latitude,
          'longitude': location.longitude,
          'user_id': 'guardian_user_01', 
          'timestamp': DateTime.now().toIso8601String(),
        }),
      ).timeout(const Duration(seconds: 5));

      if (response.statusCode == 200) {
        _isLocationShared = true;
        _lastLocationShareTime = DateTime.now().toIso8601String();
        print('EMERGENCY LOCATION SHARED SUCCESSFULLY!');
        print('   Status Code: ${response.statusCode}');
        print('   Response: ${response.body}');
      } else {
        _isLocationShared = false;
        print('FAILED TO SHARE LOCATION: ${response.statusCode}');
        print('   Response: ${response.body}');
      }
    } catch (e) {
      _isLocationShared = false;
      print('ERROR SHARING LOCATION: $e');
    }
  }

  static Future<void> _startHiddenAudioRecording() async {
    try {
      print('STARTING HIDDEN AUDIO RECORDING...');
      
      // Check and request microphone permission
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        _isAudioRecording = false;
        print('MICROPHONE PERMISSION DENIED - CANNOT RECORD');
        return;
      }

      if (await _audioRecorder.hasPermission()) {
        final dir = await getApplicationDocumentsDirectory();
        final fileName = 'emergency_audio_${DateTime.now().millisecondsSinceEpoch}.m4a';
        final filePath = '${dir.path}/$fileName';
        
        // Start recording
        await _audioRecorder.start(
          const RecordConfig(encoder: AudioEncoder.aacLc),
          path: filePath,
        );
        
        _isAudioRecording = true;
        _audioFilePath = filePath;
        print('AUDIO RECORDING STARTED SUCCESSFULLY!');
        print('   File Path: $filePath');
        print('   Duration: 2 minutes maximum');
        print('   Format: M4A (AAC)');
        
        // Automatically stop after 2 minutes to save space
        Future.delayed(const Duration(minutes: 2), () async {
          if (await _audioRecorder.isRecording()) {
            final path = await _audioRecorder.stop();
            _isAudioRecording = false;
            print('AUDIO RECORDING COMPLETED');
            print('   Final File: $path');
            print('   Duration: 2 minutes');
          }
        });
      }
    } catch (e) {
      _isAudioRecording = false;
      print('ERROR STARTING AUDIO RECORDING: $e');
    }
  }

  static Future<void> _triggerFakeCall() async {
    try {
      print('TRIGGERING FAKE INCOMING CALL...');
      print('   Caller: Mom');
      print('   Number: 0123456789');
      print('   Duration: 30 seconds');
      
      // Basic call parameters - simplified version
      await FlutterCallkitIncoming.showCallkitIncoming(
        CallKitParams(
          id: const Uuid().v4(),
          nameCaller: 'Mom',
          appName: 'GuardianAI',
          avatar: '',
          handle: '0123456789',
          type: 0,
          textAccept: 'Accept',
          textDecline: 'Decline',
          duration: 30000,
          extra: <String, dynamic>{'userId': '1a2b3c4d'},
        ),
      );
      
      _isFakeCallTriggered = true;
      print('FAKE CALL TRIGGERED SUCCESSFULLY!');
      print('   Call Screen: Active');
      print('   Ringtone: System default');
      print('   Purpose: Escape excuse');
    } catch (e) {
      _isFakeCallTriggered = false;
      print('ERROR TRIGGERING FAKE CALL: $e');
    }
  }
}

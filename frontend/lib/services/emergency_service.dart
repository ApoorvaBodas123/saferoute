import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_callkit_incoming/flutter_callkit_incoming.dart';
import 'package:flutter_callkit_incoming/entities/entities.dart';
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:latlong2/latlong.dart';
import 'package:uuid/uuid.dart';

class EmergencyService {
  static const String _baseUrl = 'http://localhost:8000';
  static final _audioRecorder = AudioRecorder();
  
  /// Triggers the full emergency sequence discreetly
  static Future<void> triggerEmergencySequence(LatLng currentLocation) async {
    print('🚨 SECRET PHRASE DETECTED -> INITIATING SOS SEQUENCE 🚨');
    
    // 1. Send live location to backend
    await _sendEmergencyAlert(currentLocation);
    
    // 2. Start recording ambient audio
    await _startHiddenAudioRecording();
    
    // 3. Initiate fake incoming call
    await _triggerFakeCall();
  }

  static Future<void> _sendEmergencyAlert(LatLng location) async {
    try {
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
        print('✅ Emergency alert sent successfully');
      } else {
        print('❌ Failed to send emergency alert: ${response.statusCode}');
      }
    } catch (e) {
      print('❌ Error sending emergency alert: $e');
    }
  }

  static Future<void> _startHiddenAudioRecording() async {
    try {
      // Check and request microphone permission
      final status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        print('❌ Microphone permission denied');
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
        
        print('🎙️ Started hidden audio recording at: $filePath');
        
        // Automatically stop after 2 minutes to save space
        Future.delayed(const Duration(minutes: 2), () async {
          if (await _audioRecorder.isRecording()) {
            final path = await _audioRecorder.stop();
            print('⏹️ Stopped hidden audio recording. Saved to: $path');
          }
        });
      }
    } catch (e) {
      print('❌ Error starting audio recording: $e');
    }
  }

  static Future<void> _triggerFakeCall() async {
    try {
      final params = CallKitParams(
        id: const Uuid().v4(),
        nameCaller: 'Mom',
        appName: 'GuardianAI',
        avatar: '',
        handle: '0123456789',
        type: 0,
        textAccept: 'Accept',
        textDecline: 'Decline',
        missedCallNotification: const NotificationParams(
          showNotification: true,
          isShowCallback: true,
          subtitle: 'Missed call',
          callbackText: 'Call back',
        ),
        duration: 30000,
        extra: <String, dynamic>{'userId': '1a2b3c4d'},
        android: const AndroidParams(
          isCustomNotification: true,
          isShowLogo: false,
          ringtonePath: 'system_ringtone_default',
          backgroundColor: '#0955fa',
          backgroundUrl: 'assets/test.png',
          actionColor: '#4CAF50',
        ),
        ios: const IOSParams(
          iconName: 'CallKitLogo',
          handleType: 'generic',
          supportsVideo: true,
          maximumCallGroups: 2,
          maximumCallsPerCallGroup: 1,
          audioSessionMode: 'default',
          audioSessionActive: true,
          audioSessionPreferredSampleRate: 44100.0,
          audioSessionPreferredIOBufferDuration: 0.005,
          supportsDTMF: true,
          supportsHolding: true,
          supportsGrouping: false,
          supportsUngrouping: false,
          ringtonePath: 'system_ringtone_default',
        ),
      );
      
      await FlutterCallkitIncoming.showCallkitIncoming(params);
      print('📞 Fake incoming call triggered');
    } catch (e) {
      print('❌ Error triggering fake call: $e');
    }
  }
}

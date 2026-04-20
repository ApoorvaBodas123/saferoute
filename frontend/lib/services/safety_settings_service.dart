import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

class EmergencyContact {
  const EmergencyContact({
    required this.name,
    required this.phone,
    this.relation,
  });

  final String name;
  final String phone;
  final String? relation;

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'phone': phone,
      'relation': relation,
    };
  }

  factory EmergencyContact.fromJson(Map<String, dynamic> json) {
    return EmergencyContact(
      name: (json['name'] ?? '').toString(),
      phone: (json['phone'] ?? '').toString(),
      relation: json['relation']?.toString(),
    );
  }
}

class SafetySettingsService {
  static const String _secretPhraseKey = 'secret_phrase';
  static const String _contactsKey = 'emergency_contacts';
  static const String defaultSecretPhrase = 'did you feed the dog';

  static Future<String> getSecretPhrase() async {
    final prefs = await SharedPreferences.getInstance();
    final phrase = prefs.getString(_secretPhraseKey)?.trim();
    if (phrase == null || phrase.isEmpty) {
      return defaultSecretPhrase;
    }
    return phrase;
  }

  static Future<void> setSecretPhrase(String phrase) async {
    final prefs = await SharedPreferences.getInstance();
    final normalized = phrase.trim().toLowerCase();
    if (normalized.isEmpty) {
      await prefs.setString(_secretPhraseKey, defaultSecretPhrase);
      return;
    }
    await prefs.setString(_secretPhraseKey, normalized);
  }

  static Future<List<EmergencyContact>> getEmergencyContacts() async {
    final prefs = await SharedPreferences.getInstance();
    final encoded = prefs.getString(_contactsKey);
    if (encoded == null || encoded.isEmpty) {
      return const <EmergencyContact>[];
    }

    try {
      final decoded = jsonDecode(encoded);
      if (decoded is! List) {
        return const <EmergencyContact>[];
      }
      return decoded
          .whereType<Map>()
          .map((item) => EmergencyContact.fromJson(Map<String, dynamic>.from(item)))
          .toList();
    } catch (_) {
      return const <EmergencyContact>[];
    }
  }

  static Future<void> saveEmergencyContacts(List<EmergencyContact> contacts) async {
    final prefs = await SharedPreferences.getInstance();
    final payload = contacts.map((c) => c.toJson()).toList();
    await prefs.setString(_contactsKey, jsonEncode(payload));
  }

  static Future<void> addEmergencyContact(EmergencyContact contact) async {
    final current = await getEmergencyContacts();
    final deduped = <EmergencyContact>[
      ...current.where((c) => c.phone != contact.phone),
      contact,
    ];
    await saveEmergencyContacts(deduped);
  }

  static Future<void> removeEmergencyContact(String phone) async {
    final current = await getEmergencyContacts();
    final filtered = current.where((c) => c.phone != phone).toList();
    await saveEmergencyContacts(filtered);
  }
}

import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../services/auth_service.dart';
import '../services/safety_settings_service.dart';

class ContactsScreen extends StatefulWidget {
  const ContactsScreen({super.key});

  @override
  State<ContactsScreen> createState() => _ContactsScreenState();
}

class _ContactsScreenState extends State<ContactsScreen> {
  final TextEditingController _phraseController = TextEditingController();
  List<EmergencyContact> _customContacts = const <EmergencyContact>[];
  bool _loading = true;
  String? _currentUserEmail;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  @override
  void dispose() {
    _phraseController.dispose();
    super.dispose();
  }

  Future<void> _loadSettings() async {
    final currentUser = await AuthService.getCurrentUser();
    final phrase = await SafetySettingsService.getSecretPhrase();
    List<EmergencyContact> contacts = await SafetySettingsService.getEmergencyContacts();

    final userEmail = currentUser?['email']?.toString();
    if (userEmail != null && userEmail.isNotEmpty) {
      try {
        final dbContacts = await ApiService.getEmergencyContacts(userEmail);
        contacts = dbContacts
            .map((e) => EmergencyContact.fromJson(e))
            .toList();
        await SafetySettingsService.saveEmergencyContacts(contacts);
      } catch (_) {
        // Keep local contacts as fallback if backend fetch fails.
      }
    }

    if (!mounted) return;
    setState(() {
      _phraseController.text = phrase;
      _customContacts = contacts;
      _currentUserEmail = userEmail;
      _loading = false;
    });
  }

  Future<void> _persistContacts(List<EmergencyContact> contacts) async {
    await SafetySettingsService.saveEmergencyContacts(contacts);

    final email = _currentUserEmail;
    if (email == null || email.isEmpty) {
      return;
    }

    final payload = contacts.map((c) => c.toJson()).toList();
    await ApiService.saveEmergencyContacts(email, payload);
  }

  Future<void> _savePhrase() async {
    await SafetySettingsService.setSecretPhrase(_phraseController.text);
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Secret phrase updated successfully.')),
    );
  }

  Future<void> _showAddContactDialog() async {
    final nameController = TextEditingController();
    final phoneController = TextEditingController();
    final relationController = TextEditingController();

    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Add Emergency Contact'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameController,
              decoration: const InputDecoration(labelText: 'Name'),
            ),
            TextField(
              controller: phoneController,
              keyboardType: TextInputType.phone,
              decoration: const InputDecoration(labelText: 'Phone Number'),
            ),
            TextField(
              controller: relationController,
              decoration: const InputDecoration(labelText: 'Relation (optional)'),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              final messenger = ScaffoldMessenger.of(this.context);
              final name = nameController.text.trim();
              final phone = phoneController.text.trim();
              final relation = relationController.text.trim();

              if (name.isEmpty || phone.isEmpty) {
                messenger.showSnackBar(
                  const SnackBar(content: Text('Name and phone are required.')),
                );
                return;
              }

              final updatedContacts = <EmergencyContact>[
                ..._customContacts.where((c) => c.phone != phone),
                EmergencyContact(
                  name: name,
                  phone: phone,
                  relation: relation.isEmpty ? null : relation,
                ),
              ];

              try {
                await _persistContacts(updatedContacts);
              } catch (e) {
                messenger.showSnackBar(
                  SnackBar(content: Text('Contact saved locally, DB sync failed: $e')),
                );
              }

              if (!mounted) return;
              Navigator.of(this.context).pop();
              await _loadSettings();
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );

    nameController.dispose();
    phoneController.dispose();
    relationController.dispose();
  }

  Future<void> _removeContact(String phone) async {
    final updatedContacts = _customContacts.where((c) => c.phone != phone).toList();
    try {
      await _persistContacts(updatedContacts);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Removed locally, DB sync failed: $e')),
        );
      }
    }
    await _loadSettings();
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Safety Settings & Contacts'),
        backgroundColor: Colors.red,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Secret Phrase Emergency Trigger',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          const Text(
            'Use an inconspicuous phrase. Example: "Did you feed the dog".',
            style: TextStyle(color: Colors.black54),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _phraseController,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: 'Secret Phrase',
            ),
          ),
          const SizedBox(height: 8),
          Align(
            alignment: Alignment.centerRight,
            child: ElevatedButton.icon(
              onPressed: _savePhrase,
              icon: const Icon(Icons.save),
              label: const Text('Save Phrase'),
            ),
          ),
          const Divider(height: 32),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Your Emergency Contacts',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              ElevatedButton.icon(
                onPressed: _showAddContactDialog,
                icon: const Icon(Icons.person_add, size: 18),
                label: const Text('Add'),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
              ),
            ],
          ),
          const SizedBox(height: 8),
          if (_customContacts.isEmpty)
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Text('No contacts added yet. Use Add to save trusted contacts.'),
              ),
            ),
          ..._customContacts.map(
            (contact) => Card(
              margin: const EdgeInsets.only(bottom: 10),
              child: ListTile(
                leading: const CircleAvatar(
                  backgroundColor: Colors.red,
                  child: Icon(Icons.phone, color: Colors.white),
                ),
                title: Text(contact.name),
                subtitle: Text(
                  '${contact.phone}${contact.relation == null ? '' : ' • ${contact.relation}'}',
                ),
                trailing: IconButton(
                  icon: const Icon(Icons.delete_outline, color: Colors.red),
                  onPressed: () => _removeContact(contact.phone),
                ),
              ),
            ),
          ),
          const Divider(height: 32),
          const Text(
            'Emergency Services',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          _buildServiceTile('Police', '100', Icons.local_police, Colors.blue),
          _buildServiceTile('Ambulance', '108', Icons.local_hospital, Colors.red),
          _buildServiceTile('Fire Department', '101', Icons.fire_truck, Colors.orange),
          _buildServiceTile('Women Helpline', '1091', Icons.support_agent, Colors.purple),
        ],
      ),
    );
  }

  Widget _buildServiceTile(
    String title,
    String number,
    IconData icon,
    Color color,
  ) {
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withValues(alpha: 0.15),
          child: Icon(icon, color: color),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        trailing: Text(
          number,
          style: TextStyle(color: color, fontWeight: FontWeight.bold),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import '../models/user_profile.dart';
import '../services/route_analytics.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  UserProfile? _profile;
  Map<String, dynamic>? _analytics;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadProfileAndAnalytics();
  }

  Future<void> _loadProfileAndAnalytics() async {
    final profile = await UserProfile.loadProfile();
    final history = await RouteAnalytics.getRouteHistory();
    final analytics = RouteAnalytics.getSafetyStats(history);

    setState(() {
      _profile = profile ?? UserProfile(
        name: 'Guest User',
        email: 'guest@example.com',
      );
      _analytics = analytics;
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Safety Profile'),
        backgroundColor: Colors.blueAccent,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Profile Section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'User Profile',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    _buildProfileField('Name', _profile!.name),
                    _buildProfileField('Email', _profile!.email),
                    _buildProfileField('Risk Tolerance', _profile!.riskTolerance.displayName),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Analytics Section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Safety Analytics',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    _buildAnalyticsRow('Total Routes', '${_analytics!['totalRoutes']}'),
                    _buildAnalyticsRow('Average Risk Score', '${_analytics!['avgRiskScore'].toStringAsFixed(2)}'),
                    _buildAnalyticsRow('Total Distance', '${_analytics!['totalDistance'].toStringAsFixed(1)} km'),
                    _buildAnalyticsRow('Most Used Strategy', '${_analytics!['mostUsedStrategy']}'),
                    if (_analytics!['safestRoute'] != null)
                      _buildAnalyticsRow('Safest Route Risk', '${_analytics!['safestRoute'].riskScore.toStringAsFixed(2)}'),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 16),
            
            // Settings Section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Safety Settings',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 16),
                    ListTile(
                      leading: const Icon(Icons.security),
                      title: const Text('Risk Tolerance'),
                      subtitle: Text(_profile!.riskTolerance.displayName),
                      trailing: const Icon(Icons.arrow_forward_ios),
                      onTap: _showRiskToleranceDialog,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProfileField(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          Expanded(
            child: Text(value),
          ),
        ],
      ),
    );
  }

  Widget _buildAnalyticsRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          Text(
            value,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              color: Colors.blueAccent,
            ),
          ),
        ],
      ),
    );
  }

  void _showRiskToleranceDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Select Risk Tolerance'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: SafetyLevel.values.map((level) {
              return RadioListTile<SafetyLevel>(
                title: Text(level.displayName),
                value: level,
                // ignore: deprecated_member_use
                groupValue: _profile!.riskTolerance,
                // ignore: deprecated_member_use
                onChanged: (SafetyLevel? value) {
                  setState(() {
                    _profile = UserProfile(
                      name: _profile!.name,
                      email: _profile!.email,
                      riskTolerance: value!,
                      preferredAreas: _profile!.preferredAreas,
                      avoidedAreas: _profile!.avoidedAreas,
                      usualTravelStart: _profile!.usualTravelStart,
                      usualTravelEnd: _profile!.usualTravelEnd,
                    );
                    UserProfile.saveProfile(_profile!);
                  });
                  Navigator.of(context).pop();
                },
              );
            }).toList(),
          ),
        );
      },
    );
  }
}

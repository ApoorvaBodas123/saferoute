import 'package:flutter/material.dart';

class HelpScreen extends StatelessWidget {
  const HelpScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Help & Support'),
        backgroundColor: Colors.green,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Quick Help
          _buildHelpSection(
            title: '🚀 Quick Start',
            content: '1. Tap on the map to set destination\n2. Choose route strategy (Safest/Fastest/Balanced)\n3. Follow the color-coded route\n4. Monitor risk levels in real-time',
            icon: Icons.play_arrow,
          ),
          
          _buildHelpSection(
            title: '🎯 Route Strategies',
            content: '• Safest (Green): Lowest risk, longer routes\n• Fastest (Orange): Quick routes, moderate risk\n• Balanced (Blue): Optimal balance of safety and speed',
            icon: Icons.route,
          ),
          
          _buildHelpSection(
            title: '⚠️ Risk Levels',
            content: '• Low Risk (0-3): Safe areas\n• Medium Risk (4-6): Caution advised\n• High Risk (7-10): Avoid if possible',
            icon: Icons.warning,
          ),
          
          _buildHelpSection(
            title: '📱 App Features',
            content: '• Real-time GPS tracking\n• ML-powered risk assessment\n• Emergency contacts\n• Route history\n• Safety alerts',
            icon: Icons.featured_play_list,
          ),
          
          _buildHelpSection(
            title: '🔧 Troubleshooting',
            content: '• GPS not working: Enable location services\n• Routes not showing: Check internet connection\n• App crashes: Restart app and update to latest version',
            icon: Icons.build,
          ),
          
          _buildHelpSection(
            title: '📞 Need More Help?',
            content: '• Call our 24/7 helpline: 1800-123-4567\n• Email: support@saferoute.in\n• Visit our website: www.saferoute.in',
            icon: Icons.support_agent,
          ),
          
          // FAQ Section
          const SizedBox(height: 24),
          const Text(
            'Frequently Asked Questions',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          
          _buildFAQItem(
            question: 'How accurate are the risk predictions?',
            answer: 'Our ML models achieve 86% accuracy based on historical crime data and real-time factors.',
          ),
          
          _buildFAQItem(
            question: 'Can I use SafeRoute in other cities?',
            answer: 'Currently optimized for Bangalore. We\'re expanding to other major cities soon.',
          ),
          
          _buildFAQItem(
            question: 'Does the app work offline?',
            answer: 'Basic features work offline, but real-time risk assessment requires internet connection.',
          ),
          
          _buildFAQItem(
            question: 'How much battery does it use?',
            answer: 'Optimized for minimal battery usage. GPS runs only when navigating.',
          ),
        ],
      ),
    );
  }

  Widget _buildHelpSection({
    required String title,
    required String content,
    required IconData icon,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: Colors.blue, size: 24),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              content,
              style: const TextStyle(fontSize: 14, height: 1.4),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFAQItem({
    required String question,
    required String answer,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ExpansionTile(
        title: Text(
          question,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Text(answer),
          ),
        ],
      ),
    );
  }
}
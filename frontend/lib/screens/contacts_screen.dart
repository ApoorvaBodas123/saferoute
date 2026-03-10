import 'package:flutter/material.dart';

class ContactsScreen extends StatelessWidget {
  const ContactsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Emergency Contacts'),
        backgroundColor: Colors.red,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'Emergency Services',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          
          // Emergency Contacts
          _buildContactCard(
            icon: Icons.local_police,
            title: 'Police',
            number: '100',
            description: 'Report crimes and emergencies',
            color: Colors.blue,
            context: context,
          ),
          _buildContactCard(
            icon: Icons.local_hospital,
            title: 'Ambulance',
            number: '108',
            description: 'Medical emergencies',
            color: Colors.red,
            context: context,
          ),
          _buildContactCard(
            icon: Icons.fire_truck,
            title: 'Fire Department',
            number: '101',
            description: 'Fire emergencies',
            color: Colors.orange,
            context: context,
          ),
          _buildContactCard(
            icon: Icons.support_agent,
            title: 'Women Helpline',
            number: '1091',
            description: 'Women safety helpline',
            color: Colors.purple,
            context: context,
          ),
          _buildContactCard(
            icon: Icons.child_care,
            title: 'Child Helpline',
            number: '1098',
            description: 'Child protection services',
            color: Colors.green,
            context: context,
          ),
          
          const SizedBox(height: 24),
          const Text(
            'SafeRoute Support',
            style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          
          _buildContactCard(
            icon: Icons.phone,
            title: 'SafeRoute Helpline',
            number: '1800-123-4567',
            description: '24/7 SafeRoute support',
            color: Colors.teal,
            context: context,
          ),
          _buildContactCard(
            icon: Icons.email,
            title: 'Email Support',
            number: 'support@saferoute.in',
            description: 'Email us for help',
            color: Colors.indigo,
            context: context,
          ),
        ],
      ),
    );
  }

  Widget _buildContactCard({
    required IconData icon,
    required String title,
    required String number,
    required String description,
    required Color color,
    required BuildContext context,  // Add this parameter
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: color.withOpacity(0.1),
          child: Icon(icon, color: color),
        ),
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Text(description),
        trailing: Text(
          number,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: color,
            fontSize: 16,
          ),
        ),
        onTap: () {
          // TODO: Implement phone call functionality
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Calling $number...')),
          );
        },
      ),
    );
  }
}
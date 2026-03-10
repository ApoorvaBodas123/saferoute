import 'package:flutter/material.dart';

class LogsScreen extends StatelessWidget {
  const LogsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Route History'),
        backgroundColor: Colors.blue,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Sample route logs
          _buildRouteLog(
            date: '2024-03-10',
            time: '14:30',
            from: 'MG Road',
            to: 'Indiranagar',
            strategy: 'Safest',
            riskScore: 3.2,
            distance: 8.5,
            duration: 25,
            status: 'Completed',
          ),
          _buildRouteLog(
            date: '2024-03-10',
            time: '09:15',
            from: 'Koramangala',
            to: 'Electronic City',
            strategy: 'Fastest',
            riskScore: 5.8,
            distance: 12.3,
            duration: 35,
            status: 'Completed',
          ),
          _buildRouteLog(
            date: '2024-03-09',
            time: '18:45',
            from: 'Whitefield',
            to: 'Marathahalli',
            strategy: 'Balanced',
            riskScore: 4.1,
            distance: 6.7,
            duration: 18,
            status: 'Completed',
          ),
          _buildRouteLog(
            date: '2024-03-09',
            time: '22:30',
            from: 'HSR Layout',
            to: 'BTM Layout',
            strategy: 'Safest',
            riskScore: 7.2,
            distance: 4.2,
            duration: 15,
            status: 'High Risk Alert',
          ),
        ],
      ),
    );
  }

  Widget _buildRouteLog({
    required String date,
    required String time,
    required String from,
    required String to,
    required String strategy,
    required double riskScore,
    required double distance,
    required int duration,
    required String status,
  }) {
    Color riskColor = riskScore < 4 ? Colors.green : 
                    riskScore < 7 ? Colors.orange : Colors.red;
    
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '$date $time',
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.grey,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: riskColor.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    status,
                    style: TextStyle(
                      color: riskColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('From: $from', style: const TextStyle(fontSize: 14)),
                      Text('To: $to', style: const TextStyle(fontSize: 14)),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      strategy,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.blue,
                      ),
                    ),
                    Text(
                      '${distance.toStringAsFixed(1)} km',
                      style: const TextStyle(fontSize: 12),
                    ),
                    Text(
                      '${duration} min',
                      style: const TextStyle(fontSize: 12),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.warning, size: 16, color: Colors.orange),
                const SizedBox(width: 4),
                Text(
                  'Risk Score: ${riskScore.toStringAsFixed(1)}',
                  style: TextStyle(
                    color: riskColor,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
import 'dart:convert';
import 'package:flutter/services.dart';
import '../models/risk_zone.dart';

class RiskService {
  static Future<List<RiskZone>> loadZones() async {
    final data = await rootBundle.loadString('assets/bangalore_risk_zones.json');
    final List decoded = json.decode(data);
    return decoded.map((e) => RiskZone.fromJson(e)).toList();
  }
}
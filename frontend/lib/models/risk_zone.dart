class RiskZone {
  final double lat;
  final double lon;
  final String zone;
  final double risk;

  RiskZone({
    required this.lat,
    required this.lon,
    required this.zone,
    required this.risk,
  });

  factory RiskZone.fromJson(Map<String, dynamic> json) {
    return RiskZone(
       lat: json['lat_grid'].toDouble(),
      lon: json['lon_grid'].toDouble(),
      zone: json['zone'],
      risk: json['risk_score'].toDouble(),
    );
  }
}

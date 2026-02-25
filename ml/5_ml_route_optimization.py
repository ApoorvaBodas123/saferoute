import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import requests
import json
import warnings
warnings.filterwarnings('ignore')


import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DynamicRiskScorer:
    
    
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        
        try:
            self.classification_model = joblib.load("./models/ensemble_model.pkl")
            self.feature_columns = joblib.load("./models/feature_columns.pkl")
            self.trend_analysis = joblib.load("./models/crime_trend_analysis.pkl")
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            
            self.classification_model = None
            self.feature_columns = []
            self.trend_analysis = {'hourly_pattern': {}, 'daily_pattern': {}, 'monthly_pattern': {}}
            
    def calculate_dynamic_risk_score(self, latitude, longitude, prediction_time=None):
        
        
        if prediction_time is None:
            prediction_time = datetime.now()
            
        
        hour = prediction_time.hour
        day_of_week = prediction_time.weekday()
        month = prediction_time.month
        
       
        base_risk = 3.0  
        
        
        if hour >= 22 or hour < 6:  
            temporal_multiplier = 1.5
        elif hour >= 18 and hour < 22:  
            temporal_multiplier = 1.2
        else:  # Day time
            temporal_multiplier = 1.0
            
        
        if day_of_week >= 5:
            temporal_multiplier *= 1.1
            
        
        distance_to_center = self._calculate_distance_to_center(latitude, longitude)
        if distance_to_center < 5:  
            spatial_multiplier = 1.3
        elif distance_to_center < 10:  
            spatial_multiplier = 1.1
        else:
            spatial_multiplier = 1.0
            
        final_risk_score = base_risk * temporal_multiplier * spatial_multiplier
        
        return {
            'current_risk_score': min(10, final_risk_score),
            'risk_probability': final_risk_score / 10,
            'temporal_multiplier': temporal_multiplier,
            'risk_level': self._categorize_risk(final_risk_score)
        }
    
    def _calculate_distance_to_center(self, lat, lon):
        
        from math import radians, cos, sin, asin, sqrt
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1 
            dlat = lat2 - lat1 
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
        
        return haversine_distance(lat, lon, 12.9716, 77.5946)
    
    def _categorize_risk(self, risk_score):
        
        if risk_score <= 3:
            return 'Low'
        elif risk_score <= 6:
            return 'Medium'
        elif risk_score <= 8:
            return 'High'
        else:
            return 'Very High'

class MLRouteOptimizer:
    
    
    def __init__(self):
        self.load_models()
        self.risk_network = None
        self.initialize_risk_network()
        
    def load_models(self):
        
        try:
            self.risk_system = joblib.load("./models/dynamic_risk_system.pkl")
            self.classification_model = joblib.load("./models/ensemble_model.pkl")
            self.feature_columns = joblib.load("./models/feature_columns.pkl")
            print("✅ Models loaded for route optimization")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def initialize_risk_network(self):
        
        
        print("🛣️ Initializing risk-based road network...")
        
       
        self.risk_network = nx.Graph()
        
        
        lat_min, lat_max = 12.8, 13.1
        lon_min, lon_max = 77.4, 77.8
        
       
        grid_size = 0.01  
        lat_points = np.arange(lat_min, lat_max, grid_size)
        lon_points = np.arange(lon_min, lon_max, grid_size)
        
        
        for lat in lat_points:
            for lon in lon_points:
                risk_result = self.risk_system.calculate_dynamic_risk_score(lat, lon)
                
                self.risk_network.add_node(
                    f"{lat:.3f},{lon:.3f}",
                    latitude=lat,
                    longitude=lon,
                    risk_score=risk_result['current_risk_score'],
                    risk_level=risk_result['risk_level']
                )
        
       
        for lat in lat_points:
            for lon in lon_points:
                current_node = f"{lat:.3f},{lon:.3f}"
                
                
                for d_lat in [-grid_size, 0, grid_size]:
                    for d_lon in [-grid_size, 0, grid_size]:
                        if d_lat == 0 and d_lon == 0:
                            continue
                            
                        neighbor_lat = lat + d_lat
                        neighbor_lon = lon + d_lon
                        
                        if (lat_min <= neighbor_lat <= lat_max and 
                            lon_min <= neighbor_lon <= lon_max):
                            
                            neighbor_node = f"{neighbor_lat:.3f},{neighbor_lon:.3f}"
                            
                            
                            if (current_node in self.risk_network.nodes and 
                                neighbor_node in self.risk_network.nodes):
                                
                                
                                distance = np.sqrt(d_lat**2 + d_lon**2) * 111000  
                                
                         
                                current_risk = self.risk_network.nodes[current_node]['risk_score']
                                neighbor_risk = self.risk_network.nodes[neighbor_node]['risk_score']
                                avg_risk = (current_risk + neighbor_risk) / 2
                                
                               
                                risk_penalty = avg_risk * 100  
                                edge_weight = distance + risk_penalty
                                
                                self.risk_network.add_edge(
                                    current_node, neighbor_node,
                                    weight=edge_weight,
                                    distance=distance,
                                    avg_risk=avg_risk
                                )
        
        print(f"✅ Network created with {self.risk_network.number_of_nodes()} nodes and {self.risk_network.number_of_edges()} edges")
    
    def find_nearest_node(self, lat, lon):
       
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.risk_network.nodes():
            node_lat = self.risk_network.nodes[node]['latitude']
            node_lon = self.risk_network.nodes[node]['longitude']
            
            distance = np.sqrt((lat - node_lat)**2 + (lon - node_lon)**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def optimize_route(self, start_lat, start_lon, end_lat, end_lon, optimization_strategy='balanced'):
      
        
        print(f"🚀 Optimizing route from ({start_lat:.3f}, {start_lon:.3f}) to ({end_lat:.3f}, {end_lon:.3f})")
        
        
        start_node = self.find_nearest_node(start_lat, start_lon)
        end_node = self.find_nearest_node(end_lat, end_lon)
        
        if not start_node or not end_node:
            return {"error": "Could not find nearby network nodes"}
        
        
        strategies = {
            'safest': lambda path: self._calculate_path_risk(path),
            'fastest': lambda path: self._calculate_path_distance(path),
            'balanced': lambda path: self._calculate_balanced_score(path)
        }
        
        
        routes = []
        
        try:
        
            shortest_path = nx.shortest_path(self.risk_network, start_node, end_node, weight='distance')
            routes.append({
                'path': shortest_path,
                'strategy': 'shortest',
                'distance': self._calculate_path_distance(shortest_path),
                'risk_score': self._calculate_path_risk(shortest_path),
                'balanced_score': self._calculate_balanced_score(shortest_path)
            })
        except nx.NetworkXNoPath:
            pass
        
        try:
           
            safest_path = nx.shortest_path(self.risk_network, start_node, end_node, weight='avg_risk')
            routes.append({
                'path': safest_path,
                'strategy': 'safest',
                'distance': self._calculate_path_distance(safest_path),
                'risk_score': self._calculate_path_risk(safest_path),
                'balanced_score': self._calculate_balanced_score(safest_path)
            })
        except nx.NetworkXNoPath:
            pass
        
        try:
          
            balanced_path = nx.shortest_path(self.risk_network, start_node, end_node, weight='weight')
            routes.append({
                'path': balanced_path,
                'strategy': 'balanced',
                'distance': self._calculate_path_distance(balanced_path),
                'risk_score': self._calculate_path_risk(balanced_path),
                'balanced_score': self._calculate_balanced_score(balanced_path)
            })
        except nx.NetworkXNoPath:
            pass
        
        if not routes:
            return {"error": "No route found"}
        
        
        if optimization_strategy in strategies:
            best_route = min(routes, key=lambda r: strategies[optimization_strategy](r['path']))
        else:
            best_route = min(routes, key=lambda r: r['balanced_score'])
        
        
        route_coordinates = []
        for node in best_route['path']:
            node_data = self.risk_network.nodes[node]
            route_coordinates.append([node_data['latitude'], node_data['longitude']])
        
       
        route_stats = self._calculate_route_statistics(best_route['path'])
        
        return {
            'route_coordinates': route_coordinates,
            'strategy': best_route['strategy'],
            'distance_km': best_route['distance'] / 1000,
            'avg_risk_score': best_route['risk_score'],
            'risk_level': self._categorize_route_risk(best_route['risk_score']),
            'estimated_time_minutes': route_stats['estimated_time'],
            'high_risk_segments': route_stats['high_risk_segments'],
            'alternative_routes': len(routes) - 1,
            'optimization_strategy': optimization_strategy
        }
    
    def _calculate_path_distance(self, path):
       
        total_distance = 0
        
        for i in range(len(path) - 1):
            try:
                edge_data = self.risk_network.get_edge_data(path[i], path[i+1])
                total_distance += edge_data['distance']
            except:
                continue
        
        return total_distance
    
    def _calculate_path_risk(self, path):
        
        total_risk = 0
        risk_count = 0
        
        for node in path:
            node_risk = self.risk_network.nodes[node]['risk_score']
            total_risk += node_risk
            risk_count += 1
        
        return total_risk / risk_count if risk_count > 0 else 0
    
    def _calculate_balanced_score(self, path):
        
        distance = self._calculate_path_distance(path)
        risk = self._calculate_path_risk(path)
        
        
        normalized_distance = distance / 10000  
        normalized_risk = risk / 10  
        
        return (normalized_distance * 0.6 + normalized_risk * 0.4)
    
    def _calculate_route_statistics(self, path):
       
        
        high_risk_segments = []
        total_risk = 0
        
        for i in range(len(path)):
            node = path[i]
            node_risk = self.risk_network.nodes[node]['risk_score']
            total_risk += node_risk
            
            
            if node_risk >= 7:
                high_risk_segments.append({
                    'segment_index': i,
                    'coordinates': [
                        self.risk_network.nodes[node]['latitude'],
                        self.risk_network.nodes[node]['longitude']
                    ],
                    'risk_score': node_risk
                })
        
        
        distance = self._calculate_path_distance(path)
        estimated_time = (distance / 1000) / 30 * 60  # Convert to minutes
        
        return {
            'estimated_time': estimated_time,
            'high_risk_segments': high_risk_segments,
            'total_risk': total_risk,
            'avg_risk': total_risk / len(path)
        }
    
    def _categorize_route_risk(self, avg_risk_score):
        
        if avg_risk_score <= 3:
            return 'Low Risk'
        elif avg_risk_score <= 5:
            return 'Medium Risk'
        elif avg_risk_score <= 7:
            return 'High Risk'
        else:
            return 'Very High Risk'
    
    def compare_routes(self, start_lat, start_lon, end_lat, end_lon):
       
        
        print("🔄 Comparing multiple routing strategies...")
        
        strategies = ['safest', 'fastest', 'balanced']
        comparison_results = []
        
        for strategy in strategies:
            route_result = self.optimize_route(start_lat, start_lon, end_lat, end_lon, strategy)
            
            if 'error' not in route_result:
                comparison_results.append({
                    'strategy': strategy,
                    'distance_km': route_result['distance_km'],
                    'avg_risk_score': route_result['avg_risk_score'],
                    'estimated_time_minutes': route_result['estimated_time_minutes'],
                    'high_risk_segments': len(route_result['high_risk_segments']),
                    'risk_level': route_result['risk_level']
                })
        
      
        if comparison_results:
            safest_route = min(comparison_results, key=lambda x: x['avg_risk_score'])
            fastest_route = min(comparison_results, key=lambda x: x['distance_km'])
            balanced_route = min(comparison_results, key=lambda x: x['avg_risk_score'] * x['distance_km'])
            
            return {
                'comparison_table': comparison_results,
                'recommendations': {
                    'safest': safest_route,
                    'fastest': fastest_route,
                    'balanced': balanced_route
                },
                'total_alternatives': len(comparison_results)
            }
        else:
            return {"error": "No routes found for comparison"}
    
    def get_real_time_route_update(self, route_coordinates, current_position_index):
       
        
        if current_position_index >= len(route_coordinates):
            return {"error": "Invalid position index"}
        
       
        remaining_route = route_coordinates[current_position_index:]
        
      
        updated_segments = []
        total_remaining_risk = 0
        
        for coord in remaining_route:
            lat, lon = coord
            risk_result = self.risk_system.calculate_dynamic_risk_score(lat, lon)
            
            updated_segments.append({
                'coordinates': coord,
                'current_risk_score': risk_result['current_risk_score'],
                'risk_level': risk_result['risk_level'],
                'temporal_multiplier': risk_result['temporal_multiplier']
            })
            
            total_remaining_risk += risk_result['current_risk_score']
        
        return {
            'remaining_segments': updated_segments,
            'avg_remaining_risk': total_remaining_risk / len(updated_segments),
            'segments_remaining': len(updated_segments),
            'update_timestamp': datetime.now().isoformat()
        }

def test_route_optimization():
    
    
    print("🧪 Testing ML Route Optimization...")
    
    optimizer = MLRouteOptimizer()
    
    
    test_routes = [
      
        (12.9762, 77.6033, 12.8452, 77.6770),
        
        (12.9784, 77.6408, 12.9279, 77.6271),
        
        (12.9295, 77.5804, 12.9698, 77.7490)
    ]
    
    for i, (start_lat, start_lon, end_lat, end_lon) in enumerate(test_routes):
        print(f"\n🚗 Test Route {i+1}:")
        
        
        comparison = optimizer.compare_routes(start_lat, start_lon, end_lat, end_lon)
        
        if 'error' not in comparison:
            print(f"   ✅ Found {comparison['total_alternatives']} route options")
            
            for rec_type, route in comparison['recommendations'].items():
                print(f"   📍 {rec_type.title()}: {route['distance_km']:.2f}km, Risk: {route['risk_level']} ({route['avg_risk_score']:.2f})")
        else:
            print(f"   ❌ {comparison['error']}")
    
    return optimizer

if __name__ == "__main__":

    route_optimizer = test_route_optimization()
    

    joblib.dump(route_optimizer, "./models/ml_route_optimizer.pkl")
    
    print("\n🎉 ML Route Optimization Complete!")
    print("💾 Optimizer saved: ml_route_optimizer.pkl")

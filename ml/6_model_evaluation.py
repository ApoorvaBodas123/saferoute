import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    
    
    def __init__(self):
        self.load_models()
        self.evaluation_results = {}
        
    def load_models(self):
      
        try:
            self.classification_model = joblib.load("./models/ensemble_model.pkl")
            self.time_series_model = joblib.load("./models/time_series_rf_model.pkl")
            self.feature_columns = joblib.load("./models/feature_columns.pkl")
            self.ts_features = joblib.load("./models/time_series_features.pkl")
            self.scaler = joblib.load("./models/feature_scaler.pkl")
            print("✅ Models loaded for evaluation")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def evaluate_classification_model(self):
        
        
        print("🔍 Evaluating Classification Model...")
        
      
        data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
        X = data[self.feature_columns]
        y = data['is_high_risk']
        

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        y_pred = self.classification_model.predict(X_test)
        y_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        
        cv_scores = cross_val_score(self.classification_model, X_train, y_train, cv=5, scoring='roc_auc')
        
    
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        
        cm = confusion_matrix(y_test, y_pred)
        
        
        self.evaluation_results['classification'] = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'avg_precision': avg_precision,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': self._get_feature_importance()
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ AUC: {auc_score:.4f}")
        print(f"   ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.evaluation_results['classification']
    
    def evaluate_time_series_model(self):
        
        
        print("🔍 Evaluating Time Series Model...")
        
        try:
            
            ts_data = self._prepare_time_series_data()
            
            if len(ts_data) < 100:
                print("⚠️ Insufficient time series data for evaluation")
                return None
            
            
            available_features = [col for col in self.ts_features if col in ts_data.columns]
            
            if len(available_features) < 5:
                print("⚠️ Insufficient features for time series evaluation")
                return None
            
           
            train_size = int(len(ts_data) * 0.8)
            train_data = ts_data[:train_size]
            test_data = ts_data[train_size:]
            
            X_train = train_data[available_features]
            y_train = train_data['high_risk_count']
            X_test = test_data[available_features]
            y_test = test_data['high_risk_count']
            
            
            y_pred = self.time_series_model.predict(X_test)
            
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
           
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
            
           
            self.evaluation_results['time_series'] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'mape': mape,
                'test_samples': len(y_test),
                'features_used': len(available_features)
            }
            
            print(f"   ✅ MSE: {mse:.4f}")
            print(f"   ✅ MAE: {mae:.4f}")
            print(f"   ✅ RMSE: {rmse:.4f}")
            print(f"   ✅ R²: {r2:.4f}")
            print(f"   ✅ MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"❌ Error in time series evaluation: {e}")
            return None
        
        return self.evaluation_results.get('time_series')
    
    def evaluate_route_optimization(self):
        
        
        print("🔍 Evaluating Route Optimization...")
        
        try:
            route_optimizer = joblib.load("./models/ml_route_optimizer.pkl")
            
           
            test_routes = [
                (12.9762, 77.6033, 12.8452, 77.6770),  # MG Road to Electronic City
                (12.9784, 77.6408, 12.9279, 77.6271),  # Indiranagar to Koramangala
                (12.9295, 77.5804, 12.9698, 77.7490)   # Jayanagar to Whitefield
            ]
            
            route_results = []
            for start_lat, start_lon, end_lat, end_lon in test_routes:
                comparison = route_optimizer.compare_routes(start_lat, start_lon, end_lat, end_lon)
                
                if 'error' not in comparison:
                    route_results.append({
                        'route_id': len(route_results) + 1,
                        'alternatives_found': comparison['total_alternatives'],
                        'safest_risk': comparison['recommendations']['safest']['avg_risk_score'],
                        'fastest_distance': comparison['recommendations']['fastest']['distance_km'],
                        'balanced_score': comparison['recommendations']['balanced']['avg_risk_score'] * comparison['recommendations']['balanced']['distance_km']
                    })
            
           
            if route_results:
                avg_alternatives = np.mean([r['alternatives_found'] for r in route_results])
                avg_risk_improvement = np.mean([
                    (r['fastest_distance'] - r['safest_risk']) / r['fastest_distance'] * 100 
                    for r in route_results
                ])
                
                self.evaluation_results['route_optimization'] = {
                    'routes_tested': len(route_results),
                    'avg_alternatives': avg_alternatives,
                    'avg_risk_improvement': avg_risk_improvement,
                    'success_rate': len(route_results) / len(test_routes) * 100
                }
                
                print(f"   ✅ Routes tested: {len(route_results)}")
                print(f"   ✅ Avg alternatives: {avg_alternatives:.1f}")
                print(f"   ✅ Risk improvement: {avg_risk_improvement:.2f}%")
            
        except Exception as e:
            print(f"❌ Error evaluating route optimization: {e}")
            return None
        
        return self.evaluation_results.get('route_optimization')
    
    def _prepare_time_series_data(self):

        try:
           
            data = pd.read_csv("./data/ml_enhanced_crime_data.csv")
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.sort_values('Datetime')
            
           
            temp_data = data.copy()
            temp_data['date'] = temp_data['Datetime'].dt.date
            temp_data['hour'] = temp_data['Datetime'].dt.hour
            
            hourly_crime = temp_data.groupby(['date', 'hour']).agg({
                'is_high_risk': ['sum', 'count'],
                'risk_score': 'mean'
            }).reset_index()
            
            hourly_crime.columns = ['date', 'hour', 'high_risk_count', 'total_crimes', 'avg_risk_score']
            hourly_crime['datetime'] = pd.to_datetime(hourly_crime['date'].astype(str) + ' ' + hourly_crime['hour'].astype(str) + ':00:00')
            hourly_crime = hourly_crime.set_index('datetime')
            
            
            hourly_crime['hour_sin'] = np.sin(2 * np.pi * hourly_crime['hour'] / 24)
            hourly_crime['hour_cos'] = np.cos(2 * np.pi * hourly_crime['hour'] / 24)
            hourly_crime['day_of_week'] = hourly_crime.index.dayofweek
            hourly_crime['day_sin'] = np.sin(2 * np.pi * hourly_crime['day_of_week'] / 7)
            hourly_crime['day_cos'] = np.cos(2 * np.pi * hourly_crime['day_of_week'] / 7)
            
           
            for lag in [1, 2, 3, 6, 12, 24]:
                hourly_crime[f'high_risk_lag_{lag}'] = hourly_crime['high_risk_count'].shift(lag)
            
            
            for window in [3, 6, 12]:
                hourly_crime[f'high_risk_rolling_{window}'] = hourly_crime['high_risk_count'].rolling(window=window).mean()
            
            hourly_crime = hourly_crime.dropna()
            return hourly_crime
            
        except Exception as e:
            print(f"❌ Error preparing time series data: {e}")
            return pd.DataFrame()
    
    def _get_feature_importance(self):
        
        try:
            if hasattr(self.classification_model, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_columns, self.classification_model.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except:
            return {}
    
    def generate_evaluation_report(self):
        
        
        print("\n📊 GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)
        
        
        self.evaluate_classification_model()
        self.evaluate_time_series_model()
        self.evaluate_route_optimization()
        
        
        report = {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'model_performance': self.evaluation_results,
            'summary': self._generate_summary()
        }
        
       
        joblib.dump(report, "./models/evaluation_report.pkl")
        
      
        self._print_summary()
        
        return report
    
    def _generate_summary(self):
        
        
        summary = {
            'classification_grade': self._calculate_grade('classification'),
            'time_series_grade': self._calculate_grade('time_series'),
            'route_optimization_grade': self._calculate_grade('route_optimization'),
            'overall_assessment': 'Good'
        }
        
        
        grades = [summary['classification_grade'], summary['time_series_grade'], summary['route_optimization_grade']]
        if all(grade in ['A', 'A+', 'B'] for grade in grades):
            summary['overall_assessment'] = 'Excellent'
        elif any(grade in ['D', 'F'] for grade in grades):
            summary['overall_assessment'] = 'Needs Improvement'
        else:
            summary['overall_assessment'] = 'Good'
        
        return summary
    
    def _calculate_grade(self, model_type):
       
        
        if model_type not in self.evaluation_results:
            return 'N/A'
        
        results = self.evaluation_results[model_type]
        
        if model_type == 'classification':
            auc = results['auc_score']
            if auc >= 0.9:
                return 'A+'
            elif auc >= 0.8:
                return 'A'
            elif auc >= 0.7:
                return 'B'
            elif auc >= 0.6:
                return 'C'
            else:
                return 'D'
        
        elif model_type == 'time_series':
            r2 = results['r2_score']
            if r2 >= 0.8:
                return 'A+'
            elif r2 >= 0.6:
                return 'A'
            elif r2 >= 0.4:
                return 'B'
            elif r2 >= 0.2:
                return 'C'
            else:
                return 'D'
        
        elif model_type == 'route_optimization':
            success_rate = results['success_rate']
            if success_rate >= 90:
                return 'A+'
            elif success_rate >= 80:
                return 'A'
            elif success_rate >= 70:
                return 'B'
            elif success_rate >= 60:
                return 'C'
            else:
                return 'D'
        
        return 'N/A'
    
    def _print_summary(self):
        
        
        print("\n🎯 PERFORMANCE SUMMARY")
        print("-" * 30)
        
        for model_type, results in self.evaluation_results.items():
            grade = self._calculate_grade(model_type)
            print(f"{model_type.replace('_', ' ').title()}: Grade {grade}")
            
            if model_type == 'classification':
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  AUC: {results['auc_score']:.4f}")
            elif model_type == 'time_series':
                print(f"  R²: {results['r2_score']:.4f}")
                print(f"  RMSE: {results['rmse']:.4f}")
            elif model_type == 'route_optimization':
                print(f"  Success Rate: {results['success_rate']:.1f}%")
                print(f"  Avg Alternatives: {results['avg_alternatives']:.1f}")
        
        print(f"\n🏆 Overall Assessment: {self._generate_summary()['overall_assessment']}")
    
    def create_performance_visualizations(self):
       
        
        print("📈 Creating performance visualizations...")
        
        try:
           
            if 'classification' in self.evaluation_results:
                self._plot_classification_metrics()
           
            if 'classification' in self.evaluation_results:
                self._plot_feature_importance()
            
            print("✅ Visualizations saved to ./models/plots/")
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def _plot_classification_metrics(self):
       
        
        import matplotlib.pyplot as plt
        
        results = self.evaluation_results['classification']
        
        
        cm = np.array(results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Risk', 'High Risk'], 
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('./models/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
       
        
        import matplotlib.pyplot as plt
        
        results = self.evaluation_results['classification']
        importance = results['feature_importance']
        
        if importance:
            
            top_features = dict(list(importance.items())[:10])
            
            plt.figure(figsize=(10, 6))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            plt.barh(features, importances)
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('./models/plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

def run_comprehensive_evaluation():
    
    
    print("🚀 Starting Comprehensive Model Evaluation...")
    
  
    import os
    os.makedirs("./models/plots", exist_ok=True)
    
    
    evaluator = ModelEvaluator()
    
   
    report = evaluator.generate_evaluation_report()
    
  
    evaluator.create_performance_visualizations()
    
    print("\n🎉 Comprehensive Evaluation Complete!")
    print("📊 Report saved: ./models/evaluation_report.pkl")
    print("📈 Visualizations saved: ./models/plots/")
    
    return report

if __name__ == "__main__":
  
    evaluation_report = run_comprehensive_evaluation()

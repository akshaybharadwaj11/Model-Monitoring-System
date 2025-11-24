"""
MCP Servers for ML Model Monitoring System
Simplified in-memory implementation for assignment demo
In production: would use PostgreSQL with full MCP protocol
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Type
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import threading
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ============================================================================
# MCP SERVER 1: Predictions & Ground Truth Store
# ============================================================================

class PredictionsMCPServer:
    """
    MCP Server for storing and retrieving model predictions
    In production: PostgreSQL with full MCP protocol
    For demo: In-memory with MCP-style interface
    """
    
    def __init__(self):
        self.predictions = []  # List of prediction dicts
        self.ground_truth = {}  # prediction_id -> actual_label
        self.lock = threading.Lock()
        
    def store_prediction(
        self,
        model_id: str,
        prediction_id: str,
        features: Dict,
        prediction: int,
        confidence: float,
        timestamp: datetime
    ) -> Dict:
        """Store a single prediction"""
        with self.lock:
            pred_entry = {
                'prediction_id': prediction_id,
                'model_id': model_id,
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': pd.Timestamp(timestamp)  # Convert to pandas Timestamp
            }
            self.predictions.append(pred_entry)
            
        return {
            'status': 'success',
            'message': f'Stored prediction {prediction_id}',
            'prediction_id': prediction_id
        }
    
    def store_ground_truth(
        self,
        prediction_id: str,
        actual_label: int,
        timestamp: datetime
    ) -> Dict:
        """Store ground truth label"""
        with self.lock:
            self.ground_truth[prediction_id] = {
                'actual_label': actual_label,
                'timestamp': pd.Timestamp(timestamp)  # Convert to pandas Timestamp
            }
        
        return {
            'status': 'success',
            'message': f'Stored ground truth for {prediction_id}'
        }
    
    def get_predictions(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> Dict:
        """Retrieve predictions with filters"""
        with self.lock:
            # Filter by model_id
            filtered = [p for p in self.predictions if p['model_id'] == model_id]
            
            # Filter by date range - safe comparison
            if start_date:
                start_date = pd.Timestamp(start_date)
                safe_filtered = []
                for p in filtered:
                    pred_time = p['timestamp']
                    if not isinstance(pred_time, pd.Timestamp):
                        pred_time = pd.Timestamp(pred_time)
                    if pred_time >= start_date:
                        safe_filtered.append(p)
                filtered = safe_filtered
            
            if end_date:
                end_date = pd.Timestamp(end_date)
                safe_filtered = []
                for p in filtered:
                    pred_time = p['timestamp']
                    if not isinstance(pred_time, pd.Timestamp):
                        pred_time = pd.Timestamp(pred_time)
                    if pred_time <= end_date:
                        safe_filtered.append(p)
                filtered = safe_filtered
            
            # Limit results
            filtered = filtered[:limit]
            
        return {
            'status': 'success',
            'count': len(filtered),
            'predictions': filtered
        }
    
    def calculate_accuracy(
        self,
        model_id: str,
        window_hours: int = 24
    ) -> Dict:
        """Calculate model accuracy over time window"""
        cutoff_time = pd.Timestamp(datetime.now() - timedelta(hours=window_hours))
        
        with self.lock:
            # Get recent predictions - use safe comparison
            recent_preds = []
            for p in self.predictions:
                if p['model_id'] != model_id:
                    continue
                
                # Safe timestamp comparison - convert both to timestamps
                pred_time = p['timestamp']
                if not isinstance(pred_time, pd.Timestamp):
                    pred_time = pd.Timestamp(pred_time)
                
                if pred_time >= cutoff_time:
                    recent_preds.append(p)
            
            # Calculate accuracy where ground truth is available
            correct = 0
            total = 0
            
            for pred in recent_preds:
                pred_id = pred['prediction_id']
                if pred_id in self.ground_truth:
                    total += 1
                    if pred['prediction'] == self.ground_truth[pred_id]['actual_label']:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
        
        return {
            'status': 'success',
            'model_id': model_id,
            'window_hours': window_hours,
            'accuracy': accuracy,
            'total_predictions': total,
            'correct_predictions': correct
        }
    
    def bulk_load_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Load predictions from simulated data"""
        with self.lock:
            for _, row in df.iterrows():
                # Ensure timestamp is pandas Timestamp
                timestamp = row['timestamp']
                if not isinstance(timestamp, pd.Timestamp):
                    timestamp = pd.Timestamp(timestamp)
                
                pred_entry = {
                    'prediction_id': row['prediction_id'],
                    'model_id': row['model_id'],
                    'features': {'hash': row['features_hash']},
                    'prediction': row['prediction'],
                    'confidence': row['confidence'],
                    'timestamp': timestamp
                }
                self.predictions.append(pred_entry)
                
                # Store ground truth
                self.ground_truth[row['prediction_id']] = {
                    'actual_label': row['true_label'],
                    'timestamp': timestamp
                }
        
        return {
            'status': 'success',
            'loaded_predictions': len(df)
        }


# ============================================================================
# MCP SERVER 2: Metrics & Performance Store
# ============================================================================

class MetricsMCPServer:
    """
    MCP Server for storing and retrieving performance metrics
    """
    
    def __init__(self):
        self.metrics = []  # Time-series metrics
        self.drift_scores = []  # Drift scores
        self.lock = threading.Lock()
    
    def store_metric(
        self,
        model_id: str,
        metric_name: str,
        value: float,
        timestamp: datetime
    ) -> Dict:
        """Store a single metric data point"""
        with self.lock:
            metric_entry = {
                'model_id': model_id,
                'metric_name': metric_name,
                'value': value,
                'timestamp': pd.Timestamp(timestamp)  # Convert to pandas Timestamp
            }
            self.metrics.append(metric_entry)
        
        return {
            'status': 'success',
            'message': f'Stored {metric_name} for {model_id}'
        }
    
    def get_metric_timeseries(
        self,
        model_id: str,
        metric_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Get time-series data for a metric"""
        with self.lock:
            filtered = [
                m for m in self.metrics
                if m['model_id'] == model_id and m['metric_name'] == metric_name
            ]
            
            # Safe date filtering
            if start_date:
                start_date = pd.Timestamp(start_date)
                safe_filtered = []
                for m in filtered:
                    m_time = m['timestamp']
                    if not isinstance(m_time, pd.Timestamp):
                        m_time = pd.Timestamp(m_time)
                    if m_time >= start_date:
                        safe_filtered.append(m)
                filtered = safe_filtered
            
            if end_date:
                end_date = pd.Timestamp(end_date)
                safe_filtered = []
                for m in filtered:
                    m_time = m['timestamp']
                    if not isinstance(m_time, pd.Timestamp):
                        m_time = pd.Timestamp(m_time)
                    if m_time <= end_date:
                        safe_filtered.append(m)
                filtered = safe_filtered
            
            # Sort by timestamp safely
            filtered.sort(key=lambda x: pd.Timestamp(x['timestamp']) if not isinstance(x['timestamp'], pd.Timestamp) else x['timestamp'])
        
        return {
            'status': 'success',
            'metric_name': metric_name,
            'data_points': len(filtered),
            'timeseries': filtered
        }
    
    def store_drift_score(
        self,
        model_id: str,
        drift_type: str,
        score: float,
        timestamp: datetime
    ) -> Dict:
        """Store drift score"""
        with self.lock:
            drift_entry = {
                'model_id': model_id,
                'drift_type': drift_type,
                'score': score,
                'timestamp': pd.Timestamp(timestamp)  # Convert to pandas Timestamp
            }
            self.drift_scores.append(drift_entry)
        
        return {
            'status': 'success',
            'message': f'Stored {drift_type} drift score'
        }
    
    def get_latest_drift_scores(
        self,
        model_id: str
    ) -> Dict:
        """Get most recent drift scores"""
        with self.lock:
            model_drifts = [
                d for d in self.drift_scores
                if d['model_id'] == model_id
            ]
            
            if not model_drifts:
                return {
                    'status': 'no_data',
                    'message': 'No drift scores found'
                }
            
            # Sort safely
            def safe_timestamp(d):
                ts = d['timestamp']
                return pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
            
            model_drifts_sorted = sorted(model_drifts, key=safe_timestamp, reverse=True)
            
            # Get latest for each drift type
            latest_by_type = {}
            for drift in model_drifts_sorted:
                drift_type = drift['drift_type']
                if drift_type not in latest_by_type:
                    latest_by_type[drift_type] = drift
        
        return {
            'status': 'success',
            'model_id': model_id,
            'drift_scores': latest_by_type
        }
    
    def get_model_health(
        self,
        model_id: str
    ) -> Dict:
        """Get overall model health status"""
        # Get latest accuracy
        accuracy_metrics = [
            m for m in self.metrics
            if m['model_id'] == model_id and m['metric_name'] == 'accuracy'
        ]
        
        if not accuracy_metrics:
            return {
                'status': 'no_data',
                'health': 'unknown'
            }
        
        # Safe sorting
        def safe_timestamp(m):
            ts = m['timestamp']
            return pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
        
        latest_accuracy = sorted(accuracy_metrics, key=safe_timestamp)[-1]
        
        # Get drift scores
        drift_result = self.get_latest_drift_scores(model_id)
        
        # Determine health status
        accuracy = latest_accuracy['value']
        
        if drift_result['status'] == 'success':
            drift_scores = drift_result['drift_scores']
            max_drift = max(d['score'] for d in drift_scores.values())
        else:
            max_drift = 0.0
        
        # Health determination
        if accuracy >= 0.85 and max_drift < 0.15:
            health = 'healthy'
        elif accuracy >= 0.80 and max_drift < 0.25:
            health = 'warning'
        else:
            health = 'critical'
        
        return {
            'status': 'success',
            'model_id': model_id,
            'health': health,
            'accuracy': float(accuracy),
            'max_drift_score': float(max_drift),
            'timestamp': latest_accuracy['timestamp']
        }
    
    def bulk_load_metrics_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Load metrics from simulated data"""
        with self.lock:
            for _, row in df.iterrows():
                # Ensure date is pandas Timestamp
                date_val = row['date']
                if not isinstance(date_val, pd.Timestamp):
                    date_val = pd.Timestamp(date_val)
                
                # Store each metric
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if metric in row:
                        self.metrics.append({
                            'model_id': row['model_id'],
                            'metric_name': metric,
                            'value': row[metric],
                            'timestamp': date_val
                        })
                
                # Store latency metrics
                for metric in ['avg_latency_ms', 'p99_latency_ms']:
                    if metric in row:
                        self.metrics.append({
                            'model_id': row['model_id'],
                            'metric_name': metric,
                            'value': row[metric],
                            'timestamp': date_val
                        })
        
        return {
            'status': 'success',
            'loaded_metrics': len(df) * 6  # 6 metrics per row
        }
    
    def bulk_load_drift_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """Load drift scores from simulated data"""
        with self.lock:
            for _, row in df.iterrows():
                # Ensure date is pandas Timestamp
                date_val = row['date']
                if not isinstance(date_val, pd.Timestamp):
                    date_val = pd.Timestamp(date_val)
                
                for drift_type in ['covariate_drift', 'prediction_drift', 'concept_drift', 'overall_drift_score']:
                    if drift_type in row:
                        self.drift_scores.append({
                            'model_id': row['model_id'],
                            'drift_type': drift_type,
                            'score': row[drift_type],
                            'timestamp': date_val
                        })
        
        return {
            'status': 'success',
            'loaded_drift_scores': len(df) * 4
        }


# ============================================================================
# MCP SERVER 3: Alerts & Incidents Store
# ============================================================================

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    model_id: str
    alert_type: str
    severity: str
    message: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict = None


@dataclass
class Incident:
    """Incident data structure"""
    incident_id: str
    alert_id: str
    model_id: str
    root_cause: str
    impact: str
    status: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    remediation_actions: List[Dict] = None


class IncidentsMCPServer:
    """
    MCP Server for alerts and incident management
    """
    
    def __init__(self):
        self.alerts = []
        self.incidents = []
        self.remediation_actions = []
        self.lock = threading.Lock()
    
    def create_alert(
        self,
        model_id: str,
        alert_type: str,
        severity: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a new alert"""
        with self.lock:
            alert_id = f"alert_{len(self.alerts)}_{datetime.now().timestamp()}"
            
            alert = Alert(
                alert_id=alert_id,
                model_id=model_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                created_at=pd.Timestamp(datetime.now()),  # Convert to pandas Timestamp
                metadata=metadata or {}
            )
            
            self.alerts.append(alert)
        
        return {
            'status': 'success',
            'alert_id': alert_id,
            'message': 'Alert created'
        }
    
    def get_active_alerts(
        self,
        model_id: Optional[str] = None
    ) -> Dict:
        """Get unresolved alerts"""
        with self.lock:
            active = [
                a for a in self.alerts
                if a.resolved_at is None
            ]
            
            if model_id:
                active = [a for a in active if a.model_id == model_id]
        
        return {
            'status': 'success',
            'count': len(active),
            'alerts': [asdict(a) for a in active]
        }
    
    def resolve_alert(
        self,
        alert_id: str
    ) -> Dict:
        """Mark alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved_at = pd.Timestamp(datetime.now())  # Convert to pandas Timestamp
                    return {
                        'status': 'success',
                        'message': f'Alert {alert_id} resolved'
                    }
        
        return {
            'status': 'error',
            'message': 'Alert not found'
        }
    
    def create_incident(
        self,
        alert_id: str,
        model_id: str,
        root_cause: str,
        impact: str
    ) -> Dict:
        """Create incident from alert"""
        with self.lock:
            incident_id = f"incident_{len(self.incidents)}_{datetime.now().timestamp()}"
            
            incident = Incident(
                incident_id=incident_id,
                alert_id=alert_id,
                model_id=model_id,
                root_cause=root_cause,
                impact=impact,
                status='open',
                created_at=pd.Timestamp(datetime.now()),  # Convert to pandas Timestamp
                remediation_actions=[]
            )
            
            self.incidents.append(incident)
        
        return {
            'status': 'success',
            'incident_id': incident_id,
            'message': 'Incident created'
        }
    
    def store_remediation_action(
        self,
        incident_id: str,
        action_type: str,
        details: Dict,
        outcome: Dict
    ) -> Dict:
        """Store remediation action taken"""
        with self.lock:
            action = {
                'action_id': f"action_{len(self.remediation_actions)}",
                'incident_id': incident_id,
                'action_type': action_type,
                'details': details,
                'outcome': outcome,
                'timestamp': pd.Timestamp(datetime.now())  # Convert to pandas Timestamp
            }
            
            self.remediation_actions.append(action)
            
            # Update incident
            for incident in self.incidents:
                if incident.incident_id == incident_id:
                    if incident.remediation_actions is None:
                        incident.remediation_actions = []
                    incident.remediation_actions.append(action)
        
        return {
            'status': 'success',
            'action_id': action['action_id'],
            'message': 'Remediation action stored'
        }
    
    def get_incident_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> Dict:
        """Get incident history for model"""
        with self.lock:
            model_incidents = [
                i for i in self.incidents
                if i.model_id == model_id
            ]
            
            # Safe sorting by creation time
            def safe_timestamp(i):
                ts = i.created_at
                return pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
            
            model_incidents.sort(key=safe_timestamp, reverse=True)
            model_incidents = model_incidents[:limit]
        
        return {
            'status': 'success',
            'count': len(model_incidents),
            'incidents': [asdict(i) for i in model_incidents]
        }


# ============================================================================
# MCP Manager - Coordinates all servers
# ============================================================================

class MCPManager:
    """
    Manager that coordinates all MCP servers
    Provides unified interface for agents
    """
    
    def __init__(self):
        self.predictions_server = PredictionsMCPServer()
        self.metrics_server = MetricsMCPServer()
        self.incidents_server = IncidentsMCPServer()
    
    def load_simulated_data(self, data_dir: str):
        """Load simulated data into all servers"""
        import os
        
        print("Loading simulated data into MCP servers...")
        
        for model_dir in os.listdir(data_dir):
            model_path = os.path.join(data_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            # Load predictions with timestamp parsing
            predictions_file = os.path.join(model_path, 'predictions.csv')
            if os.path.exists(predictions_file):
                df = pd.read_csv(predictions_file, parse_dates=['timestamp'])
                self.predictions_server.bulk_load_from_dataframe(df)
                print(f"  Loaded {len(df)} predictions for {model_dir}")
            
            # Load metrics with date parsing
            metrics_file = os.path.join(model_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file, parse_dates=['date'])
                self.metrics_server.bulk_load_metrics_from_dataframe(df)
                print(f"  Loaded metrics for {model_dir}")
            
            # Load drift scores with date parsing
            drift_file = os.path.join(model_path, 'drift_scores.csv')
            if os.path.exists(drift_file):
                df = pd.read_csv(drift_file, parse_dates=['date'])
                self.metrics_server.bulk_load_drift_from_dataframe(df)
                print(f"  Loaded drift scores for {model_dir}")
        
        print("Data loading complete!")
    
    def get_monitoring_data(self, model_id: str) -> Dict:
        """
        Get comprehensive monitoring data for RL agent
        This is what the RL agent uses to make decisions
        """
        # Get latest metrics
        accuracy_data = self.metrics_server.get_metric_timeseries(
            model_id, 'accuracy'
        )
        
        if accuracy_data['status'] != 'success' or not accuracy_data['timeseries']:
            return {
                'current_accuracy': 0.85,
                'drift_score': 0.05,
                'days_since_retrain': 30,
                'retraining_cost': 5000,
                'business_impact': 50000,
                'data_available': True,
                'accuracy_trend': 0.0,
                'alert_count': 0,
                'model_age_days': 60,
                'previous_action_success': 0.5
            }
        
        # Get current accuracy
        latest_accuracy = accuracy_data['timeseries'][-1]['value']
        
        # Calculate accuracy trend
        if len(accuracy_data['timeseries']) >= 3:
            recent_accuracies = [t['value'] for t in accuracy_data['timeseries'][-3:]]
            accuracy_trend = recent_accuracies[-1] - recent_accuracies[0]
        else:
            accuracy_trend = 0.0
        
        # Get drift scores
        drift_data = self.metrics_server.get_latest_drift_scores(model_id)
        if drift_data['status'] == 'success':
            max_drift = max(d['score'] for d in drift_data['drift_scores'].values())
        else:
            max_drift = 0.05
        
        # Get active alerts
        alerts_data = self.incidents_server.get_active_alerts(model_id)
        alert_count = alerts_data['count']
        
        return {
            'current_accuracy': float(latest_accuracy),
            'drift_score': float(max_drift),
            'days_since_retrain': 30,  # Simulated
            'retraining_cost': 5000,
            'business_impact': 75000,
            'data_available': True,
            'accuracy_trend': float(accuracy_trend),
            'alert_count': alert_count,
            'model_age_days': 60,
            'previous_action_success': 0.7
        }


# ============================================================================
# Tool Wrappers for CrewAI Agents
# ============================================================================

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class QueryPredictionsInput(BaseModel):
    """Input for querying predictions"""
    model_id: str = Field(..., description="Model ID to query")
    hours: int = Field(24, description="Hours of data to retrieve")

class QueryPredictionsTool(BaseTool):
    name: str = "Query Model Predictions"
    description: str = "Retrieve model predictions and calculate accuracy over a time window"
    args_schema: Type[BaseModel] = QueryPredictionsInput
    mcp_manager: Any = None
    
    def _run(self, model_id: str, hours: int = 24) -> str:
        result = self.mcp_manager.predictions_server.calculate_accuracy(
            model_id=model_id,
            window_hours=hours
        )
        return json.dumps(result, indent=2, default=str)

class QueryMetricsInput(BaseModel):
    """Input for querying metrics"""
    model_id: str = Field(..., description="Model ID to query")
    metric_name: str = Field(..., description="Metric to retrieve (accuracy, latency, etc)")

class QueryMetricsTool(BaseTool):
    name: str = "Query Performance Metrics"
    description: str = "Get time-series performance metrics for a model"
    args_schema: Type[BaseModel] = QueryMetricsInput
    mcp_manager: Any = None
    
    def _run(self, model_id: str, metric_name: str = "accuracy") -> str:
        result = self.mcp_manager.metrics_server.get_metric_timeseries(
            model_id=model_id,
            metric_name=metric_name
        )
        
        # Convert timestamps to strings for JSON serialization
        if result['status'] == 'success' and 'timeseries' in result:
            for item in result['timeseries']:
                if 'timestamp' in item and hasattr(item['timestamp'], 'isoformat'):
                    item['timestamp'] = item['timestamp'].isoformat()
        
        return json.dumps(result, indent=2, default=str)

class QueryDriftInput(BaseModel):
    """Input for querying drift scores"""
    model_id: str = Field(..., description="Model ID to query")

class QueryDriftTool(BaseTool):
    name: str = "Query Drift Scores"
    description: str = "Get latest drift scores (covariate, prediction, concept drift)"
    args_schema: Type[BaseModel] = QueryDriftInput
    mcp_manager: Any = None
    
    def _run(self, model_id: str) -> str:
        result = self.mcp_manager.metrics_server.get_latest_drift_scores(
            model_id=model_id
        )
        
        # Convert timestamps in drift scores
        if result['status'] == 'success' and 'drift_scores' in result:
            for drift_type, drift_data in result['drift_scores'].items():
                if 'timestamp' in drift_data and hasattr(drift_data['timestamp'], 'isoformat'):
                    drift_data['timestamp'] = drift_data['timestamp'].isoformat()
        
        return json.dumps(result, indent=2, default=str)

class QueryModelHealthInput(BaseModel):
    """Input for querying model health"""
    model_id: str = Field(..., description="Model ID to query")

class QueryModelHealthTool(BaseTool):
    name: str = "Query Model Health"
    description: str = "Get overall model health status (healthy/warning/critical)"
    args_schema: Type[BaseModel] = QueryModelHealthInput
    mcp_manager: Any = None
    
    def _run(self, model_id: str) -> str:
        result = self.mcp_manager.metrics_server.get_model_health(
            model_id=model_id
        )
        
        # Convert timestamp if present
        if 'timestamp' in result and hasattr(result['timestamp'], 'isoformat'):
            result['timestamp'] = result['timestamp'].isoformat()
        
        return json.dumps(result, indent=2, default=str)

class CreateAlertInput(BaseModel):
    """Input for creating alerts"""
    model_id: str = Field(..., description="Model ID")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Severity level")
    message: str = Field(..., description="Alert message")

class CreateAlertTool(BaseTool):
    name: str = "Create Alert"
    description: str = "Create a new alert for model issues"
    args_schema: Type[BaseModel] = CreateAlertInput
    mcp_manager: Any = None
    
    def _run(self, model_id: str, alert_type: str, severity: str, message: str) -> str:
        result = self.mcp_manager.incidents_server.create_alert(
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message
        )
        return json.dumps(result, indent=2, default=str)


if __name__ == "__main__":
    # Test MCP servers
    print("Testing MCP Servers...")
    
    manager = MCPManager()
    
    # Test predictions server
    print("\n1. Testing Predictions Server...")
    result = manager.predictions_server.store_prediction(
        model_id='test_model',
        prediction_id='pred_001',
        features={'feature1': 0.5},
        prediction=1,
        confidence=0.92,
        timestamp=datetime.now()
    )
    print(f"   {result['message']}")
    
    # Test metrics server
    print("\n2. Testing Metrics Server...")
    result = manager.metrics_server.store_metric(
        model_id='test_model',
        metric_name='accuracy',
        value=0.87,
        timestamp=datetime.now()
    )
    print(f"   {result['message']}")
    
    # Test incidents server
    print("\n3. Testing Incidents Server...")
    result = manager.incidents_server.create_alert(
        model_id='test_model',
        alert_type='performance_degradation',
        severity='high',
        message='Accuracy dropped below threshold'
    )
    print(f"   {result['message']}")
    
    print("\n✅ All MCP servers operational!")
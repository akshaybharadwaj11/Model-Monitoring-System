"""
Controller/Orchestrator for ML Model Monitoring System
Integrates RL agents, MCP servers, and specialized agents
"""

from crewai import Crew, Process, Agent, Task
from datetime import datetime
from typing import Dict, Any, List
import logging
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitoringOrchestrator:
    """
    Main controller that orchestrates the entire monitoring system
    Integrates RL for continuous improvement
    """
    
    def __init__(
        self,
        mcp_manager,
        rl_remediation_agent,
        threshold_bandit,
        llm
    ):
        self.mcp_manager = mcp_manager
        self.rl_agent = rl_remediation_agent
        self.threshold_bandit = threshold_bandit
        self.llm = llm
        
        # Session tracking
        self.session_memory = {
            'monitoring_runs': [],
            'alerts_created': [],
            'remediations_performed': [],
            'rl_training_history': []
        }
        
        # Performance metrics
        self.metrics = {
            'total_monitoring_runs': 0,
            'successful_remediations': 0,
            'failed_remediations': 0,
            'total_cost_saved': 0,
            'rl_episodes': 0,
            'avg_rl_reward': 0
        }
    
    def run_monitoring_cycle(
        self,
        model_id: str,
        day: int = 0
    ) -> Dict[str, Any]:
        """
        Execute one complete monitoring cycle for a model
        Includes RL-based decision making
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"MONITORING CYCLE - Day {day} - Model: {model_id}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Get current monitoring data from MCP
            monitoring_data = self.mcp_manager.get_monitoring_data(model_id)
            logger.info(f"Current Accuracy: {monitoring_data['current_accuracy']:.3f}")
            logger.info(f"Drift Score: {monitoring_data['drift_score']:.3f}")
            logger.info(f"Active Alerts: {monitoring_data['alert_count']}")
            
            # Create agents (import here to avoid circular dependency)
            from specialized_monitoring_agents import create_monitoring_agents, create_monitoring_tasks
            
            agents = create_monitoring_agents(self.llm, self.mcp_manager)
            
            # Execute agent analysis
            if self.llm is not None:
                logger.info("\n--- Running REAL LLM Agent Analysis ---")
                analysis_results = self._execute_llm_agent_workflow(
                    agents, model_id, monitoring_data
                )
            else:
                logger.info("\n--- Running Simulated Agent Analysis ---")
                analysis_results = self._simulate_agent_analysis(monitoring_data, model_id)
            
            # RL Agent selects remediation action
            logger.info("\n--- RL Remediation Agent Decision ---")
            rl_decision = self._execute_rl_remediation(
                monitoring_data,
                analysis_results,
                model_id
            )
            
            # Store results
            self.session_memory['monitoring_runs'].append({
                'model_id': model_id,
                'day': day,
                'timestamp': datetime.now().isoformat(),
                'monitoring_data': monitoring_data,
                'analysis_results': analysis_results,
                'rl_decision': rl_decision
            })
            
            self.metrics['total_monitoring_runs'] += 1
            
            # Update RL metrics
            if 'reward' in rl_decision:
                self.metrics['rl_episodes'] += 1
                self.metrics['avg_rl_reward'] = (
                    (self.metrics['avg_rl_reward'] * (self.metrics['rl_episodes'] - 1) +
                     rl_decision['reward']) / self.metrics['rl_episodes']
                )
            
            logger.info(f"\n✓ Monitoring cycle complete")
            
            return {
                'status': 'success',
                'day': day,
                'model_id': model_id,
                'monitoring_data': monitoring_data,
                'analysis': analysis_results,
                'rl_decision': rl_decision,
                'metrics': self.metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_llm_agent_workflow(
        self,
        agents: Dict[str, Agent],
        model_id: str,
        monitoring_data: Dict
    ) -> Dict[str, Any]:
        """
        Execute REAL CrewAI workflow with LLM agents
        This makes actual OpenAI API calls
        """
        from crewai import Crew, Task, Process
        
        try:
            # Create tasks for the workflow
            tasks = self._create_llm_tasks(agents, model_id, monitoring_data)
            
            # Create crew
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                memory=False,  # Disable memory to reduce context
                max_rpm=10  # Rate limit
            )
            
            logger.info("Executing CrewAI workflow with LLM agents...")
            
            # Execute crew (THIS MAKES REAL API CALLS)
            result = crew.kickoff()
            
            logger.info("LLM workflow complete!")
            
            # Parse LLM output into structured analysis
            analysis_results = self._parse_llm_output(result, monitoring_data)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in LLM workflow: {str(e)}")
            logger.info("Falling back to simulation mode...")
            return self._simulate_agent_analysis(monitoring_data, model_id)
    
    def _create_llm_tasks(
        self,
        agents: Dict[str, Agent],
        model_id: str,
        monitoring_data: Dict
    ) -> List[Task]:
        """
        Create CrewAI tasks for LLM execution
        """
        tasks = []
        
        # Task 1: Performance Analysis
        performance_task = Task(
            description=f"""
            Analyze performance for model: {model_id}
            
            Use the Query Performance Metrics tool to get:
            - Current accuracy (last 24 hours)
            - Latency metrics
            - Error rates
            
            Current status from monitoring:
            - Accuracy: {monitoring_data['current_accuracy']:.3f}
            - Drift Score: {monitoring_data['drift_score']:.3f}
            
            Determine the performance status:
            - healthy (accuracy >= 0.85, drift < 0.15)
            - warning (accuracy >= 0.80, drift < 0.25)
            - critical (accuracy < 0.80 or drift >= 0.25)
            
            Output JSON format:
            {{
                "status": "healthy|warning|critical",
                "accuracy": float,
                "recommendation": "string"
            }}
            """,
            agent=agents['performance_agent'],
            expected_output="JSON with performance status"
        )
        tasks.append(performance_task)
        
        # Task 2: Drift Detection
        drift_task = Task(
            description=f"""
            Detect drift for model: {model_id}
            
            Use the Query Drift Scores tool to analyze:
            - Covariate drift
            - Prediction drift  
            - Concept drift
            
            Current drift score: {monitoring_data['drift_score']:.3f}
            
            Classify drift level:
            - low: < 0.15
            - medium: 0.15-0.25
            - high: 0.25-0.35
            - critical: >= 0.35
            
            Output JSON format:
            {{
                "level": "low|medium|high|critical",
                "score": float,
                "urgency": "monitor|action_within_7_days|action_within_3_days|immediate_action"
            }}
            """,
            agent=agents['drift_agent'],
            expected_output="JSON with drift assessment",
            context=[performance_task]
        )
        tasks.append(drift_task)
        
        # Task 3: Alert Decision (simplified)
        alert_task = Task(
            description=f"""
            Decide if alerts should be created for model: {model_id}
            
            Based on previous analyses, determine:
            - Should we create an alert?
            - What severity? (low/medium/high/critical)
            
            If alert is needed, use Create Alert tool.
            
            Output JSON format:
            {{
                "should_alert": boolean,
                "triggered": boolean,
                "severity": "string"
            }}
            """,
            agent=agents['alert_agent'],
            expected_output="JSON with alert decision",
            context=[performance_task, drift_task]
        )
        tasks.append(alert_task)
        
        return tasks
    
    def _parse_llm_output(
        self, 
        crew_result: Any, 
        monitoring_data: Dict
    ) -> Dict[str, Any]:
        """
        Parse LLM output into structured analysis results
        """
        try:
            # Try to extract JSON from crew result
            result_str = str(crew_result)
            
            # Parse performance status
            if 'critical' in result_str.lower():
                performance_status = 'critical'
            elif 'warning' in result_str.lower():
                performance_status = 'warning'
            else:
                performance_status = 'healthy'
            
            # Parse drift level
            drift_score = monitoring_data['drift_score']
            if drift_score < 0.15:
                drift_level = 'low'
                drift_urgency = 'monitor'
            elif drift_score < 0.25:
                drift_level = 'medium'
                drift_urgency = 'action_within_7_days'
            elif drift_score < 0.35:
                drift_level = 'high'
                drift_urgency = 'action_within_3_days'
            else:
                drift_level = 'critical'
                drift_urgency = 'immediate_action'
            
            # Parse alert decision
            should_alert = performance_status in ['warning', 'critical'] or drift_level in ['high', 'critical']
            
            return {
                'performance': {
                    'status': performance_status,
                    'accuracy': monitoring_data['current_accuracy'],
                    'recommendation': f"Status is {performance_status}"
                },
                'drift': {
                    'level': drift_level,
                    'score': drift_score,
                    'urgency': drift_urgency
                },
                'quality': {
                    'score': monitoring_data['current_accuracy'] * 100
                },
                'alert': {
                    'triggered': should_alert,
                    'should_alert': should_alert
                },
                'llm_output': result_str[:500]  # Store first 500 chars of LLM output
            }
            
        except Exception as e:
            logger.error(f"Error parsing LLM output: {str(e)}")
            # Fallback to simple parsing
            return self._simulate_agent_analysis(monitoring_data, "unknown")
    
    def _simulate_agent_analysis(
        self,
        monitoring_data: Dict,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Simulate agent analysis (simplified for demo)
        In production: Would run full CrewAI tasks with LLM
        """
        
        accuracy = monitoring_data['current_accuracy']
        drift_score = monitoring_data['drift_score']
        
        # Performance Analysis
        if accuracy >= 0.85 and drift_score < 0.15:
            performance_status = 'healthy'
            performance_recommendation = 'Continue normal monitoring'
        elif accuracy >= 0.80 and drift_score < 0.25:
            performance_status = 'warning'
            performance_recommendation = 'Increase monitoring frequency'
        else:
            performance_status = 'critical'
            performance_recommendation = 'Immediate action required'
        
        # Drift Analysis
        if drift_score < 0.15:
            drift_level = 'low'
            drift_urgency = 'monitor'
        elif drift_score < 0.25:
            drift_level = 'medium'
            drift_urgency = 'action_within_7_days'
        elif drift_score < 0.35:
            drift_level = 'high'
            drift_urgency = 'action_within_3_days'
        else:
            drift_level = 'critical'
            drift_urgency = 'immediate_action'
        
        # Quality Analysis
        quality_score = accuracy * 100
        
        # Alert Management
        should_alert = (
            performance_status in ['warning', 'critical'] or
            drift_level in ['high', 'critical']
        )
        
        if should_alert:
            # Use RL threshold
            threshold = self.threshold_bandit.select_threshold()
            alert_triggered = drift_score > (threshold - 0.70)  # Normalize
            
            if alert_triggered:
                alert_result = self.mcp_manager.incidents_server.create_alert(
                    model_id=model_id,
                    alert_type='performance_degradation' if performance_status == 'critical' else 'drift_detected',
                    severity=performance_status,
                    message=f"Accuracy: {accuracy:.3f}, Drift: {drift_score:.3f}"
                )
                logger.info(f"Alert created: {alert_result['alert_id']}")
        else:
            alert_triggered = False
        
        return {
            'performance': {
                'status': performance_status,
                'accuracy': accuracy,
                'recommendation': performance_recommendation
            },
            'drift': {
                'level': drift_level,
                'score': drift_score,
                'urgency': drift_urgency
            },
            'quality': {
                'score': quality_score
            },
            'alert': {
                'triggered': alert_triggered,
                'should_alert': should_alert
            }
        }
    
    def _execute_rl_remediation(
        self,
        monitoring_data: Dict,
        analysis_results: Dict,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Use RL agent to select and execute remediation action
        This is where the RL learning happens!
        """
        
        # Get state for RL agent
        state = self.rl_agent.get_state(monitoring_data)
        
        # RL agent selects action
        action, action_prob, value = self.rl_agent.select_action(state, training=True)
        action_info = self.rl_agent.get_action_info(action)
        
        logger.info(f"RL Agent selected: {action_info.name}")
        logger.info(f"Action probability: {action_prob:.3f}")
        logger.info(f"Value estimate: {value:.3f}")
        
        # Simulate action outcome based on current state
        outcome = self._simulate_action_outcome(
            action,
            monitoring_data,
            analysis_results
        )
        
        logger.info(f"Outcome: Accuracy {outcome['accuracy_before']:.3f} → {outcome['accuracy_after']:.3f}")
        logger.info(f"Cost: ${outcome['cost']}")
        
        # Calculate reward
        reward = self.rl_agent.calculate_reward(action, outcome)
        logger.info(f"Reward: {reward:.2f}")
        
        # Store experience for training
        next_monitoring_data = monitoring_data.copy()
        next_monitoring_data['current_accuracy'] = outcome['accuracy_after']
        next_state = self.rl_agent.get_state(next_monitoring_data)
        
        self.rl_agent.store_experience(
            state=state,
            action=action,
            action_prob=action_prob,
            reward=reward,
            next_state=next_state,
            value=value,
            done=True
        )
        
        # Train RL agent if enough experience
        if len(self.rl_agent.memory) >= 32:
            training_metrics = self.rl_agent.train(batch_size=32, epochs=4)
            if training_metrics:
                logger.info(f"RL Training: Loss={training_metrics['loss']:.4f}, Reward={training_metrics['avg_reward']:.2f}")
                self.session_memory['rl_training_history'].append(training_metrics)
        
        # Update success metrics
        if reward > 5:
            self.metrics['successful_remediations'] += 1
            self.metrics['total_cost_saved'] += (outcome.get('business_impact', 0) - outcome['cost'])
        else:
            self.metrics['failed_remediations'] += 1
        
        # Store remediation action in MCP
        if analysis_results['alert']['triggered']:
            # Get most recent alert
            alerts = self.mcp_manager.incidents_server.get_active_alerts(model_id)
            if alerts['count'] > 0:
                latest_alert_id = alerts['alerts'][0]['alert_id']
                
                # Create incident
                incident_result = self.mcp_manager.incidents_server.create_incident(
                    alert_id=latest_alert_id,
                    model_id=model_id,
                    root_cause=f"Drift level: {analysis_results['drift']['level']}, Performance: {analysis_results['performance']['status']}",
                    impact=f"Accuracy at {monitoring_data['current_accuracy']:.3f}"
                )
                
                # Store remediation
                self.mcp_manager.incidents_server.store_remediation_action(
                    incident_id=incident_result['incident_id'],
                    action_type=action_info.name,
                    details={
                        'action_id': action,
                        'cost': outcome['cost'],
                        'expected_improvement': action_info.expected_improvement
                    },
                    outcome=outcome
                )
        
        return {
            'action': action_info.name,
            'action_id': action,
            'action_probability': action_prob,
            'value_estimate': value,
            'expected_cost': action_info.cost,
            'expected_improvement': action_info.expected_improvement,
            'outcome': outcome,
            'reward': reward,
            'rl_performance': self.rl_agent.get_performance_metrics()
        }
    
    def _simulate_action_outcome(
        self,
        action: int,
        monitoring_data: Dict,
        analysis_results: Dict
    ) -> Dict[str, Any]:
        """
        Simulate the outcome of a remediation action
        In production: Would execute real remediation and measure results
        """
        
        current_accuracy = monitoring_data['current_accuracy']
        drift_score = monitoring_data['drift_score']
        
        action_info = self.rl_agent.get_action_info(action)
        
        # Simulate accuracy improvement based on action
        if action in [0, 1, 2]:  # Retraining actions
            # Retraining improves accuracy proportional to drift
            improvement = min(action_info.expected_improvement + drift_score * 0.1, 0.15)
            new_accuracy = min(current_accuracy + improvement, 0.95)
            actual_cost = action_info.cost
            downtime = action_info.implementation_time_hours
            
        elif action == 3:  # Rollback
            # Rollback to previous version (slight improvement)
            improvement = 0.03
            new_accuracy = min(current_accuracy + improvement, 0.88)
            actual_cost = action_info.cost
            downtime = action_info.implementation_time_hours
            
        elif action == 4:  # Adjust threshold
            # Quick fix, minimal improvement
            improvement = 0.01
            new_accuracy = current_accuracy + improvement
            actual_cost = action_info.cost
            downtime = action_info.implementation_time_hours
            
        elif action == 5:  # Increase monitoring
            # No immediate accuracy improvement
            improvement = 0.0
            new_accuracy = current_accuracy
            actual_cost = action_info.cost
            downtime = 0
            
        else:  # Continue monitoring (action 6)
            # No action, accuracy may continue to degrade
            improvement = -0.01 if drift_score > 0.3 else 0.0
            new_accuracy = max(current_accuracy + improvement, 0.70)
            actual_cost = 0
            downtime = 0
        
        # Add some randomness
        new_accuracy += np.random.normal(0, 0.01)
        new_accuracy = np.clip(new_accuracy, 0.70, 0.95)
        
        # Calculate business impact (revenue saved by improving accuracy)
        business_impact = (new_accuracy - current_accuracy) * 500000  # $500K per 1% accuracy
        
        return {
            'accuracy_before': current_accuracy,
            'accuracy_after': new_accuracy,
            'accuracy_improvement': new_accuracy - current_accuracy,
            'cost': actual_cost,
            'downtime_hours': downtime,
            'business_impact': max(business_impact, 0)
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""
        
        rl_metrics = self.rl_agent.get_performance_metrics()
        threshold_stats = self.threshold_bandit.get_statistics()
        
        return {
            'monitoring_metrics': self.metrics,
            'rl_agent_performance': rl_metrics,
            'threshold_optimization': threshold_stats,
            'session_summary': {
                'total_runs': len(self.session_memory['monitoring_runs']),
                'alerts_created': len(self.session_memory['alerts_created']),
                'remediations_performed': len(self.session_memory['remediations_performed']),
                'rl_training_iterations': len(self.session_memory['rl_training_history'])
            }
        }


if __name__ == "__main__":
    print("Controller module loaded successfully")
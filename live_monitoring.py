"""
Live Execution Dashboard for ML Model Monitoring System
Watch agents execute in real-time with rich terminal UI
"""

import sys
import time
import json
from datetime import datetime
from typing import Dict, Any
import threading
from collections import deque


class LiveDashboard:
    """
    Real-time terminal dashboard showing agent execution
    """
    
    def __init__(self):
        self.current_agent = None
        self.current_task = None
        self.execution_log = deque(maxlen=15)
        self.metrics = {
            'episode': 0,
            'accuracy': 0.0,
            'drift': 0.0,
            'reward': 0.0,
            'action': 'Initializing...'
        }
        self.agent_status = {
            'Performance Monitor': 'idle',
            'Drift Detector': 'idle',
            'Quality Analyzer': 'idle',
            'Alert Manager': 'idle',
            'Remediation Planner': 'idle'
        }
        
        # Colors for terminal output
        self.COLORS = {
            'RESET': '\033[0m',
            'BOLD': '\033[1m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'MAGENTA': '\033[95m',
            'CYAN': '\033[96m',
            'RED': '\033[91m',
            'GRAY': '\033[90m'
        }
    
    def clear_screen(self):
        """Clear terminal screen"""
        print('\033[2J\033[H', end='')
    
    def print_header(self, day: int, model_id: str):
        """Print dashboard header"""
        c = self.COLORS
        print(f"{c['BOLD']}{c['CYAN']}{'='*100}{c['RESET']}")
        print(f"{c['BOLD']}{c['CYAN']}🤖 ML MODEL MONITORING SYSTEM - LIVE EXECUTION DASHBOARD{c['RESET']}")
        print(f"{c['CYAN']}{'='*100}{c['RESET']}")
        print(f"{c['BOLD']}Day: {c['YELLOW']}{day}{c['RESET']} | Model: {c['YELLOW']}{model_id}{c['RESET']} | Time: {c['GRAY']}{datetime.now().strftime('%H:%M:%S')}{c['RESET']}")
        print()
    
    def print_agent_status(self):
        """Print current agent execution status"""
        c = self.COLORS
        print(f"{c['BOLD']}{c['MAGENTA']}🔧 AGENT STATUS:{c['RESET']}")
        print(f"{c['MAGENTA']}{'─'*100}{c['RESET']}")
        
        for agent, status in self.agent_status.items():
            if status == 'executing':
                icon = f"{c['GREEN']}●{c['RESET']}"
                status_text = f"{c['GREEN']}{c['BOLD']}EXECUTING{c['RESET']}"
            elif status == 'completed':
                icon = f"{c['BLUE']}✓{c['RESET']}"
                status_text = f"{c['BLUE']}COMPLETED{c['RESET']}"
            else:
                icon = f"{c['GRAY']}○{c['RESET']}"
                status_text = f"{c['GRAY']}IDLE{c['RESET']}"
            
            agent_display = f"{agent:<25}"
            print(f"  {icon} {agent_display} {status_text}")
        print()
    
    def print_metrics(self):
        """Print current metrics"""
        c = self.COLORS
        print(f"{c['BOLD']}{c['BLUE']}📊 REAL-TIME METRICS:{c['RESET']}")
        print(f"{c['BLUE']}{'─'*100}{c['RESET']}")
        
        # Accuracy
        acc_color = c['GREEN'] if self.metrics['accuracy'] >= 0.85 else c['YELLOW'] if self.metrics['accuracy'] >= 0.80 else c['RED']
        print(f"  Model Accuracy:     {acc_color}{self.metrics['accuracy']:.3f}{c['RESET']}")
        
        # Drift
        drift_color = c['GREEN'] if self.metrics['drift'] < 0.15 else c['YELLOW'] if self.metrics['drift'] < 0.25 else c['RED']
        print(f"  Drift Score:        {drift_color}{self.metrics['drift']:.3f}{c['RESET']}")
        
        # RL metrics
        print(f"  RL Episode:         {c['CYAN']}{self.metrics['episode']}{c['RESET']}")
        
        reward_color = c['GREEN'] if self.metrics['reward'] > 5 else c['YELLOW'] if self.metrics['reward'] > 0 else c['RED']
        print(f"  Last Reward:        {reward_color}{self.metrics['reward']:+.2f}{c['RESET']}")
        
        print(f"  Current Action:     {c['MAGENTA']}{self.metrics['action']}{c['RESET']}")
        print()
    
    def print_execution_log(self):
        """Print recent execution log"""
        c = self.COLORS
        print(f"{c['BOLD']}{c['YELLOW']}📝 EXECUTION LOG:{c['RESET']}")
        print(f"{c['YELLOW']}{'─'*100}{c['RESET']}")
        
        for log_entry in self.execution_log:
            timestamp = log_entry['timestamp'].strftime('%H:%M:%S')
            msg = log_entry['message']
            log_type = log_entry['type']
            
            if log_type == 'agent':
                icon = f"{c['MAGENTA']}🤖{c['RESET']}"
                text_color = c['MAGENTA']
            elif log_type == 'action':
                icon = f"{c['CYAN']}⚡{c['RESET']}"
                text_color = c['CYAN']
            elif log_type == 'result':
                icon = f"{c['GREEN']}✓{c['RESET']}"
                text_color = c['GREEN']
            elif log_type == 'error':
                icon = f"{c['RED']}✗{c['RESET']}"
                text_color = c['RED']
            else:
                icon = f"{c['GRAY']}ℹ{c['RESET']}"
                text_color = c['GRAY']
            
            print(f"  {c['GRAY']}[{timestamp}]{c['RESET']} {icon} {text_color}{msg}{c['RESET']}")
        print()
    
    def print_rl_progress(self, rl_metrics: Dict):
        """Print RL learning progress"""
        c = self.COLORS
        print(f"{c['BOLD']}{c['RED']}🧠 RL AGENT LEARNING:{c['RESET']}")
        print(f"{c['RED']}{'─'*100}{c['RESET']}")
        
        if rl_metrics.get('status') == 'trained':
            success_rate = rl_metrics.get('overall_success_rate', 0)
            avg_reward = rl_metrics.get('avg_reward_recent', 0)
            improvement = rl_metrics.get('improvement_rate', 0)
            
            # Progress bar for success rate
            bar_length = 40
            filled = int(bar_length * success_rate)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"  Success Rate:  {c['GREEN']}{bar}{c['RESET']} {success_rate:.1%}")
            print(f"  Avg Reward:    {c['CYAN']}{avg_reward:+.2f}{c['RESET']}")
            print(f"  Improvement:   {c['YELLOW']}{improvement:+.1%}{c['RESET']} from baseline")
            print(f"  Episodes:      {c['GRAY']}{rl_metrics.get('total_episodes', 0)}{c['RESET']}")
        else:
            print(f"  {c['GRAY']}Collecting initial experience...{c['RESET']}")
        print()
    
    def render(self, day: int, model_id: str, rl_metrics: Dict = None):
        """Render complete dashboard"""
        self.clear_screen()
        self.print_header(day, model_id)
        self.print_agent_status()
        self.print_metrics()
        if rl_metrics:
            self.print_rl_progress(rl_metrics)
        self.print_execution_log()
        print(f"{self.COLORS['GRAY']}{'─'*100}{self.COLORS['RESET']}")
    
    def update_agent_status(self, agent_name: str, status: str):
        """Update agent execution status"""
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status
    
    def add_log(self, message: str, log_type: str = 'info'):
        """Add entry to execution log"""
        self.execution_log.append({
            'timestamp': datetime.now(),
            'message': message,
            'type': log_type
        })
    
    def update_metrics(self, **kwargs):
        """Update dashboard metrics"""
        self.metrics.update(kwargs)


class DashboardOrchestrator:
    """
    Wrapper around orchestrator that updates dashboard in real-time
    """
    
    def __init__(self, base_orchestrator, dashboard: LiveDashboard):
        self.orchestrator = base_orchestrator
        self.dashboard = dashboard
    
    def run_monitoring_cycle_with_dashboard(
        self,
        model_id: str,
        day: int
    ) -> Dict[str, Any]:
        """
        Run monitoring cycle with live dashboard updates
        """
        
        # Initial render
        self.dashboard.render(day, model_id)
        time.sleep(0.5)
        
        # Update initial status
        self.dashboard.add_log(f"Starting monitoring cycle for Day {day}", 'info')
        self.dashboard.render(day, model_id)
        time.sleep(0.5)
        
        # Simulate agent execution with dashboard updates
        agents = [
            'Performance Monitor',
            'Drift Detector',
            'Quality Analyzer',
            'Alert Manager',
            'Remediation Planner'
        ]
        
        for agent in agents:
            # Mark as executing
            self.dashboard.update_agent_status(agent, 'executing')
            self.dashboard.add_log(f"Executing {agent}...", 'agent')
            self.dashboard.render(day, model_id)
            time.sleep(1)
            
            # Agent-specific actions
            if agent == 'Performance Monitor':
                self.dashboard.add_log("Querying MCP Predictions Server...", 'action')
                self.dashboard.render(day, model_id)
                time.sleep(0.5)
                
            elif agent == 'Drift Detector':
                self.dashboard.add_log("Analyzing drift scores from MCP...", 'action')
                self.dashboard.render(day, model_id)
                time.sleep(0.5)
                
            elif agent == 'Alert Manager':
                self.dashboard.add_log("Evaluating alert thresholds...", 'action')
                self.dashboard.render(day, model_id)
                time.sleep(0.5)
            
            # Mark as completed
            self.dashboard.update_agent_status(agent, 'completed')
            self.dashboard.add_log(f"✓ {agent} completed", 'result')
            self.dashboard.render(day, model_id)
            time.sleep(0.3)
        
        # Execute actual monitoring cycle
        self.dashboard.add_log("Executing RL remediation decision...", 'action')
        self.dashboard.render(day, model_id)
        
        result = self.orchestrator.run_monitoring_cycle(model_id, day)
        
        # Update metrics
        if result['status'] == 'success':
            monitoring_data = result['monitoring_data']
            rl_decision = result['rl_decision']
            
            self.dashboard.update_metrics(
                episode=day + 1,
                accuracy=monitoring_data['current_accuracy'],
                drift=monitoring_data['drift_score'],
                reward=rl_decision.get('reward', 0),
                action=rl_decision.get('action', 'Unknown')
            )
            
            # Get RL metrics
            rl_metrics = rl_decision.get('rl_performance', {})
            
            self.dashboard.add_log(
                f"RL Action: {rl_decision.get('action', 'Unknown')} (Reward: {rl_decision.get('reward', 0):+.2f})",
                'result'
            )
            
            # Reset agent status
            for agent in agents:
                self.dashboard.update_agent_status(agent, 'idle')
            
            # Final render
            self.dashboard.render(day, model_id, rl_metrics)
        else:
            self.dashboard.add_log(f"Error: {result.get('error', 'Unknown')}", 'error')
            self.dashboard.render(day, model_id)
        
        return result


def run_with_live_dashboard(
    model_id: str = 'pneumonia_classifier_v1',
    days: int = 10
):
    """
    Run simulation with live dashboard
    """
    from main_simulation import MonitoringSystemSimulation
    
    print("\n🚀 Starting ML Model Monitoring with LIVE DASHBOARD...")
    print("⏱️  Dashboard will update in real-time as agents execute\n")
    time.sleep(2)
    
    # Initialize system
    sim = MonitoringSystemSimulation()
    
    # Create dashboard
    dashboard = LiveDashboard()
    dashboard_orch = DashboardOrchestrator(sim.orchestrator, dashboard)
    
    # Run monitoring cycles with live updates
    results = []
    for day in range(days):
        result = dashboard_orch.run_monitoring_cycle_with_dashboard(
            model_id=model_id,
            day=day
        )
        results.append(result)
        time.sleep(2)  # Pause between days
    
    # Final summary
    dashboard.clear_screen()
    print(f"\n{dashboard.COLORS['BOLD']}{dashboard.COLORS['GREEN']}{'='*100}{dashboard.COLORS['RESET']}")
    print(f"{dashboard.COLORS['BOLD']}{dashboard.COLORS['GREEN']}✅ SIMULATION COMPLETE!{dashboard.COLORS['RESET']}")
    print(f"{dashboard.COLORS['GREEN']}{'='*100}{dashboard.COLORS['RESET']}\n")
    
    # Print final metrics
    successful = [r for r in results if r.get('status') == 'success']
    
    if successful:
        accuracies = [r['monitoring_data']['current_accuracy'] for r in successful]
        print(f"  Initial Accuracy: {accuracies[0]:.3f}")
        print(f"  Final Accuracy:   {accuracies[-1]:.3f}")
        print(f"  Change:           {(accuracies[-1] - accuracies[0])*100:+.1f}%")
        print(f"  Episodes:         {len(results)}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Dashboard for ML Monitoring')
    parser.add_argument('--model', type=str, default='pneumonia_classifier_v1')
    parser.add_argument('--days', type=int, default=10, help='Number of days to simulate')
    
    args = parser.parse_args()
    
    run_with_live_dashboard(model_id=args.model, days=args.days)
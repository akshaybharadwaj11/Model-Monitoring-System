# """
# ML Model Monitoring System - Main Simulation
# Runs complete 30-day demonstration with RL learning
# """

# import os
# import sys
# import json
# import argparse
# from datetime import datetime
# from pathlib import Path
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from langchain_openai import ChatOpenAI

# # Import our modules
# from data_simulator import ModelDataSimulator
# from rl_agents import RLRemediationAgent, ThresholdBandit
# from mcp_servers import MCPManager
# from controller_with_rl import ModelMonitoringOrchestrator


# class MonitoringSystemSimulation:
#     """
#     Main simulation runner for the ML Model Monitoring System
#     """
    
#     def __init__(
#         self,
#         data_dir: str = './simulated_data',
#         output_dir: str = './simulation_results',
#         use_llm: bool = False,
#         openai_api_key: str = None
#     ):
#         self.data_dir = Path(data_dir)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True, parents=True)
#         self.use_llm = use_llm
        
#         # Initialize components
#         print("Initializing ML Model Monitoring System...")
        
#         # 1. Generate simulated data
#         print("\n1. Generating simulated model data...")
#         simulator = ModelDataSimulator(seed=42)
#         self.model_data = simulator.save_to_files(self.data_dir)
        
#         # 2. Initialize MCP servers
#         print("\n2. Initializing MCP servers...")
#         self.mcp_manager = MCPManager()
#         self.mcp_manager.load_simulated_data(self.data_dir)
        
#         # 3. Initialize RL components
#         print("\n3. Initializing RL agents...")
#         self.rl_agent = RLRemediationAgent(state_dim=10, action_dim=7)
#         self.threshold_bandit = ThresholdBandit()
        
#         # Try to load pre-trained RL policy
#         policy_path = self.output_dir / 'rl_policy.pt'
#         if policy_path.exists():
#             self.rl_agent.load_policy(str(policy_path))
        
#         # 4. Initialize LLM (optional)
#         if use_llm and openai_api_key:
#             print("\n4. Initializing LLM for REAL agent orchestration...")
#             print("   ⚠️  This will make actual OpenAI API calls")
#             print("   ⚠️  Estimated cost: $1-2 per 30-day simulation")
#             print("   ⚠️  Runtime: 10-15 minutes (vs 2-3 min simulation)")
#             self.llm = ChatOpenAI(
#                 model="gpt-4o-mini",
#                 temperature=0.3,
#                 api_key=openai_api_key
#             )
#         else:
#             print("\n4. Running in SIMULATION mode (no LLM calls)")
#             print("   ✅ No API costs")
#             print("   ✅ Fast execution (2-3 minutes)")
#             print("   ✅ Deterministic results")
#             self.llm = None
        
#         # 5. Initialize orchestrator
#         print("\n5. Initializing orchestrator...")
#         self.orchestrator = ModelMonitoringOrchestrator(
#             mcp_manager=self.mcp_manager,
#             rl_remediation_agent=self.rl_agent,
#             threshold_bandit=self.threshold_bandit,
#             llm=self.llm if use_llm else None
#         )
        
#         print("\n✓ System initialization complete!\n")
    
#     def run_30_day_simulation(
#         self,
#         model_id: str = 'pneumonia_classifier_v1',
#         checkpoint_days: list = [10, 20, 30]
#     ):
#         """
#         Run complete 30-day monitoring simulation
#         """
#         print("="*80)
#         print("STARTING 30-DAY SIMULATION")
#         print("="*80)
#         print(f"Model: {model_id}")
#         print(f"Monitoring period: Day 0 to Day 30")
#         print("="*80)
        
#         results = []
        
#         # Run monitoring for each day
#         for day in range(31):  # Days 0-30
#             print(f"\n{'='*80}")
#             print(f"DAY {day}")
#             print(f"{'='*80}")
            
#             result = self.orchestrator.run_monitoring_cycle(
#                 model_id=model_id,
#                 day=day
#             )
            
#             results.append(result)
            
#             # Checkpoint evaluation
#             if day in checkpoint_days:
#                 self._print_checkpoint_summary(day, results)
        
#         # Final results
#         print("\n" + "="*80)
#         print("SIMULATION COMPLETE!")
#         print("="*80)
        
#         # Generate comprehensive report
#         self._generate_final_report(results, model_id)
        
#         # Save RL policy
#         policy_path = self.output_dir / 'rl_policy.pt'
#         self.rl_agent.save_policy(str(policy_path))
        
#         return results
    
#     def _print_checkpoint_summary(self, day: int, results: list):
#         """Print checkpoint summary"""
#         print(f"\n{'─'*80}")
#         print(f"CHECKPOINT: Day {day} Summary")
#         print(f"{'─'*80}")
        
#         recent_results = results[-10:] if len(results) >= 10 else results
        
#         # Calculate metrics
#         avg_accuracy = np.mean([
#             r['monitoring_data']['current_accuracy'] 
#             for r in recent_results
#         ])
        
#         avg_drift = np.mean([
#             r['monitoring_data']['drift_score'] 
#             for r in recent_results
#         ])
        
#         avg_reward = np.mean([
#             r['rl_decision'].get('reward', 0)
#             for r in recent_results
#             if 'rl_decision' in r
#         ])
        
#         print(f"Recent Average Accuracy: {avg_accuracy:.3f}")
#         print(f"Recent Average Drift: {avg_drift:.3f}")
#         print(f"Recent Average RL Reward: {avg_reward:.2f}")
        
#         # RL Performance
#         rl_metrics = self.rl_agent.get_performance_metrics()
#         if rl_metrics['status'] == 'trained':
#             print(f"\nRL Agent Performance:")
#             print(f"  Episodes: {rl_metrics['total_episodes']}")
#             print(f"  Success Rate: {rl_metrics['overall_success_rate']:.1%}")
#             print(f"  Avg Reward: {rl_metrics['avg_reward_recent']:.2f}")
        
#         print(f"{'─'*80}\n")
    
#     def _generate_final_report(self, results: list, model_id: str):
#         """Generate comprehensive final report"""
#         print("\n" + "="*80)
#         print("FINAL REPORT")
#         print("="*80)
        
#         # Extract data
#         days = [r['day'] for r in results]
#         accuracies = [r['monitoring_data']['current_accuracy'] for r in results]
#         drift_scores = [r['monitoring_data']['drift_score'] for r in results]
#         rewards = [r['rl_decision'].get('reward', 0) for r in results if 'rl_decision' in r]
        
#         # System metrics
#         system_summary = self.orchestrator.get_system_summary()
        
#         print("\n1. OVERALL PERFORMANCE")
#         print(f"   Initial Accuracy: {accuracies[0]:.3f}")
#         print(f"   Final Accuracy: {accuracies[-1]:.3f}")
#         print(f"   Accuracy Change: {(accuracies[-1]-accuracies[0])*100:+.1f}%")
#         print(f"   Max Drift Score: {max(drift_scores):.3f}")
        
#         print("\n2. RL AGENT LEARNING")
#         rl_perf = system_summary['rl_agent_performance']
#         if rl_perf['status'] == 'trained':
#             print(f"   Total Episodes: {rl_perf['total_episodes']}")
#             print(f"   Success Rate: {rl_perf['overall_success_rate']:.1%}")
#             print(f"   Avg Reward (All): {rl_perf['avg_reward_all_time']:.2f}")
#             print(f"   Avg Reward (Recent): {rl_perf['avg_reward_recent']:.2f}")
#             print(f"   Improvement: {rl_perf['improvement_rate']:.1%}")
            
#             print(f"\n   Action Distribution:")
#             for action, count in rl_perf['action_distribution'].items():
#                 success_rate = rl_perf['action_success_rates'].get(action, 0)
#                 print(f"     {action}: {count} times ({success_rate:.1%} success)")
        
#         print("\n3. BUSINESS IMPACT")
#         monitoring_metrics = system_summary['monitoring_metrics']
#         print(f"   Successful Remediations: {monitoring_metrics['successful_remediations']}")
#         print(f"   Failed Remediations: {monitoring_metrics['failed_remediations']}")
#         print(f"   Cost Saved: ${monitoring_metrics['total_cost_saved']:,.0f}")
        
#         print("\n4. THRESHOLD OPTIMIZATION")
#         threshold_stats = system_summary['threshold_optimization']
#         print(f"   Best Threshold: {threshold_stats['best_threshold']}")
#         print(f"   Total Selections: {threshold_stats['total_selections']}")
#         print(f"   Cumulative Reward: {threshold_stats['cumulative_reward']:.2f}")
        
#         # Generate plots
#         self._generate_plots(days, accuracies, drift_scores, rewards, model_id)
        
#         # Save JSON report
#         report_path = self.output_dir / f'final_report_{model_id}.json'
#         with open(report_path, 'w') as f:
#             # Convert numpy types for JSON serialization
#             serializable_results = []
#             for r in results:
#                 r_copy = r.copy()
#                 for key in ['monitoring_data', 'analysis', 'rl_decision']:
#                     if key in r_copy:
#                         r_copy[key] = self._convert_to_serializable(r_copy[key])
#                 serializable_results.append(r_copy)
            
#             json.dump({
#                 'model_id': model_id,
#                 'simulation_date': datetime.now().isoformat(),
#                 'results': serializable_results,
#                 'system_summary': self._convert_to_serializable(system_summary)
#             }, f, indent=2)
        
#         print(f"\n✓ Report saved to: {report_path}")
#         print("="*80)
    
#     def _convert_to_serializable(self, obj):
#         """Convert numpy types to Python native types"""
#         if isinstance(obj, dict):
#             return {k: self._convert_to_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [self._convert_to_serializable(i) for i in obj]
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj
    
#     def _generate_plots(
#         self,
#         days: list,
#         accuracies: list,
#         drift_scores: list,
#         rewards: list,
#         model_id: str
#     ):
#         """Generate visualization plots"""
#         print("\n5. GENERATING VISUALIZATIONS")
        
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#         fig.suptitle(f'ML Model Monitoring System - 30 Day Simulation\nModel: {model_id}', 
#                      fontsize=16, fontweight='bold')
        
#         # Plot 1: Accuracy over time
#         ax1 = axes[0, 0]
#         ax1.plot(days, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
#         ax1.axhline(y=0.85, color='g', linestyle='--', label='Healthy Threshold')
#         ax1.axhline(y=0.80, color='orange', linestyle='--', label='Warning Threshold')
#         ax1.axhline(y=0.75, color='r', linestyle='--', label='Critical Threshold')
#         ax1.set_xlabel('Day')
#         ax1.set_ylabel('Accuracy')
#         ax1.set_title('Model Accuracy Over Time')
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
        
#         # Plot 2: Drift scores
#         ax2 = axes[0, 1]
#         ax2.plot(days, drift_scores, 'r-', linewidth=2, marker='s', markersize=3)
#         ax2.axhline(y=0.15, color='g', linestyle='--', label='Low Drift')
#         ax2.axhline(y=0.25, color='orange', linestyle='--', label='Medium Drift')
#         ax2.axhline(y=0.35, color='r', linestyle='--', label='High Drift')
#         ax2.set_xlabel('Day')
#         ax2.set_ylabel('Drift Score')
#         ax2.set_title('Data Drift Over Time')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # Plot 3: RL Rewards (cumulative)
#         ax3 = axes[1, 0]
#         cumulative_rewards = np.cumsum(rewards)
#         ax3.plot(range(len(cumulative_rewards)), cumulative_rewards, 'g-', linewidth=2)
#         ax3.set_xlabel('Episode')
#         ax3.set_ylabel('Cumulative Reward')
#         ax3.set_title('RL Agent Learning Progress')
#         ax3.grid(True, alpha=0.3)
        
#         # Plot 4: Action distribution
#         ax4 = axes[1, 1]
#         action_counts = self.rl_agent.action_counts
#         action_names = [a.name for a in self.rl_agent.ACTIONS]
#         ax4.bar(range(len(action_names)), action_counts, color='steelblue')
#         ax4.set_xlabel('Action')
#         ax4.set_ylabel('Count')
#         ax4.set_title('RL Agent Action Distribution')
#         ax4.set_xticks(range(len(action_names)))
#         ax4.set_xticklabels(action_names, rotation=45, ha='right')
#         ax4.grid(True, alpha=0.3, axis='y')
        
#         plt.tight_layout()
        
#         # Save plot
#         plot_path = self.output_dir / f'simulation_plots_{model_id}.png'
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"   Plots saved to: {plot_path}")
        
#         plt.close()


# def main():
#     """Main entry point"""
#     parser = argparse.ArgumentParser(
#         description='ML Model Monitoring System - 30 Day Simulation'
#     )
    
#     parser.add_argument(
#         '--model',
#         type=str,
#         default='pneumonia_classifier_v1',
#         choices=[
#             'pneumonia_classifier_v1',
#             'fraud_detector_v2',
#             'object_detector_v1'
#         ],
#         help='Model to monitor'
#     )
    
#     parser.add_argument(
#         '--use-llm',
#         action='store_true',
#         help='Use actual LLM for agent reasoning (requires OpenAI key)'
#     )
    
#     parser.add_argument(
#         '--output-dir',
#         type=str,
#         default='./simulation_results',
#         help='Output directory for results'
#     )
    
#     args = parser.parse_args()
    
#     # Get OpenAI key if using LLM
#     openai_key = None
#     if args.use_llm:
#         openai_key = os.getenv('OPENAI_API_KEY')
#         if not openai_key:
#             print("ERROR: OPENAI_API_KEY environment variable not set")
#             print("Run: export OPENAI_API_KEY='your-key-here'")
#             sys.exit(1)
    
#     # Run simulation
#     print("\n" + "="*80)
#     print("ML MODEL MONITORING SYSTEM")
#     print("Agentic AI with Reinforcement Learning")
#     print("="*80)
    
#     simulation = MonitoringSystemSimulation(
#         output_dir=args.output_dir,
#         use_llm=args.use_llm,
#         openai_api_key=openai_key
#     )
    
#     results = simulation.run_30_day_simulation(
#         model_id=args.model,
#         checkpoint_days=[10, 20, 30]
#     )
    
#     print("\n" + "="*80)
#     print("✓ SIMULATION COMPLETE")
#     print("="*80)
#     print(f"\nResults saved to: {args.output_dir}")
#     print("\nFiles generated:")
#     print(f"  - final_report_{args.model}.json")
#     print(f"  - simulation_plots_{args.model}.png")
#     print(f"  - rl_policy.pt (trained RL agent)")
#     print("\nTo view plots: open simulation_plots_*.png")
#     print("To analyze results: check final_report_*.json")


# if __name__ == "__main__":
#     main()

"""
ML Model Monitoring System - Main Simulation
Runs complete 30-day demonstration with RL learning
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI

# Import our modules
from data_simulator import ModelDataSimulator
from rl_agents import RLRemediationAgent, ThresholdBandit
from mcp_servers import MCPManager
from controller_with_rl import ModelMonitoringOrchestrator


class MonitoringSystemSimulation:
    """
    Main simulation runner for the ML Model Monitoring System
    """
    
    def __init__(
        self,
        data_dir: str = './simulated_data',
        output_dir: str = './simulation_results',
        use_llm: bool = False,
        openai_api_key: str = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.use_llm = use_llm
        
        # Initialize components
        print("Initializing ML Model Monitoring System...")
        
        # 1. Generate simulated data
        print("\n1. Generating simulated model data...")
        simulator = ModelDataSimulator(seed=42)
        self.model_data = simulator.save_to_files(self.data_dir)
        
        # 2. Initialize MCP servers
        print("\n2. Initializing MCP servers...")
        self.mcp_manager = MCPManager()
        self.mcp_manager.load_simulated_data(self.data_dir)
        
        # 3. Initialize RL components
        print("\n3. Initializing RL agents...")
        self.rl_agent = RLRemediationAgent(state_dim=10, action_dim=7)
        self.threshold_bandit = ThresholdBandit()
        
        # Try to load pre-trained RL policy
        policy_path = self.output_dir / 'rl_policy.pt'
        if policy_path.exists():
            self.rl_agent.load_policy(str(policy_path))
        
        # 4. Initialize LLM (optional)
        if use_llm and openai_api_key:
            print("\n4. Initializing LLM for REAL agent orchestration...")
            print("   ⚠️  This will make actual OpenAI API calls")
            print("   ⚠️  Estimated cost: $1-2 per 30-day simulation")
            print("   ⚠️  Runtime: 10-15 minutes (vs 2-3 min simulation)")
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                api_key=openai_api_key
            )
        else:
            print("\n4. Running in SIMULATION mode (no LLM calls)")
            print("   ✅ No API costs")
            print("   ✅ Fast execution (2-3 minutes)")
            print("   ✅ Deterministic results")
            self.llm = None
        
        # 5. Initialize orchestrator
        print("\n5. Initializing orchestrator...")
        self.orchestrator = ModelMonitoringOrchestrator(
            mcp_manager=self.mcp_manager,
            rl_remediation_agent=self.rl_agent,
            threshold_bandit=self.threshold_bandit,
            llm=self.llm if use_llm else None
        )
        
        print("\n✓ System initialization complete!\n")
    
    def run_30_day_simulation(
        self,
        model_id: str = 'pneumonia_classifier_v1',
        checkpoint_days: list = [10, 20, 30]
    ):
        """
        Run complete 30-day monitoring simulation
        """
        print("="*80)
        print("STARTING 30-DAY SIMULATION")
        print("="*80)
        print(f"Model: {model_id}")
        print(f"Monitoring period: Day 0 to Day 30")
        print("="*80)
        
        results = []
        
        # Run monitoring for each day
        for day in range(31):  # Days 0-30
            print(f"\n{'='*80}")
            print(f"DAY {day}")
            print(f"{'='*80}")
            
            result = self.orchestrator.run_monitoring_cycle(
                model_id=model_id,
                day=day
            )
            
            results.append(result)
            
            # Checkpoint evaluation
            if day in checkpoint_days:
                self._print_checkpoint_summary(day, results)
        
        # Final results
        print("\n" + "="*80)
        print("SIMULATION COMPLETE!")
        print("="*80)
        
        # Generate comprehensive report
        self._generate_final_report(results, model_id)
        
        # Save RL policy
        policy_path = self.output_dir / 'rl_policy.pt'
        self.rl_agent.save_policy(str(policy_path))
        
        return results
    
    def _print_checkpoint_summary(self, day: int, results: list):
        """Print checkpoint summary"""
        print(f"\n{'─'*80}")
        print(f"CHECKPOINT: Day {day} Summary")
        print(f"{'─'*80}")
        
        recent_results = results[-10:] if len(results) >= 10 else results
        
        # Filter only successful results
        successful_results = [
            r for r in recent_results 
            if r.get('status') == 'success' and 'monitoring_data' in r
        ]
        
        if not successful_results:
            print("No successful results to summarize yet")
            print(f"{'─'*80}\n")
            return
        
        # Calculate metrics
        avg_accuracy = np.mean([
            r['monitoring_data']['current_accuracy'] 
            for r in successful_results
        ])
        
        avg_drift = np.mean([
            r['monitoring_data']['drift_score'] 
            for r in successful_results
        ])
        
        avg_reward = np.mean([
            r['rl_decision'].get('reward', 0)
            for r in successful_results
            if 'rl_decision' in r
        ])
        
        print(f"Recent Average Accuracy: {avg_accuracy:.3f}")
        print(f"Recent Average Drift: {avg_drift:.3f}")
        print(f"Recent Average RL Reward: {avg_reward:.2f}")
        
        # RL Performance
        rl_metrics = self.rl_agent.get_performance_metrics()
        if rl_metrics['status'] == 'trained':
            print(f"\nRL Agent Performance:")
            print(f"  Episodes: {rl_metrics['total_episodes']}")
            print(f"  Success Rate: {rl_metrics['overall_success_rate']:.1%}")
            print(f"  Avg Reward: {rl_metrics['avg_reward_recent']:.2f}")
        
        print(f"{'─'*80}\n")
    
    def _generate_final_report(self, results: list, model_id: str):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        
        # Filter successful results
        successful_results = [
            r for r in results 
            if r.get('status') == 'success' and 'monitoring_data' in r
        ]
        
        if not successful_results:
            print("\n⚠️  No successful results to report")
            print("="*80)
            return
        
        # Extract data
        days = [r['day'] for r in successful_results]
        accuracies = [r['monitoring_data']['current_accuracy'] for r in successful_results]
        drift_scores = [r['monitoring_data']['drift_score'] for r in successful_results]
        rewards = [
            r['rl_decision'].get('reward', 0) 
            for r in successful_results 
            if 'rl_decision' in r
        ]
        
        # System metrics
        system_summary = self.orchestrator.get_system_summary()
        
        print("\n1. OVERALL PERFORMANCE")
        print(f"   Initial Accuracy: {accuracies[0]:.3f}")
        print(f"   Final Accuracy: {accuracies[-1]:.3f}")
        print(f"   Accuracy Change: {(accuracies[-1]-accuracies[0])*100:+.1f}%")
        print(f"   Max Drift Score: {max(drift_scores):.3f}")
        
        print("\n2. RL AGENT LEARNING")
        rl_perf = system_summary['rl_agent_performance']
        if rl_perf['status'] == 'trained':
            print(f"   Total Episodes: {rl_perf['total_episodes']}")
            print(f"   Success Rate: {rl_perf['overall_success_rate']:.1%}")
            print(f"   Avg Reward (All): {rl_perf['avg_reward_all_time']:.2f}")
            print(f"   Avg Reward (Recent): {rl_perf['avg_reward_recent']:.2f}")
            print(f"   Improvement: {rl_perf['improvement_rate']:.1%}")
            
            print(f"\n   Action Distribution:")
            for action, count in rl_perf['action_distribution'].items():
                success_rate = rl_perf['action_success_rates'].get(action, 0)
                print(f"     {action}: {count} times ({success_rate:.1%} success)")
        
        print("\n3. BUSINESS IMPACT")
        monitoring_metrics = system_summary['monitoring_metrics']
        print(f"   Successful Remediations: {monitoring_metrics['successful_remediations']}")
        print(f"   Failed Remediations: {monitoring_metrics['failed_remediations']}")
        print(f"   Cost Saved: ${monitoring_metrics['total_cost_saved']:,.0f}")
        
        print("\n4. THRESHOLD OPTIMIZATION")
        threshold_stats = system_summary['threshold_optimization']
        print(f"   Best Threshold: {threshold_stats['best_threshold']}")
        print(f"   Total Selections: {threshold_stats['total_selections']}")
        print(f"   Cumulative Reward: {threshold_stats['cumulative_reward']:.2f}")
        
        # Generate plots
        self._generate_plots(days, accuracies, drift_scores, rewards, model_id)
        
        # Save JSON report
        report_path = self.output_dir / f'final_report_{model_id}.json'
        with open(report_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = []
            for r in results:
                r_copy = r.copy()
                for key in ['monitoring_data', 'analysis', 'rl_decision']:
                    if key in r_copy:
                        r_copy[key] = self._convert_to_serializable(r_copy[key])
                serializable_results.append(r_copy)
            
            json.dump({
                'model_id': model_id,
                'simulation_date': datetime.now().isoformat(),
                'results': serializable_results,
                'system_summary': self._convert_to_serializable(system_summary)
            }, f, indent=2)
        
        print(f"\n✓ Report saved to: {report_path}")
        print("="*80)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_plots(
        self,
        days: list,
        accuracies: list,
        drift_scores: list,
        rewards: list,
        model_id: str
    ):
        """Generate visualization plots"""
        print("\n5. GENERATING VISUALIZATIONS")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ML Model Monitoring System - 30 Day Simulation\nModel: {model_id}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy over time
        ax1 = axes[0, 0]
        ax1.plot(days, accuracies, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.axhline(y=0.85, color='g', linestyle='--', label='Healthy Threshold')
        ax1.axhline(y=0.80, color='orange', linestyle='--', label='Warning Threshold')
        ax1.axhline(y=0.75, color='r', linestyle='--', label='Critical Threshold')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drift scores
        ax2 = axes[0, 1]
        ax2.plot(days, drift_scores, 'r-', linewidth=2, marker='s', markersize=3)
        ax2.axhline(y=0.15, color='g', linestyle='--', label='Low Drift')
        ax2.axhline(y=0.25, color='orange', linestyle='--', label='Medium Drift')
        ax2.axhline(y=0.35, color='r', linestyle='--', label='High Drift')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Drift Score')
        ax2.set_title('Data Drift Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RL Rewards (cumulative)
        ax3 = axes[1, 0]
        cumulative_rewards = np.cumsum(rewards)
        ax3.plot(range(len(cumulative_rewards)), cumulative_rewards, 'g-', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title('RL Agent Learning Progress')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Action distribution
        ax4 = axes[1, 1]
        action_counts = self.rl_agent.action_counts
        action_names = [a.name for a in self.rl_agent.ACTIONS]
        ax4.bar(range(len(action_names)), action_counts, color='steelblue')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Count')
        ax4.set_title('RL Agent Action Distribution')
        ax4.set_xticks(range(len(action_names)))
        ax4.set_xticklabels(action_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'simulation_plots_{model_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Plots saved to: {plot_path}")
        
        plt.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='ML Model Monitoring System - 30 Day Simulation'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='pneumonia_classifier_v1',
        choices=[
            'pneumonia_classifier_v1',
            'fraud_detector_v2',
            'object_detector_v1'
        ],
        help='Model to monitor'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use actual LLM for agent reasoning (requires OpenAI key)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./simulation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Get OpenAI key if using LLM
    openai_key = None
    if args.use_llm:
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("ERROR: OPENAI_API_KEY environment variable not set")
            print("Run: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
    
    # Run simulation
    print("\n" + "="*80)
    print("ML MODEL MONITORING SYSTEM")
    print("Agentic AI with Reinforcement Learning")
    print("="*80)
    
    simulation = MonitoringSystemSimulation(
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        openai_api_key=openai_key
    )
    
    results = simulation.run_30_day_simulation(
        model_id=args.model,
        checkpoint_days=[10, 20, 30]
    )
    
    print("\n" + "="*80)
    print("✓ SIMULATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nFiles generated:")
    print(f"  - final_report_{args.model}.json")
    print(f"  - simulation_plots_{args.model}.png")
    print(f"  - rl_policy.pt (trained RL agent)")
    print("\nTo view plots: open simulation_plots_*.png")
    print("To analyze results: check final_report_*.json")


if __name__ == "__main__":
    main()
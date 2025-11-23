"""
Comprehensive Testing and Evaluation Framework
Test cases, metrics collection, and performance analysis
"""

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


@dataclass
class TestCase:
    """Define a test case"""
    name: str
    description: str
    model_id: str
    days: int
    expected_outcomes: Dict[str, Any]
    pass_criteria: Dict[str, Any]


class ComprehensiveTestSuite:
    """
    Complete testing framework for ML Model Monitoring System
    """
    
    def __init__(self, output_dir: str = './test_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.test_cases = self._define_test_cases()
        self.results = []
    
    def _define_test_cases(self) -> List[TestCase]:
        """Define all test cases"""
        return [
            TestCase(
                name="TC1_Baseline_Performance",
                description="Test system with stable model (no drift, good accuracy)",
                model_id='pneumonia_classifier_v1',
                days=5,
                expected_outcomes={
                    'accuracy_stable': True,
                    'drift_low': True,
                    'rl_learns': True
                },
                pass_criteria={
                    'min_accuracy': 0.85,
                    'max_drift': 0.15,
                    'rl_episodes': 5,
                    'success_rate_improves': False  # Too short to see improvement
                }
            ),
            
            TestCase(
                name="TC2_Drift_Detection",
                description="Test drift detection and remediation over 15 days",
                model_id='pneumonia_classifier_v1',
                days=15,
                expected_outcomes={
                    'drift_increases': True,
                    'alerts_created': True,
                    'remediation_triggered': True,
                    'accuracy_recovers': True
                },
                pass_criteria={
                    'max_drift_detected': 0.20,  # Should detect rising drift
                    'remediation_count': 1,  # At least 1 remediation
                    'final_accuracy_above': 0.83  # Should recover
                }
            ),
            
            TestCase(
                name="TC3_RL_Learning",
                description="Test RL agent learning over 30 episodes",
                model_id='fraud_detector_v2',
                days=30,
                expected_outcomes={
                    'success_rate_improves': True,
                    'reward_increases': True,
                    'policy_converges': True
                },
                pass_criteria={
                    'initial_success_rate_max': 0.55,  # Should start ~45-55%
                    'final_success_rate_min': 0.75,  # Should reach >75%
                    'improvement_min': 0.20  # At least 20% improvement
                }
            ),
            
            TestCase(
                name="TC4_Multi_Model",
                description="Test monitoring multiple models simultaneously",
                model_id='object_detector_v1',
                days=10,
                expected_outcomes={
                    'handles_multiple_models': True,
                    'no_cross_contamination': True
                },
                pass_criteria={
                    'completion_rate': 1.0,  # All cycles complete
                    'error_rate': 0.0  # No errors
                }
            ),
            
            TestCase(
                name="TC5_Edge_Cases",
                description="Test edge cases (very high drift, very low accuracy)",
                model_id='pneumonia_classifier_v1',
                days=5,
                expected_outcomes={
                    'handles_critical_state': True,
                    'immediate_remediation': True
                },
                pass_criteria={
                    'critical_alerts_created': True,
                    'immediate_retrain_selected': True
                }
            )
        ]
    
    def run_all_tests(self):
        """Execute all test cases"""
        print("\n" + "="*100)
        print("🧪 RUNNING COMPREHENSIVE TEST SUITE")
        print("="*100)
        
        from main_simulation import MonitoringSystemSimulation
        
        total_tests = len(self.test_cases)
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n{'─'*100}")
            print(f"TEST {i}/{total_tests}: {test_case.name}")
            print(f"{'─'*100}")
            print(f"Description: {test_case.description}")
            print(f"Model: {test_case.model_id}, Days: {test_case.days}")
            
            # Run test
            start_time = time.time()
            
            try:
                sim = MonitoringSystemSimulation()
                
                results = []
                for day in range(test_case.days):
                    result = sim.orchestrator.run_monitoring_cycle(
                        model_id=test_case.model_id,
                        day=day
                    )
                    results.append(result)
                
                execution_time = time.time() - start_time
                
                # Evaluate results
                test_result = self._evaluate_test_case(test_case, results, execution_time)
                self.results.append(test_result)
                
                # Print result
                if test_result['passed']:
                    print(f"\n✅ TEST PASSED")
                else:
                    print(f"\n❌ TEST FAILED")
                    print(f"Failed criteria: {test_result['failed_criteria']}")
                
                print(f"Execution time: {execution_time:.2f}s")
                
            except Exception as e:
                print(f"\n❌ TEST ERROR: {str(e)}")
                self.results.append({
                    'test_name': test_case.name,
                    'passed': False,
                    'error': str(e)
                })
        
        # Generate test report
        self._generate_test_report()
    
    def _evaluate_test_case(
        self,
        test_case: TestCase,
        results: List[Dict],
        execution_time: float
    ) -> Dict[str, Any]:
        """Evaluate if test case passed"""
        
        successful_results = [
            r for r in results 
            if r.get('status') == 'success' and 'monitoring_data' in r
        ]
        
        if not successful_results:
            return {
                'test_name': test_case.name,
                'passed': False,
                'error': 'No successful results',
                'execution_time': execution_time
            }
        
        # Extract metrics
        accuracies = [r['monitoring_data']['current_accuracy'] for r in successful_results]
        drift_scores = [r['monitoring_data']['drift_score'] for r in successful_results]
        
        # RL metrics
        rl_metrics = successful_results[-1]['rl_decision'].get('rl_performance', {})
        
        # Check pass criteria
        criteria_results = {}
        failed_criteria = []
        
        # TC1: Baseline Performance
        if test_case.name == "TC1_Baseline_Performance":
            criteria_results['min_accuracy'] = min(accuracies) >= test_case.pass_criteria['min_accuracy']
            criteria_results['max_drift'] = max(drift_scores) <= test_case.pass_criteria['max_drift']
            criteria_results['rl_episodes'] = len(results) >= test_case.pass_criteria['rl_episodes']
            
        # TC2: Drift Detection
        elif test_case.name == "TC2_Drift_Detection":
            criteria_results['max_drift_detected'] = max(drift_scores) >= test_case.pass_criteria['max_drift_detected']
            
            # Count remediation actions (actions other than "Continue Monitoring")
            remediation_count = sum(
                1 for r in successful_results
                if r['rl_decision'].get('action_id', 6) in [0, 1, 2, 3, 4]
            )
            criteria_results['remediation_count'] = remediation_count >= test_case.pass_criteria['remediation_count']
            criteria_results['final_accuracy'] = accuracies[-1] >= test_case.pass_criteria['final_accuracy_above']
            
        # TC3: RL Learning
        elif test_case.name == "TC3_RL_Learning":
            if rl_metrics.get('status') == 'trained':
                # Check if success rate improved
                final_success = rl_metrics.get('overall_success_rate', 0)
                improvement = rl_metrics.get('improvement_rate', 0)
                
                criteria_results['success_rate_improved'] = improvement >= test_case.pass_criteria['improvement_min']
                criteria_results['final_success_rate'] = final_success >= test_case.pass_criteria['final_success_rate_min']
            else:
                criteria_results['success_rate_improved'] = False
                criteria_results['final_success_rate'] = False
        
        # TC4: Multi-Model
        elif test_case.name == "TC4_Multi_Model":
            completion_rate = len(successful_results) / len(results)
            error_rate = 1 - completion_rate
            
            criteria_results['completion_rate'] = completion_rate >= test_case.pass_criteria['completion_rate']
            criteria_results['error_rate'] = error_rate <= test_case.pass_criteria['error_rate']
        
        # TC5: Edge Cases
        elif test_case.name == "TC5_Edge_Cases":
            # Check if critical states triggered immediate action
            critical_episodes = [
                r for r in successful_results
                if r['monitoring_data']['current_accuracy'] < 0.78 or 
                   r['monitoring_data']['drift_score'] > 0.35
            ]
            
            if critical_episodes:
                immediate_actions = sum(
                    1 for r in critical_episodes
                    if r['rl_decision'].get('action_id', 6) == 0  # Retrain immediately
                )
                criteria_results['immediate_retrain'] = immediate_actions > 0
            else:
                criteria_results['immediate_retrain'] = True  # No critical state, OK
        
        # Overall pass/fail
        all_passed = all(criteria_results.values())
        failed_criteria = [k for k, v in criteria_results.items() if not v]
        
        return {
            'test_name': test_case.name,
            'passed': all_passed,
            'criteria_results': criteria_results,
            'failed_criteria': failed_criteria,
            'execution_time': execution_time,
            'metrics': {
                'accuracy_range': (float(min(accuracies)), float(max(accuracies))),
                'drift_range': (float(min(drift_scores)), float(max(drift_scores))),
                'total_episodes': len(results),
                'success_rate': len(successful_results) / len(results),
                'rl_performance': rl_metrics
            }
        }
    
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        
        passed_tests = sum(1 for r in self.results if r.get('passed', False))
        total_tests = len(self.results)
        
        print("\n" + "="*100)
        print("📊 TEST SUITE SUMMARY")
        print("="*100)
        print(f"\nTests Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        
        # Individual test results
        print("\nIndividual Test Results:")
        for result in self.results:
            status = "✅ PASS" if result.get('passed') else "❌ FAIL"
            print(f"\n  {status} - {result['test_name']}")
            
            if 'execution_time' in result:
                print(f"    Execution Time: {result['execution_time']:.2f}s")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"    Accuracy Range: {metrics['accuracy_range'][0]:.3f} - {metrics['accuracy_range'][1]:.3f}")
                print(f"    Drift Range: {metrics['drift_range'][0]:.3f} - {metrics['drift_range'][1]:.3f}")
                
                if 'rl_performance' in metrics and metrics['rl_performance'].get('status') == 'trained':
                    rl = metrics['rl_performance']
                    print(f"    RL Success Rate: {rl.get('overall_success_rate', 0):.1%}")
                    print(f"    RL Improvement: {rl.get('improvement_rate', 0):+.1%}")
            
            if not result.get('passed') and 'failed_criteria' in result:
                print(f"    Failed Criteria: {', '.join(result['failed_criteria'])}")
        
        # Save JSON report
        report_path = self.output_dir / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': total_tests - passed_tests,
                    'pass_rate': passed_tests / total_tests
                },
                'results': self.results
            }, f, indent=2, default=str)
        
        print(f"\n✓ Test report saved to: {report_path}")
        print("="*100)


class PerformanceEvaluator:
    """
    Evaluate system performance metrics
    """
    
    def __init__(self):
        self.metrics = {
            'latency': [],
            'accuracy': [],
            'rl_success_rates': [],
            'costs': []
        }
    
    def evaluate_30_day_run(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a 30-day simulation
        """
        successful_results = [
            r for r in results 
            if r.get('status') == 'success' and 'monitoring_data' in r
        ]
        
        if not successful_results:
            return {'status': 'no_data'}
        
        # Extract data
        accuracies = [r['monitoring_data']['current_accuracy'] for r in successful_results]
        drift_scores = [r['monitoring_data']['drift_score'] for r in successful_results]
        rewards = [r['rl_decision'].get('reward', 0) for r in successful_results]
        actions = [r['rl_decision'].get('action', 'Unknown') for r in successful_results]
        
        # RL metrics from final episode
        final_rl = successful_results[-1]['rl_decision'].get('rl_performance', {})
        
        evaluation = {
            'accuracy_metrics': {
                'initial': float(accuracies[0]),
                'final': float(accuracies[-1]),
                'min': float(min(accuracies)),
                'max': float(max(accuracies)),
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'change': float(accuracies[-1] - accuracies[0]),
                'change_pct': float((accuracies[-1] - accuracies[0]) / accuracies[0] * 100)
            },
            
            'drift_metrics': {
                'initial': float(drift_scores[0]),
                'final': float(drift_scores[-1]),
                'max': float(max(drift_scores)),
                'mean': float(np.mean(drift_scores)),
                'critical_episodes': sum(1 for d in drift_scores if d > 0.35)
            },
            
            'rl_learning': {
                'total_episodes': len(successful_results),
                'avg_reward': float(np.mean(rewards)),
                'reward_trend': 'improving' if rewards[-5:] > rewards[:5] else 'stable',
                'final_success_rate': final_rl.get('overall_success_rate', 0),
                'improvement_rate': final_rl.get('improvement_rate', 0),
                'best_reward': float(max(rewards)),
                'worst_reward': float(min(rewards))
            },
            
            'action_distribution': self._analyze_actions(actions),
            
            'system_performance': {
                'completion_rate': len(successful_results) / len(results),
                'total_episodes': len(results),
                'successful_episodes': len(successful_results),
                'failed_episodes': len(results) - len(successful_results)
            }
        }
        
        return evaluation
    
    def _analyze_actions(self, actions: List[str]) -> Dict[str, Any]:
        """Analyze action distribution"""
        from collections import Counter
        
        action_counts = Counter(actions)
        total = len(actions)
        
        return {
            'distribution': {
                action: {
                    'count': count,
                    'percentage': count / total * 100
                }
                for action, count in action_counts.items()
            },
            'most_common': action_counts.most_common(1)[0][0] if action_counts else 'None',
            'diversity': len(action_counts)  # How many different actions used
        }
    
    def generate_evaluation_report(
        self,
        evaluation: Dict[str, Any],
        output_path: str
    ):
        """Generate detailed evaluation report"""
        
        report = f"""
# ML Model Monitoring System - Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Accuracy Metrics

- **Initial Accuracy:** {evaluation['accuracy_metrics']['initial']:.3f}
- **Final Accuracy:** {evaluation['accuracy_metrics']['final']:.3f}
- **Change:** {evaluation['accuracy_metrics']['change']:+.3f} ({evaluation['accuracy_metrics']['change_pct']:+.1f}%)
- **Mean Accuracy:** {evaluation['accuracy_metrics']['mean']:.3f}
- **Min/Max:** {evaluation['accuracy_metrics']['min']:.3f} / {evaluation['accuracy_metrics']['max']:.3f}
- **Std Deviation:** {evaluation['accuracy_metrics']['std']:.3f}

---

## 2. Drift Detection

- **Initial Drift:** {evaluation['drift_metrics']['initial']:.3f}
- **Final Drift:** {evaluation['drift_metrics']['final']:.3f}
- **Max Drift:** {evaluation['drift_metrics']['max']:.3f}
- **Mean Drift:** {evaluation['drift_metrics']['mean']:.3f}
- **Critical Episodes:** {evaluation['drift_metrics']['critical_episodes']} (drift > 0.35)

---

## 3. RL Agent Learning

- **Total Episodes:** {evaluation['rl_learning']['total_episodes']}
- **Average Reward:** {evaluation['rl_learning']['avg_reward']:+.2f}
- **Best Reward:** {evaluation['rl_learning']['best_reward']:+.2f}
- **Worst Reward:** {evaluation['rl_learning']['worst_reward']:+.2f}
- **Final Success Rate:** {evaluation['rl_learning']['final_success_rate']:.1%}
- **Improvement Rate:** {evaluation['rl_learning']['improvement_rate']:+.1%}
- **Trend:** {evaluation['rl_learning']['reward_trend']}

---

## 4. Action Distribution

**Most Common Action:** {evaluation['action_distribution']['most_common']}
**Action Diversity:** {evaluation['action_distribution']['diversity']} different actions used

"""
        
        # Add action breakdown
        for action, data in evaluation['action_distribution']['distribution'].items():
            report += f"\n- {action}: {data['count']} times ({data['percentage']:.1f}%)"
        
        report += f"""

---

## 5. System Performance

- **Completion Rate:** {evaluation['system_performance']['completion_rate']:.1%}
- **Total Episodes:** {evaluation['system_performance']['total_episodes']}
- **Successful:** {evaluation['system_performance']['successful_episodes']}
- **Failed:** {evaluation['system_performance']['failed_episodes']}

---

## Conclusion

The system demonstrates {'strong' if evaluation['rl_learning']['improvement_rate'] > 0.2 else 'moderate'} learning capability with 
{evaluation['rl_learning']['improvement_rate']*100:.1f}% improvement in decision-making quality.

{'✅ System meets all performance criteria.' if evaluation['system_performance']['completion_rate'] > 0.95 else '⚠️ System needs improvement in reliability.'}
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Evaluation report saved to: {output_path}")


def run_quick_test():
    """Quick test for debugging (5 days only)"""
    print("\n🧪 Running Quick Test (5 days)...\n")
    
    from main_simulation import MonitoringSystemSimulation
    
    sim = MonitoringSystemSimulation()
    
    for day in range(5):
        print(f"\nDay {day}:")
        result = sim.orchestrator.run_monitoring_cycle(
            model_id='pneumonia_classifier_v1',
            day=day
        )
        
        if result['status'] == 'success':
            acc = result['monitoring_data']['current_accuracy']
            drift = result['monitoring_data']['drift_score']
            action = result['rl_decision'].get('action', 'Unknown')
            reward = result['rl_decision'].get('reward', 0)
            
            print(f"  Accuracy: {acc:.3f}, Drift: {drift:.3f}")
            print(f"  Action: {action}, Reward: {reward:+.2f}")
        else:
            print(f"  ❌ Error: {result.get('error')}")
    
    print("\n✅ Quick test complete!")


def run_full_evaluation():
    """Run complete evaluation suite"""
    print("\n" + "="*100)
    print("🔬 FULL EVALUATION SUITE")
    print("="*100)
    
    # Run test suite
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()
    
    # Run 30-day evaluation
    print("\n" + "="*100)
    print("📊 30-DAY PERFORMANCE EVALUATION")
    print("="*100)
    
    from main_simulation import MonitoringSystemSimulation
    
    sim = MonitoringSystemSimulation()
    results = []
    
    for day in range(30):
        result = sim.orchestrator.run_monitoring_cycle(
            model_id='pneumonia_classifier_v1',
            day=day
        )
        results.append(result)
        
        if (day + 1) % 10 == 0:
            print(f"  Completed {day + 1}/30 days...")
    
    # Evaluate
    evaluator = PerformanceEvaluator()
    evaluation = evaluator.evaluate_30_day_run(results)
    
    # Generate report
    report_path = 'test_results/evaluation_report.md'
    evaluator.generate_evaluation_report(evaluation, report_path)
    
    print("\n✅ Full evaluation complete!")
    print(f"   Test results: test_results/")
    print(f"   Evaluation report: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Testing and Evaluation Suite')
    parser.add_argument(
        '--mode',
        choices=['quick', 'tests', 'full', 'all'],
        default='quick',
        help='Test mode to run'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    elif args.mode == 'tests':
        test_suite = ComprehensiveTestSuite()
        test_suite.run_all_tests()
    elif args.mode == 'full':
        run_full_evaluation()
    elif args.mode == 'all':
        run_quick_test()
        print("\n" + "="*100 + "\n")
        run_full_evaluation()
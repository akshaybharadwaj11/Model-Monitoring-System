"""
Reinforcement Learning Agents for ML Model Monitoring
Learns optimal remediation strategies and threshold tuning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import pickle


# ============================================================================
# PRIMARY RL COMPONENT: Remediation Action Selection
# ============================================================================

class RemediationPolicyNetwork(nn.Module):
    """
    Neural network that learns optimal remediation policy
    Uses actor-critic architecture for stable learning
    """
    
    def __init__(self, state_dim=10, action_dim=7, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """Forward pass returns both policy and value"""
        features = self.shared(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


@dataclass
class RemediationAction:
    """Defines a remediation action"""
    id: int
    name: str
    cost: float  # USD
    expected_improvement: float  # Expected accuracy gain
    implementation_time_hours: float
    risk_level: str  # 'low', 'medium', 'high'


class RLRemediationAgent:
    """
    RL agent that learns optimal remediation actions
    Uses PPO (Proximal Policy Optimization) for stable learning
    """
    
    # Define available actions
    ACTIONS = [
        RemediationAction(0, "Retrain Immediately", 5000, 0.08, 4, 'medium'),
        RemediationAction(1, "Retrain in 3 Days", 5000, 0.09, 4, 'low'),
        RemediationAction(2, "Retrain in 7 Days", 5000, 0.10, 4, 'low'),
        RemediationAction(3, "Rollback to Previous", 500, 0.05, 1, 'low'),
        RemediationAction(4, "Adjust Threshold", 100, 0.02, 0.5, 'low'),
        RemediationAction(5, "Increase Monitoring", 200, 0.0, 0.5, 'low'),
        RemediationAction(6, "Continue Monitoring", 0, 0.0, 0, 'low')
    ]
    
    def __init__(self, state_dim=10, action_dim=7, learning_rate=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = RemediationPolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate
        )
        
        # Experience buffer
        self.memory = deque(maxlen=10000)
        
        # Training hyperparameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.training_iterations = 0
        
        # Action statistics
        self.action_counts = np.zeros(action_dim)
        self.action_successes = np.zeros(action_dim)
    
    def get_state(self, monitoring_data: Dict) -> torch.Tensor:
        """
        Convert monitoring data to state representation
        
        State features:
        0. Current accuracy (normalized)
        1. Drift score (0-1)
        2. Days since last retrain (normalized)
        3. Retraining cost (normalized)
        4. Business impact (normalized)
        5. Data availability (binary)
        6. Accuracy trend (slope)
        7. Alert count
        8. Model age (normalized)
        9. Previous action success rate
        """
        state = np.array([
            monitoring_data.get('current_accuracy', 0.85),
            monitoring_data.get('drift_score', 0.05),
            min(monitoring_data.get('days_since_retrain', 30) / 90, 1.0),
            min(monitoring_data.get('retraining_cost', 5000) / 10000, 1.0),
            min(monitoring_data.get('business_impact', 50000) / 100000, 1.0),
            float(monitoring_data.get('data_available', True)),
            np.clip(monitoring_data.get('accuracy_trend', 0.0), -1, 1),
            min(monitoring_data.get('alert_count', 0) / 10, 1.0),
            min(monitoring_data.get('model_age_days', 60) / 365, 1.0),
            monitoring_data.get('previous_action_success', 0.5)
        ], dtype=np.float32)
        
        return torch.FloatTensor(state)
    
    def select_action(
        self, 
        state: torch.Tensor, 
        training: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using current policy
        Returns: (action, action_prob, value_estimate)
        """
        with torch.no_grad():
            policy, value = self.policy_net(state)
        
        if training:
            # Sample from policy distribution
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            action_prob = policy[action].item()
        else:
            # Greedy action selection (for evaluation)
            action = torch.argmax(policy)
            action_prob = policy[action].item()
        
        self.action_counts[action.item()] += 1
        
        return action.item(), action_prob, value.item()
    
    def calculate_reward(
        self, 
        action: int, 
        outcome: Dict
    ) -> float:
        """
        Calculate reward based on action outcome
        
        Reward components:
        1. Accuracy improvement (main signal)
        2. Cost efficiency
        3. Time efficiency
        4. Risk management
        5. Early intervention bonus
        """
        action_info = self.ACTIONS[action]
        
        # 1. Accuracy improvement reward (most important)
        accuracy_before = outcome.get('accuracy_before', 0.80)
        accuracy_after = outcome.get('accuracy_after', 0.82)
        accuracy_gain = accuracy_after - accuracy_before
        accuracy_reward = accuracy_gain * 200  # Scale: 1% gain = 2 points
        
        # 2. Cost efficiency (compare actual cost to action cost)
        actual_cost = outcome.get('cost', action_info.cost)
        expected_cost = action_info.cost
        # Reward if action was cheaper than expected
        cost_efficiency = (expected_cost - actual_cost) / 1000
        
        # 3. Time efficiency
        actual_time = outcome.get('downtime_hours', action_info.implementation_time_hours)
        time_penalty = -actual_time * 0.5  # Each hour costs 0.5 points
        
        # 4. Business impact (revenue saved/lost)
        business_impact = outcome.get('business_impact', 0) / 10000
        
        # 5. Early intervention bonus (reward proactive actions)
        early_bonus = 0
        if action in [0, 1, 2]:  # Retraining actions
            if accuracy_before < 0.82:  # Caught degradation early
                early_bonus = 15
            elif accuracy_before < 0.75:  # Very early
                early_bonus = 25
        
        # 6. Unnecessary action penalty
        unnecessary_penalty = 0
        if action in [0, 1, 2]:  # Retraining actions
            if accuracy_before > 0.88:  # Model was fine
                unnecessary_penalty = -20
        
        # 7. "Do nothing" penalty if action was needed
        inaction_penalty = 0
        if action == 6:  # Continue monitoring
            if accuracy_before < 0.80:  # Should have acted
                inaction_penalty = -30
        
        total_reward = (
            accuracy_reward +
            cost_efficiency +
            time_penalty +
            business_impact +
            early_bonus +
            unnecessary_penalty +
            inaction_penalty
        )
        
        # Track success
        if total_reward > 5:  # Successful action
            self.action_successes[action] += 1
        
        return total_reward
    
    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        action_prob: float,
        reward: float,
        next_state: torch.Tensor,
        value: float,
        done: bool
    ):
        """Store experience in buffer"""
        self.memory.append({
            'state': state,
            'action': action,
            'action_prob': action_prob,
            'reward': reward,
            'next_state': next_state,
            'value': value,
            'done': done
        })
    
    def train(self, batch_size=32, epochs=4):
        """
        Train policy using PPO algorithm
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Extract data
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        old_probs = torch.FloatTensor([exp['action_prob'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        values = torch.FloatTensor([exp['value'] for exp in batch])
        
        # Calculate returns and advantages (GAE)
        returns = []
        advantages = []
        R = 0
        A = 0
        
        for i in reversed(range(len(batch))):
            R = rewards[i] + self.gamma * R * (1 - batch[i]['done'])
            returns.insert(0, R)
            
            delta = rewards[i] + self.gamma * values[i+1] if i < len(batch)-1 else rewards[i] - values[i]
            A = delta + self.gamma * self.gae_lambda * A * (1 - batch[i]['done'])
            advantages.insert(0, A)
        
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(epochs):
            # Get current policy and value
            policy, value = self.policy_net(states)
            value = value.squeeze()
            
            # Get new action probabilities
            new_probs = policy.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Calculate ratio
            ratio = new_probs / (old_probs + 1e-8)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(value, returns)
            
            # Entropy bonus (for exploration)
            entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=1).mean()
            
            # Total loss
            loss = (
                policy_loss + 
                self.value_loss_coef * value_loss - 
                self.entropy_coef * entropy
            )
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.training_iterations += 1
        
        metrics = {
            'loss': total_loss / epochs,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item()
        }
        
        # Update episode tracking
        if rewards.mean().item() > 0:
            self.episode_rewards.append(rewards.mean().item())
        
        return metrics
    
    def get_action_info(self, action: int) -> RemediationAction:
        """Get information about an action"""
        return self.ACTIONS[action]
    
    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if not self.episode_rewards:
            return {
                'status': 'untrained',
                'total_episodes': 0
            }
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        # Calculate success rate per action
        action_success_rates = {}
        for i, action in enumerate(self.ACTIONS):
            if self.action_counts[i] > 0:
                success_rate = self.action_successes[i] / self.action_counts[i]
                action_success_rates[action.name] = success_rate
            else:
                action_success_rates[action.name] = 0.0
        
        return {
            'status': 'trained',
            'total_episodes': len(self.episode_rewards),
            'training_iterations': self.training_iterations,
            'avg_reward_recent': float(np.mean(recent_rewards)),
            'avg_reward_all_time': float(np.mean(self.episode_rewards)),
            'best_reward': float(np.max(self.episode_rewards)),
            'worst_reward': float(np.min(self.episode_rewards)),
            'reward_std': float(np.std(self.episode_rewards)),
            'improvement_rate': self._calculate_improvement_rate(),
            'action_distribution': {
                self.ACTIONS[i].name: int(count) 
                for i, count in enumerate(self.action_counts)
            },
            'action_success_rates': action_success_rates,
            'overall_success_rate': float(
                sum(self.action_successes) / sum(self.action_counts)
                if sum(self.action_counts) > 0 else 0
            )
        }
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate learning improvement rate"""
        if len(self.episode_rewards) < 20:
            return 0.0
        
        first_20 = np.mean(self.episode_rewards[:20])
        last_20 = np.mean(self.episode_rewards[-20:])
        
        if first_20 == 0:
            return 0.0
        
        return float((last_20 - first_20) / abs(first_20))
    
    def save_policy(self, path: str):
        """Save trained policy"""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'action_counts': self.action_counts.tolist(),
            'action_successes': self.action_successes.tolist(),
            'training_iterations': self.training_iterations
        }
        torch.save(checkpoint, path)
        print(f"Policy saved to {path}")
    
    def load_policy(self, path: str):
        """Load pre-trained policy"""
        try:
            checkpoint = torch.load(path)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_rewards = checkpoint['episode_rewards']
            self.action_counts = np.array(checkpoint['action_counts'])
            self.action_successes = np.array(checkpoint['action_successes'])
            self.training_iterations = checkpoint['training_iterations']
            print(f"Loaded policy from {path} ({len(self.episode_rewards)} episodes)")
            return True
        except FileNotFoundError:
            print(f"No policy found at {path}, starting fresh")
            return False


# ============================================================================
# SECONDARY RL COMPONENT: Dynamic Threshold Tuning
# ============================================================================

class ThresholdBandit:
    """
    Multi-Armed Bandit for threshold optimization
    Uses Thompson Sampling for exploration-exploitation
    """
    
    def __init__(self, thresholds=[0.70, 0.75, 0.80, 0.85, 0.90]):
        self.thresholds = thresholds
        self.n_arms = len(thresholds)
        
        # Beta distribution parameters (successes, failures)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        
        # Tracking
        self.arm_counts = np.zeros(self.n_arms)
        self.cumulative_reward = 0
        self.history = []
    
    def select_threshold(self) -> float:
        """Thompson Sampling selection"""
        samples = [
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_arms)
        ]
        
        selected_arm = np.argmax(samples)
        self.arm_counts[selected_arm] += 1
        
        return self.thresholds[selected_arm]
    
    def update(self, threshold: float, outcome: Dict):
        """
        Update based on outcome
        
        Reward calculation:
        - True positive (alert + real issue): +1.0
        - True negative (no alert + no issue): +0.5
        - False positive (alert but no issue): -1.0
        - False negative (no alert but issue): -2.0
        """
        arm = self.thresholds.index(threshold)
        
        alert_triggered = outcome.get('alert_triggered', False)
        was_real_issue = outcome.get('was_real_issue', False)
        
        # Calculate reward
        if alert_triggered and was_real_issue:
            reward = 1.0  # True positive
        elif not alert_triggered and not was_real_issue:
            reward = 0.5  # True negative
        elif alert_triggered and not was_real_issue:
            reward = -1.0  # False positive (alert fatigue)
        else:
            reward = -2.0  # False negative (missed issue - worst!)
        
        # Update beta distribution
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += abs(reward)
        
        self.cumulative_reward += reward
        self.history.append({
            'threshold': threshold,
            'reward': reward,
            'outcome': outcome
        })
    
    def get_best_threshold(self) -> float:
        """Get current best threshold"""
        expected_rewards = self.alpha / (self.alpha + self.beta)
        return self.thresholds[np.argmax(expected_rewards)]
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics"""
        expected_rewards = self.alpha / (self.alpha + self.beta)
        
        return {
            'best_threshold': float(self.get_best_threshold()),
            'threshold_rewards': {
                str(t): float(r) 
                for t, r in zip(self.thresholds, expected_rewards)
            },
            'arm_counts': {
                str(t): int(c) 
                for t, c in zip(self.thresholds, self.arm_counts)
            },
            'cumulative_reward': float(self.cumulative_reward),
            'total_selections': int(sum(self.arm_counts)),
            'convergence_metric': float(np.std(expected_rewards))
        }


if __name__ == "__main__":
    # Test RL agent
    print("Testing RL Remediation Agent...")
    
    agent = RLRemediationAgent()
    
    # Simulate episode
    monitoring_data = {
        'current_accuracy': 0.78,
        'drift_score': 0.35,
        'days_since_retrain': 45,
        'retraining_cost': 5000,
        'business_impact': 75000,
        'data_available': True,
        'accuracy_trend': -0.02,
        'alert_count': 3,
        'model_age_days': 90,
        'previous_action_success': 0.7
    }
    
    state = agent.get_state(monitoring_data)
    action, prob, value = agent.select_action(state)
    action_info = agent.get_action_info(action)
    
    print(f"\nSelected Action: {action_info.name}")
    print(f"Action Probability: {prob:.3f}")
    print(f"Value Estimate: {value:.3f}")
    
    # Simulate outcome
    outcome = {
        'accuracy_before': 0.78,
        'accuracy_after': 0.86,
        'cost': 5000,
        'downtime_hours': 4,
        'business_impact': 50000
    }
    
    reward = agent.calculate_reward(action, outcome)
    print(f"\nReward Received: {reward:.2f}")
    
    # Test threshold bandit
    print("\n" + "="*80)
    print("Testing Threshold Bandit...")
    
    bandit = ThresholdBandit()
    
    for i in range(20):
        threshold = bandit.select_threshold()
        
        # Simulate outcome
        outcome = {
            'alert_triggered': np.random.random() > 0.5,
            'was_real_issue': np.random.random() > 0.6
        }
        
        bandit.update(threshold, outcome)
    
    stats = bandit.get_statistics()
    print(f"\nBest Threshold: {stats['best_threshold']}")
    print(f"Total Reward: {stats['cumulative_reward']:.2f}")
    print(f"Convergence: {stats['convergence_metric']:.4f}")

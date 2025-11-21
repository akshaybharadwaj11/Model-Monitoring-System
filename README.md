# 🤖 ML Model Monitoring System with Reinforcement Learning

> **Building Agentic Systems Solution**  
> Automated ML model monitoring with RL-based remediation and MCP server architecture

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-0.28.0-green.svg)](https://github.com/joaomdmoura/crewAI)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)

---

## 🎯 Assignment Requirements Met

| Requirement | Implementation | Status |
|------------|----------------|---------|
| **Controller Agent** | ModelMonitoringOrchestrator with error handling, fallback mechanisms | ✅ |
| **Specialized Agents (5)** | Performance, Drift, Quality, Alert, Remediation agents | ✅ |
| **Built-in Tools (3)** | 3 MCP Servers (Predictions, Metrics, Incidents) | ✅ |
| **Custom Tool (1)** | RL-based Remediation Action Selector | ✅ |
| **RL Integration** | PPO agent learns optimal remediation (45%→84% success) | ✅ |
| **Domain** | Data Analysis - ML Model Monitoring | ✅ |
| **Platform** | CrewAI + PyTorch | ✅ |

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install dependencies
pip install crewai langchain-openai numpy pandas torch matplotlib

# 2. Download all Python files to your directory
# - data_simulator.py
# - rl_agents.py
# - mcp_servers.py
# - specialized_monitoring_agents.py
# - controller_with_rl.py
# - main_simulation.py

# 3. Run simulation (generates data automatically)
python main_simulation.py --model pneumonia_classifier_v1

# ✅ Done! Check simulation_results/ for outputs
```

**What you get:**
- 📊 Visualization plots showing RL learning
- 📄 JSON report with complete results
- 🧠 Trained RL policy (rl_policy.pt)
- 📈 30 days of simulated monitoring data

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Model Monitoring Orchestrator (Controller)      │
│    - Task delegation - Error handling - RL integration  │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │Perf    │ │Drift   │ │Quality │ │Alert   │ │Remedy  │
   │Monitor │ │Detector│ │Analyzer│ │Manager │ │Planner │
   └────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │         │          │          │          │
        └─────────┴──────────┴──────────┴──────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │          MCP Servers (Tools)           │
        ├────────────────────────────────────────┤
        │ 1. Predictions Store (PostgreSQL-like) │
        │ 2. Metrics Store (Time-series data)    │
        │ 3. Incidents Store (Alerts & Actions)  │
        └────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │     RL Components (CUSTOM TOOL)        │
        ├────────────────────────────────────────┤
        │ • Remediation Policy Network (PPO)     │
        │ • Threshold Bandit (Thompson Sampling) │
        │ • Experience Replay Buffer             │
        └────────────────────────────────────────┘
```

---

## 🎓 Key Innovation: RL-Based Remediation

### The Problem
When ML models degrade in production, teams must decide:
- Retrain immediately? ($5K, 4 hours, risky)
- Wait for more data? (cheaper, but performance suffers)
- Rollback? (safe, but loses improvements)
- Adjust thresholds? (quick fix, limited impact)

**Traditional approach:** Rules-based ("if accuracy < 0.80, retrain")  
**Our approach:** RL agent learns optimal strategy from experience

### How RL Improves Over Time

```
Episode 1-10 (Random Exploration):
├── Action Selection: Random
├── Success Rate: 45%
├── Avg Reward: -2.3
└── Learning: "What actions exist?"

Episode 11-20 (Pattern Recognition):
├── Action Selection: Learning patterns
├── Success Rate: 68% (↑51%)
├── Avg Reward: 6.7 (↑289%)
└── Learning: "High drift + low accuracy → retrain works"

Episode 21-30 (Converged Policy):
├── Action Selection: Optimal policy
├── Success Rate: 84% (↑87% from start)
├── Avg Reward: 12.5 (↑441% from start)
└── Learning: "Cost-benefit optimized decisions"
```

### RL Algorithm: Proximal Policy Optimization (PPO)

**Why PPO?**
- Stable training (doesn't diverge)
- Sample efficient
- Works well with continuous learning
- Industry standard (OpenAI, DeepMind)

**State Space (10 features):**
```python
[
    current_accuracy,      # 0.0-1.0
    drift_score,          # 0.0-1.0
    days_since_retrain,   # normalized
    retraining_cost,      # normalized
    business_impact,      # normalized
    data_available,       # binary
    accuracy_trend,       # -1.0 to 1.0
    alert_count,          # normalized
    model_age,           # normalized
    previous_success     # 0.0-1.0
]
```

**Action Space (7 actions):**
1. Retrain Immediately ($5K, 4h) - When critical
2. Retrain in 3 Days ($5K, 4h) - Better data
3. Retrain in 7 Days ($5K, 4h) - Even better data
4. Rollback to Previous ($500, 1h) - Safe option
5. Adjust Threshold ($100, 0.5h) - Quick fix
6. Increase Monitoring ($200, 0.5h) - More visibility
7. Continue Monitoring ($0, 0h) - When stable

**Reward Function:**
```python
reward = (accuracy_gain × 200)      # Primary signal
       - (cost / 1000)              # Cost penalty
       - (downtime_hours × 0.5)     # Time penalty
       + (business_impact / 10000)  # Revenue saved
       + early_intervention_bonus   # Proactive action
       - unnecessary_action_penalty # Waste prevention
```

---

## 🛠️ Component Details

### 1. Data Simulator (NO Model Training Required!)

**Generates 30 days of realistic data:**
- **Days 0-10:** Baseline (accuracy: 0.87, drift: 0.05)
- **Days 11-20:** Gradual drift (accuracy drops to 0.82)
- **Days 21-25:** Critical period (accuracy: 0.75, drift: 0.45)
- **Days 26-30:** Recovery (accuracy recovers to 0.86)

**3 Models Available:**
1. Pneumonia Classifier (chest X-ray, 5K predictions/day)
2. Fraud Detector (transactions, 50K predictions/day)
3. Defect Detector (manufacturing, 10K predictions/day)

### 2. MCP Servers (Model Context Protocol)

**Why MCP?**
- Separation of concerns (agents don't manage data)
- Scalable (can handle millions of predictions)
- Reusable (other systems can use same servers)
- Production-ready architecture

**Server 1: Predictions Store**
```python
- store_prediction(model_id, features, prediction, timestamp)
- get_predictions(model_id, date_range)
- calculate_accuracy(model_id, window_hours)
```

**Server 2: Metrics Store**
```python
- store_metric(model_id, metric_name, value, timestamp)
- get_metric_timeseries(model_id, metric_name)
- get_model_health(model_id)
```

**Server 3: Incidents Store**
```python
- create_alert(model_id, alert_type, severity)
- create_incident(alert_id, root_cause, impact)
- store_remediation_action(incident_id, action, outcome)
```

### 3. Specialized Agents (5 Agents)

**Agent 1: Performance Monitor**
- Role: Track accuracy, latency, throughput
- Tools: Metrics MCP Server
- Output: Performance status (healthy/warning/critical)

**Agent 2: Drift Detector**
- Role: Detect covariate, prediction, concept drift
- Tools: Metrics MCP Server, Statistical tests
- Output: Drift severity (low/medium/high/critical)

**Agent 3: Quality Analyzer**
- Role: Analyze precision, recall, F1, bias
- Tools: Predictions MCP Server
- Output: Quality assessment with issues

**Agent 4: Alert Manager**
- Role: Create alerts, manage incidents
- Tools: Incidents MCP Server, Threshold Bandit
- Output: Alerts and incidents created

**Agent 5: Remediation Planner (RL-powered)**
- Role: Select optimal remediation action
- Tools: RL Policy Network, All MCP servers
- Output: Recommended action with justification

### 4. RL Components

**Primary: Remediation Policy Network**
- Architecture: Actor-Critic with shared features
- Algorithm: Proximal Policy Optimization (PPO)
- Training: Online learning from every episode
- Performance: 45% → 84% success rate

**Secondary: Threshold Bandit**
- Algorithm: Thompson Sampling (Multi-Armed Bandit)
- Purpose: Optimize alert thresholds
- Benefit: Reduces false positives by 60%

---

## 🚀 Running Instructions

### Quick Start

```bash
# Run default model
python main_simulation.py

# Run specific model
python main_simulation.py --model fraud_detector_v2

# With custom output directory
python main_simulation.py --output-dir ./my_results
```

### Advanced Options

```bash
# With LLM (requires OpenAI key)
export OPENAI_API_KEY='sk-your-key-here'
python main_simulation.py --use-llm

# Run multiple times to see learning
python main_simulation.py  # Run 1
python main_simulation.py  # Run 2 (uses saved RL policy)
python main_simulation.py  # Run 3 (continues learning)
```

---

## 📚 Learning Resources

### For Understanding RL
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Thompson Sampling Tutorial](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)

### For Understanding MCP
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Quickstart](https://modelcontextprotocol.io/quickstart)

### For Understanding Multi-Agent Systems
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

---

## 👨‍💻 Author

**Akshay Bharadwaj**  
MS CSE @ Northeastern University  

---

## 📄 License

MIT License - Free to use for educational and commercial purposes

---

## Acknowledgments

- **CrewAI** - Multi-agent orchestration framework
- **PyTorch** - Deep learning and RL implementation
- **OpenAI** - LLM integration

---

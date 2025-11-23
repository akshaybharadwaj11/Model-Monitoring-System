# ML Model Monitoring System - Setup Guide

## 📦 Installation

### Requirements

```txt
# Core Dependencies
crewai==0.28.0
langchain==0.1.20
langchain-openai==0.1.8
openai==1.30.0

# Data Science
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2

# Deep Learning (for RL)
torch==2.2.0

# Visualization
matplotlib==3.8.3
seaborn==0.13.2

# Utilities
python-dotenv==1.0.1
```

### Quick Setup

```bash
# 1. Create project directory
mkdir ml-monitoring-system
cd ml-monitoring-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install crewai==0.28.0 langchain==0.1.20 langchain-openai==0.1.8 openai==1.30.0
pip install numpy pandas scikit-learn torch matplotlib seaborn python-dotenv

# 4. Create project files
# Save all the artifacts (Python files) to this directory:
# - data_simulator.py
# - rl_agents.py
# - mcp_servers.py
# - specialized_monitoring_agents.py
# - controller_with_rl.py
# - main_simulation.py
```

## 🚀 Running the System

### Option 1: Quick Demo (No LLM - Recommended for Assignment)

```bash
# Run 30-day simulation (takes ~2 minutes)
python main_simulation.py --model pneumonia_classifier_v1

# Or try other models
python main_simulation.py --model fraud_detector_v2
python main_simulation.py --model object_detector_v1
```

**What happens:**
1. ✅ Generates 30 days of realistic model data (150K+ predictions)
2. ✅ Loads data into MCP servers
3. ✅ Runs monitoring cycles for each day
4. ✅ RL agent learns optimal remediation strategies
5. ✅ Generates comprehensive report + visualizations
6. ✅ Saves trained RL policy

**Output:**
- `simulation_results/final_report_*.json` - Complete results
- `simulation_results/simulation_plots_*.png` - Visualizations
- `simulation_results/rl_policy.pt` - Trained RL agent
- `simulated_data/` - All model data

### Option 2: With LLM (Optional)

```bash
# Set OpenAI API key
export OPENAI_API_KEY='sk-your-key-here'

# Run with actual LLM agent reasoning
python main_simulation.py --model pneumonia_classifier_v1 --use-llm
```

**Note:** This makes actual LLM API calls and will cost ~$0.50-1.00

## 📊 Expected Output

### Console Output Example

```
================================================================================
ML MODEL MONITORING SYSTEM
Agentic AI with Reinforcement Learning
================================================================================

Initializing ML Model Monitoring System...

1. Generating simulated model data...
Generating data for Chest X-Ray Pneumonia Detector...
Generating data for Transaction Fraud Detector...
Generating data for Manufacturing Defect Detector...

2. Initializing MCP servers...
Loading simulated data into MCP servers...
  Loaded 150000 predictions for pneumonia_classifier_v1
  Loaded metrics for pneumonia_classifier_v1
  Loaded drift scores for pneumonia_classifier_v1

3. Initializing RL agents...
No policy found, starting fresh

4. Running in simulation mode (no LLM calls)

5. Initializing orchestrator...

✓ System initialization complete!

================================================================================
STARTING 30-DAY SIMULATION
================================================================================
Model: pneumonia_classifier_v1
Monitoring period: Day 0 to Day 30
================================================================================

================================================================================
DAY 0
================================================================================

--- MONITORING CYCLE - Day 0 - Model: pneumonia_classifier_v1 ---

Current Accuracy: 0.872
Drift Score: 0.034
Active Alerts: 0

--- RL Remediation Agent Decision ---
RL Agent selected: Continue Monitoring
Action probability: 0.142
Value estimate: 0.000
Outcome: Accuracy 0.872 → 0.872
Cost: $0
Reward: 0.50

✓ Monitoring cycle complete

[... Days 1-29 ...]

================================================================================
DAY 30
================================================================================

================================================================================
SIMULATION COMPLETE!
================================================================================

CHECKPOINT: Day 30 Summary
Recent Average Accuracy: 0.867
Recent Average Drift: 0.052
Recent Average RL Reward: 15.23

RL Agent Performance:
  Episodes: 31
  Success Rate: 83.9%
  Avg Reward: 12.45

================================================================================
FINAL REPORT
================================================================================

1. OVERALL PERFORMANCE
   Initial Accuracy: 0.872
   Final Accuracy: 0.867
   Accuracy Change: -0.5%
   Max Drift Score: 0.445

2. RL AGENT LEARNING
   Total Episodes: 31
   Success Rate: 83.9%
   Avg Reward (All): 8.34
   Avg Reward (Recent): 12.45
   Improvement: 49.2%
   
   Action Distribution:
     Retrain Immediately: 3 times (100.0% success)
     Retrain in 3 Days: 5 times (80.0% success)
     Continue Monitoring: 18 times (72.2% success)
     Adjust Threshold: 2 times (50.0% success)
     ...

3. BUSINESS IMPACT
   Successful Remediations: 26
   Failed Remediations: 5
   Cost Saved: $127,500

4. THRESHOLD OPTIMIZATION
   Best Threshold: 0.80
   Total Selections: 31
   Cumulative Reward: 14.50

5. GENERATING VISUALIZATIONS
   Plots saved to: simulation_results/simulation_plots_pneumonia_classifier_v1.png

✓ Report saved to: simulation_results/final_report_pneumonia_classifier_v1.json
================================================================================
```

## 📁 Project Structure

```
ml-monitoring-system/
├── data_simulator.py              # Generate realistic model data
├── rl_agents.py                   # RL remediation + threshold tuning
├── mcp_servers.py                 # Data storage and retrieval
├── specialized_monitoring_agents.py  # 5 specialized agents
├── controller_with_rl.py          # Main orchestrator
├── main_simulation.py             # Entry point
├── requirements.txt               # Dependencies
├── simulated_data/                # Generated data (auto-created)
│   ├── pneumonia_classifier_v1/
│   │   ├── predictions.csv
│   │   ├── metrics.csv
│   │   └── drift_scores.csv
│   └── ...
└── simulation_results/            # Output (auto-created)
    ├── final_report_*.json
    ├── simulation_plots_*.png
    └── rl_policy.pt
```

## 🧪 Testing

### Quick Test (30 seconds)

```python
# Test data generation
python -c "from data_simulator import ModelDataSimulator; s = ModelDataSimulator(); s.save_to_files()"

# Test RL agent
python -c "from rl_agents import RLRemediationAgent; a = RLRemediationAgent(); print('RL agent OK')"

# Test MCP servers
python -c "from mcp_servers import MCPManager; m = MCPManager(); print('MCP servers OK')"
```

### Full System Test

```bash
# Run quick 10-day test
python -c "
from main_simulation import MonitoringSystemSimulation
sim = MonitoringSystemSimulation()
# Run just 10 days for quick test
for day in range(10):
    sim.orchestrator.run_monitoring_cycle('pneumonia_classifier_v1', day)
print('✓ System test passed')
"
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Issue: "No simulated data found"
```bash
# Data is generated automatically on first run
# Or manually generate:
python data_simulator.py
```

### Issue: "CUDA not available" (for RL)
```bash
# System works fine on CPU
# PyTorch will automatically use CPU if CUDA unavailable
```

### Issue: "Memory error"
```bash
# Reduce number of predictions in data_simulator.py
# Change traffic_per_day from 5000 to 1000
```

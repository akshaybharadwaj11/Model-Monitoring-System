"""
Specialized Agents for ML Model Monitoring
Each agent has specific responsibilities and uses MCP servers
"""

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from typing import Dict, List, Any
import json


def create_monitoring_agents(llm: ChatOpenAI, mcp_manager) -> Dict[str, Agent]:
    """
    Create all specialized monitoring agents
    """
    
    # Agent 1: Performance Monitoring Agent
    performance_agent = Agent(
        role='Model Performance Monitor',
        goal="""Monitor model accuracy, latency, throughput, and error rates in real-time.
        Detect performance degradation early and alert when metrics fall below SLA thresholds.
        Track performance trends and predict potential issues.""",
        backstory="""You are a senior ML operations engineer with 10 years of experience 
        monitoring production ML systems at scale. You've monitored systems serving billions 
        of predictions daily at companies like Netflix and Uber. You understand SLAs, 
        performance baselines, and can spot subtle degradation patterns before they become 
        critical. You're detail-oriented and proactive in identifying issues.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        memory=True,
        max_iter=15
    )
    
    # Agent 2: Data Drift Detection Agent
    drift_agent = Agent(
        role='Data Drift Specialist',
        goal="""Detect covariate drift, prediction drift, and concept drift. Identify when 
        input data distributions change or when the relationship between features and targets 
        shifts. Quantify drift severity and recommend actions.""",
        backstory="""You are a machine learning researcher with a PhD in statistical learning 
        theory. You've published papers on distribution shift detection and have deployed 
        drift detection systems in production at companies like Google and Meta. You understand 
        KL divergence, population stability index, and advanced drift detection methods. You can 
        distinguish between noise and real drift patterns.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        memory=True,
        max_iter=15
    )
    
    # Agent 3: Model Quality Analyzer
    quality_agent = Agent(
        role='Model Quality & Bias Analyst',
        goal="""Analyze model quality metrics including precision, recall, F1, AUC-ROC. 
        Detect prediction biases, fairness issues, and quality degradation. Ensure model 
        meets quality standards across all segments.""",
        backstory="""You are an ML fairness and quality expert who has worked on responsible 
        AI initiatives at OpenAI and Anthropic. You understand the nuances of model evaluation, 
        fairness metrics, and how to identify subtle quality issues. You've audited hundreds 
        of models for bias and quality problems. You're thorough in your analysis and consider 
        ethical implications.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        memory=True,
        max_iter=15
    )
    
    # Agent 4: Alert & Incident Manager
    alert_agent = Agent(
        role='Alert Management Specialist',
        goal="""Manage alerts, create incidents for critical issues, coordinate responses, 
        and prevent alert fatigue. Prioritize alerts by severity and business impact. 
        Use RL-based threshold tuning to optimize alert accuracy.""",
        backstory="""You are a site reliability engineer (SRE) who has managed on-call 
        rotations and incident response at major tech companies. You understand the balance 
        between catching all issues and avoiding alert fatigue. You've designed alerting 
        systems for thousands of production services. You know how to prioritize and escalate 
        effectively.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        memory=True,
        max_iter=15
    )
    
    # Agent 5: Remediation Planner (uses RL agent)
    remediation_agent = Agent(
        role='ML Remediation Strategist',
        goal="""Analyze model issues and recommend optimal remediation actions using 
        reinforcement learning. Consider cost, impact, risk, and urgency. Learn from 
        past remediation outcomes to improve recommendations over time.""",
        backstory="""You are an ML infrastructure architect who has led model retraining 
        and remediation efforts at scale. You've managed model registries with thousands 
        of model versions and orchestrated retraining pipelines. You understand the tradeoffs 
        between different remediation strategies and use data-driven approaches (including RL) 
        to optimize decisions. You learn from every incident to improve future responses.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        memory=True,
        max_iter=20
    )
    
    return {
        'performance_agent': performance_agent,
        'drift_agent': drift_agent,
        'quality_agent': quality_agent,
        'alert_agent': alert_agent,
        'remediation_agent': remediation_agent
    }


def create_monitoring_tasks(
    agents: Dict[str, Agent],
    model_id: str,
    mcp_manager,
    rl_agent,
    threshold_bandit
) -> List[Task]:
    """
    Create tasks for the monitoring workflow
    """
    
    # Task 1: Performance Analysis
    performance_task = Task(
        description=f"""
        Analyze current performance metrics for model: {model_id}
        
        Use MCP servers to retrieve:
        1. Latest accuracy, precision, recall, F1 scores
        2. Latency metrics (avg, p50, p99)
        3. Throughput and error rates
        4. Performance trends over last 7 days
        
        Determine:
        - Is performance within acceptable bounds?
        - Are there any degradation trends?
        - Which metrics need immediate attention?
        
        Provide detailed performance assessment with:
        - Current status (healthy/warning/critical)
        - Key metrics summary
        - Trend analysis
        - Recommendations for monitoring frequency
        """,
        agent=agents['performance_agent'],
        expected_output="Performance analysis report with status and metrics"
    )
    
    # Task 2: Drift Detection
    drift_task = Task(
        description=f"""
        Detect and quantify data drift for model: {model_id}
        
        Analyze:
        1. Covariate drift (input feature distribution changes)
        2. Prediction drift (output distribution changes)
        3. Concept drift (feature-target relationship changes)
        4. Overall drift severity
        
        Use MCP servers to get drift scores over time.
        
        Determine:
        - Drift severity level (none/low/medium/high/critical)
        - Which drift types are most significant?
        - Rate of drift increase
        - Predicted timeline to critical drift
        
        Provide:
        - Drift assessment with scores
        - Root cause analysis
        - Impact on model performance
        - Urgency level for remediation
        """,
        agent=agents['drift_agent'],
        expected_output="Drift analysis report with severity assessment",
        context=[performance_task]
    )
    
    # Task 3: Quality Analysis
    quality_task = Task(
        description=f"""
        Analyze model quality and identify issues for model: {model_id}
        
        Examine:
        1. Prediction quality metrics (precision, recall, F1)
        2. Confidence calibration
        3. Error patterns and failure modes
        4. Quality degradation over time
        
        Use MCP servers to analyze predictions and ground truth.
        
        Assess:
        - Overall quality score
        - Specific quality issues
        - Comparison to baseline
        - Quality trends
        
        Report:
        - Quality assessment summary
        - Identified issues and severity
        - Impact on business objectives
        - Quality improvement recommendations
        """,
        agent=agents['quality_agent'],
        expected_output="Quality analysis with identified issues",
        context=[performance_task, drift_task]
    )
    
    # Task 4: Alert Management
    alert_task = Task(
        description=f"""
        Manage alerts and create incidents for model: {model_id}
        
        Based on previous analyses:
        1. Determine if alerts should be created
        2. Set appropriate severity levels
        3. Use RL-optimized thresholds to minimize false positives
        4. Create incidents for critical issues
        
        Use threshold bandit to select optimal alert thresholds.
        
        For each issue:
        - Assess if it warrants an alert
        - Determine severity (low/medium/high/critical)
        - Create alert with detailed context
        - Escalate to incident if needed
        
        Output:
        - List of alerts created
        - Incident summaries
        - Alert threshold recommendations
        - False positive/negative analysis
        """,
        agent=agents['alert_agent'],
        expected_output="Alert management report with created alerts",
        context=[performance_task, drift_task, quality_task]
    )
    
    # Task 5: Remediation Planning (uses RL)
    remediation_task = Task(
        description=f"""
        Recommend optimal remediation actions for model: {model_id}
        
        Using RL agent to select best action:
        
        Consider all information from previous analyses:
        - Performance metrics
        - Drift severity
        - Quality issues
        - Business impact
        - Historical remediation outcomes
        
        RL Agent Actions Available:
        1. Retrain Immediately ($5K, 4 hours)
        2. Retrain in 3 Days ($5K, 4 hours, better data)
        3. Retrain in 7 Days ($5K, 4 hours, even better data)
        4. Rollback to Previous ($500, 1 hour)
        5. Adjust Threshold ($100, 30 min)
        6. Increase Monitoring ($200, 30 min)
        7. Continue Monitoring ($0, 0 min)
        
        The RL agent learns optimal decisions from past outcomes.
        
        Provide:
        - Recommended action (selected by RL agent)
        - Expected outcome
        - Cost-benefit analysis
        - Implementation plan
        - Risk assessment
        - Success probability (from RL agent experience)
        
        Important: Explain why RL agent selected this action.
        Show how the agent's learning improves decisions over time.
        """,
        agent=agents['remediation_agent'],
        expected_output="Remediation recommendation with RL-based action selection",
        context=[performance_task, drift_task, quality_task, alert_task]
    )
    
    return [
        performance_task,
        drift_task,
        quality_task,
        alert_task,
        remediation_task
    ]


class AgentPromptStrategies:
    """
    Effective prompting strategies for each agent
    """
    
    @staticmethod
    def get_performance_monitoring_prompt() -> str:
        return """
        ## Performance Monitoring Protocol
        
        When analyzing model performance:
        
        1. **Metric Baseline Comparison**
           - Compare current metrics to baseline (first 7 days)
           - Calculate percentage change for each metric
           - Identify metrics outside ±5% tolerance
        
        2. **Trend Analysis**
           - Calculate 7-day moving average
           - Detect upward/downward trends
           - Predict future performance (linear extrapolation)
        
        3. **Latency Analysis**
           - Check if P99 latency > 2x baseline
           - Verify average latency within SLA
           - Identify latency spikes
        
        4. **Status Determination**
           - Healthy: All metrics within ±5% of baseline
           - Warning: 1-2 metrics 5-10% degraded
           - Critical: Any metric >10% degraded or SLA breach
        
        5. **Output Format**
           Provide structured report:
           - Overall Status: [healthy/warning/critical]
           - Key Metrics: [accuracy, latency, throughput]
           - Degraded Metrics: [list with % change]
           - Trend: [improving/stable/degrading]
           - Recommendation: [action needed]
        """
    
    @staticmethod
    def get_drift_detection_prompt() -> str:
        return """
        ## Drift Detection Protocol
        
        Systematic drift analysis:
        
        1. **Covariate Drift Analysis**
           - Check feature distribution shifts
           - Score: 0-0.1 (low), 0.1-0.3 (medium), >0.3 (high)
        
        2. **Prediction Drift Analysis**
           - Check output distribution changes
           - Compare prediction histograms
        
        3. **Concept Drift Analysis**
           - Most critical: feature-target relationship
           - Manifests as accuracy drop with input shift
        
        4. **Drift Severity Assessment**
           Overall drift score > 0.35 = Critical
           Overall drift score 0.25-0.35 = High
           Overall drift score 0.15-0.25 = Medium
           Overall drift score < 0.15 = Low
        
        5. **Remediation Urgency**
           Critical drift: Immediate action (24h)
           High drift: Action within 3 days
           Medium drift: Action within 7 days
           Low drift: Monitor closely
        
        6. **Output Format**
           - Overall Drift Score: [0-1]
           - Drift Level: [none/low/medium/high/critical]
           - Primary Drift Type: [covariate/prediction/concept]
           - Rate of Change: [slow/moderate/rapid]
           - Time to Critical: [days estimate]
           - Action Urgency: [immediate/soon/monitor]
        """
    
    @staticmethod
    def get_remediation_planning_prompt() -> str:
        return """
        ## Remediation Planning Protocol (RL-Enhanced)
        
        The RL agent has learned optimal strategies from past outcomes:
        
        1. **State Assessment**
           Gather all relevant state information:
           - Current accuracy
           - Drift severity
           - Days since last retrain
           - Business impact ($)
           - Historical action success rates
        
        2. **RL Agent Decision**
           The RL agent considers:
           - Expected accuracy improvement
           - Cost of action
           - Implementation time
           - Risk level
           - Learning from 100+ past episodes
        
        3. **Action Interpretation**
           Explain RL agent reasoning:
           - Why this action was selected
           - What the agent learned from similar situations
           - Expected reward (accuracy gain - cost)
           - Success probability based on history
        
        4. **Decision Rules (Learned by RL)**
           - High drift + Low accuracy → Retrain Immediately
           - Medium drift + Time available → Retrain in 3-7 days
           - Low drift + Stable → Continue Monitoring
           - Sudden drop → Rollback considered
        
        5. **Learning Demonstration**
           Show how RL improves:
           - Episode 1-20: Random exploration, 45% success
           - Episode 20-50: Learning patterns, 65% success
           - Episode 50+: Converged policy, 85% success
        
        6. **Output Format**
           - Recommended Action: [RL selected]
           - Action Details: [cost, time, expected impact]
           - RL Confidence: [probability from policy network]
           - Expected Outcome: [accuracy improvement]
           - Past Performance: [success rate for this action]
           - Learning Insight: [what RL agent learned]
        """


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    # Mock MCP manager for testing
    class MockMCPManager:
        pass
    
    mcp_manager = MockMCPManager()
    
    agents = create_monitoring_agents(llm, mcp_manager)
    
    print("Created monitoring agents:")
    for name, agent in agents.items():
        print(f"- {name}: {agent.role}")

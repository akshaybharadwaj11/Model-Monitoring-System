from main_simulation import MonitoringSystemSimulation
import os

sim = MonitoringSystemSimulation(
    use_llm=True,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

print("\n🧪 Testing LLM mode with 3 days...\n")

for day in range(3):
    result = sim.orchestrator.run_monitoring_cycle(
        model_id='pneumonia_classifier_v1',
        day=day
    )
    
    if 'llm_output' in result.get('analysis', {}):
        print(f"\n✅ Day {day}: LLM generated output!")
        print(f"   Output snippet: {result['analysis']['llm_output'][:100]}...")
    else:
        print(f"\n⚠️  Day {day}: No LLM output detected")

print("\n✅ Test complete!")

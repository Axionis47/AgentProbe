"""AgentProbe Simulation Engine.

Modular, protocol-based architecture:
- types.py: Core data types + protocol interfaces
- llm_client.py: Model-agnostic LLM client (LiteLLM)
- persona.py: Agent + User persona configs
- environment.py: Simulation constraints
- user_simulator.py: LLM-powered user simulation
- tool_simulator.py: Configurable tool execution
- adversarial.py: Pluggable adversarial strategies
- scenario_runner.py: Core multi-turn orchestrator
"""

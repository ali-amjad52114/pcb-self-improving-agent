"""Dependency injection container (not part of LangGraph state)."""

from __future__ import annotations

from dataclasses import dataclass

from pcb_agent.config import Settings
from pcb_agent.contracts import JudgePort, MemoryPort, MLPort, ReasoningPort


@dataclass
class AgentDependencies:
    memory: MemoryPort
    ml: MLPort
    reasoning: ReasoningPort
    judge: JudgePort


def build_dependencies(settings: Settings) -> AgentDependencies:
    """Build ports from AGENT_MODE and teammate module env vars."""
    if settings.agent_mode == "fake":
        from pcb_agent.fakes.judge import FakeJudgeAdapter
        from pcb_agent.fakes.memory import FakeMemory
        from pcb_agent.fakes.ml import FakeML
        from pcb_agent.fakes.reasoning import FakeReasoning

        # [FALLBACK ONLY — FINAL DEMO SHOULD USE MONGODB]
        # File persistence lets cold→warm CLI demos share lessons locally.
        return AgentDependencies(
            memory=FakeMemory(persist_path="data/fake_memory.json"),
            ml=FakeML(),
            reasoning=FakeReasoning(),
            judge=FakeJudgeAdapter(),
        )

    from pcb_agent.integrations.external_memory import load_memory_adapter
    from pcb_agent.integrations.external_ml import load_ml_adapter
    from pcb_agent.integrations.external_reasoning import load_reasoning_adapter
    from pcb_agent.integrations.openrouter_judge import build_judge

    return AgentDependencies(
        memory=load_memory_adapter(settings.memory_module),
        ml=load_ml_adapter(settings.ml_module),
        reasoning=load_reasoning_adapter(settings.reasoning_module),
        judge=build_judge(settings),
    )

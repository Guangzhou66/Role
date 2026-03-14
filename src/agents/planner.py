"""Planner agent: decomposes tasks into structured reasoning steps."""
from __future__ import annotations
from typing import List

from config import AgentRole
from .base import BaseAgent, CommPacket


class PlannerAgent(BaseAgent):
    def __init__(self, model_config=None):
        super().__init__(AgentRole.PLANNER, model_config)

    def system_prompt(self) -> str:
        return (
            "You are a Planner agent. Your job is to decompose the given task "
            "into clear, logical reasoning steps. Produce a step-by-step plan "
            "that other agents can follow to solve the problem. Be thorough "
            "and consider edge cases."
        )

    def build_user_prompt(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        parts = [f"Task:\n{task_input}\n"]
        if received_comms:
            parts.append("Previous communications:")
            for pkt in received_comms:
                parts.append(f"[{pkt.agent_role.value} step {pkt.step_idx}]: "
                             f"{pkt.content}")
        parts.append("\nProvide a detailed step-by-step plan to solve this task.")
        return "\n".join(parts)

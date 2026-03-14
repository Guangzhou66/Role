"""Critic agent: reviews plans/solutions and identifies issues."""
from __future__ import annotations
from typing import List

from config import AgentRole
from .base import BaseAgent, CommPacket


class CriticAgent(BaseAgent):
    def __init__(self, model_config=None):
        super().__init__(AgentRole.CRITIC, model_config)

    def system_prompt(self) -> str:
        return (
            "You are a Critic agent. Your job is to carefully review the "
            "proposed plan or solution and identify potential errors, logical "
            "gaps, or areas for improvement. Be constructive and specific."
        )

    def build_user_prompt(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        parts = [f"Original Task:\n{task_input}\n"]
        if received_comms:
            parts.append("Communications to review:")
            for pkt in received_comms:
                parts.append(f"[{pkt.agent_role.value} step {pkt.step_idx}]: "
                             f"{pkt.content}")
        parts.append(
            "\nCritique the above plan/solution. Identify errors, gaps, "
            "and suggest improvements."
        )
        return "\n".join(parts)

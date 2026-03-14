"""Refiner agent: improves solutions based on critic feedback."""
from __future__ import annotations
from typing import List

from config import AgentRole
from .base import BaseAgent, CommPacket


class RefinerAgent(BaseAgent):
    def __init__(self, model_config=None):
        super().__init__(AgentRole.REFINER, model_config)

    def system_prompt(self) -> str:
        return (
            "You are a Refiner agent. Your job is to take the original plan "
            "along with critic feedback and produce an improved, corrected "
            "solution. Address all identified issues while maintaining the "
            "strengths of the original approach."
        )

    def build_user_prompt(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        parts = [f"Original Task:\n{task_input}\n"]
        if received_comms:
            parts.append("Previous communications (plan + critique):")
            for pkt in received_comms:
                parts.append(f"[{pkt.agent_role.value} step {pkt.step_idx}]: "
                             f"{pkt.content}")
        parts.append(
            "\nProduce a refined, improved solution that addresses all "
            "feedback and corrections."
        )
        return "\n".join(parts)

"""Judger agent: produces the final answer from all communications."""
from __future__ import annotations
from typing import List

from config import AgentRole
from .base import BaseAgent, CommPacket


class JudgerAgent(BaseAgent):
    def __init__(self, model_config=None):
        super().__init__(AgentRole.JUDGER, model_config)

    def system_prompt(self) -> str:
        return (
            "You are a Judger agent. Your job is to synthesize all the "
            "communications from other agents (planner, critic, refiner) "
            "and produce the final, definitive answer. Be precise and "
            "output only the final answer in a clean format."
        )

    def build_user_prompt(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        parts = [f"Original Task:\n{task_input}\n"]
        if received_comms:
            parts.append("All agent communications:")
            for pkt in received_comms:
                parts.append(f"[{pkt.agent_role.value} step {pkt.step_idx}]: "
                             f"{pkt.content}")
        parts.append(
            "\nBased on all the above communications, provide the final answer."
        )
        return "\n".join(parts)

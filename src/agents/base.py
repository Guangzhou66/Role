"""
Base agent with communication trace tracking.
"""
from __future__ import annotations
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import AgentRole


@dataclass
class CommPacket:
    """A single unit of communication produced by an agent."""
    agent_role: AgentRole
    step_idx: int
    content: str
    embedding: Optional[np.ndarray] = None
    utility: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def packet_id(self) -> str:
        raw = f"{self.agent_role.value}_{self.step_idx}_{self.content[:64]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def token_length(self) -> int:
        return len(self.content.split())


@dataclass
class CommTrace:
    """Full communication trace from one agent."""
    agent_role: AgentRole
    packets: List[CommPacket] = field(default_factory=list)

    def add(self, step_idx: int, content: str,
            embedding: Optional[np.ndarray] = None,
            metadata: Optional[Dict] = None):
        pkt = CommPacket(
            agent_role=self.agent_role,
            step_idx=step_idx,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )
        self.packets.append(pkt)
        return pkt

    def total_tokens(self) -> int:
        return sum(p.token_length() for p in self.packets)

    def to_text(self) -> str:
        return "\n".join(p.content for p in self.packets)


class BaseAgent(ABC):
    """Abstract base agent for the multi-agent pipeline."""

    def __init__(self, role: AgentRole, model_config: Optional[Dict] = None):
        self.role = role
        self.model_config = model_config or {}
        self.trace = CommTrace(agent_role=role)
        self._call_count = 0

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent's role."""

    @abstractmethod
    def build_user_prompt(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        """Build the user prompt incorporating received communications."""

    def run(self, task_input: str, received_comms: List[CommPacket],
            llm_fn=None) -> str:
        """
        Execute one step of the agent.
        llm_fn: callable(system, user) -> str
        """
        system = self.system_prompt()
        user = self.build_user_prompt(task_input, received_comms)

        if llm_fn is None:
            response = self._default_response(task_input, received_comms)
        else:
            response = llm_fn(system, user)

        self._call_count += 1
        self.trace.add(
            step_idx=self._call_count,
            content=response,
            metadata={"task_input_hash": hashlib.md5(
                task_input.encode()).hexdigest()[:8]},
        )
        return response

    def _default_response(self, task_input: str,
                          received_comms: List[CommPacket]) -> str:
        return f"[{self.role.value}] processed: {task_input[:100]}..."

    def reset(self):
        self.trace = CommTrace(agent_role=self.role)
        self._call_count = 0

    def get_trace(self) -> CommTrace:
        return self.trace

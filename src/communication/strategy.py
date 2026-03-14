"""
Communication compression strategies.

Implements all selection/truncation policies described in the experiment plan:
- Full communication (no compression)
- Recency (keep last k packets)
- Fixed truncation (keep first k packets)
- Uniform sampling
- Random sampling
- Role-aware selection
- Oracle (utility-based optimal selection)
"""
from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from config import AgentRole, CommStrategy
from src.agents.base import CommPacket


class CommunicationStrategy(ABC):
    """Base class for communication compression strategies."""

    def __init__(self, budget: float = 1.0):
        assert 0.0 < budget <= 1.0, f"Budget must be in (0, 1], got {budget}"
        self.budget = budget

    @abstractmethod
    def select(self, packets: List[CommPacket],
               receiver_role: Optional[AgentRole] = None) -> List[CommPacket]:
        """Select packets to retain under the budget constraint."""

    def _target_count(self, total: int) -> int:
        return max(1, math.ceil(total * self.budget))

    @property
    def name(self) -> str:
        return self.__class__.__name__


class FullStrategy(CommunicationStrategy):
    """No compression — keep everything."""

    def __init__(self):
        super().__init__(budget=1.0)

    def select(self, packets, receiver_role=None):
        return list(packets)


class RecencyStrategy(CommunicationStrategy):
    """Keep only the most recent k packets (tail-only)."""

    def select(self, packets, receiver_role=None):
        k = self._target_count(len(packets))
        return packets[-k:]


class FixedTruncationStrategy(CommunicationStrategy):
    """Keep the first k packets (head-only)."""

    def select(self, packets, receiver_role=None):
        k = self._target_count(len(packets))
        return packets[:k]


class UniformStrategy(CommunicationStrategy):
    """Uniformly sample k packets, preserving temporal order."""

    def select(self, packets, receiver_role=None):
        n = len(packets)
        k = self._target_count(n)
        if k >= n:
            return list(packets)
        indices = np.linspace(0, n - 1, k, dtype=int)
        return [packets[i] for i in indices]


class RandomStrategy(CommunicationStrategy):
    """Randomly sample k packets, preserving temporal order."""

    def select(self, packets, receiver_role=None):
        n = len(packets)
        k = self._target_count(n)
        if k >= n:
            return list(packets)
        chosen = sorted(random.sample(range(n), k))
        return [packets[i] for i in chosen]


class RoleAwareStrategy(CommunicationStrategy):
    """
    Role-aware communication selection.

    Uses per-role priority weights and receiver-conditioned scoring
    to select the most valuable packets for a specific receiver.
    """

    def __init__(self, budget: float = 1.0,
                 role_weights: Optional[Dict[AgentRole, float]] = None,
                 receiver_weights: Optional[Dict[tuple, float]] = None):
        super().__init__(budget)
        self.role_weights = role_weights or {
            AgentRole.PLANNER: 1.0,
            AgentRole.CRITIC: 0.8,
            AgentRole.REFINER: 0.9,
            AgentRole.JUDGER: 0.7,
        }
        self.receiver_weights = receiver_weights or {}

    def _score_packet(self, packet: CommPacket,
                      receiver_role: Optional[AgentRole] = None) -> float:
        role_w = self.role_weights.get(packet.agent_role, 1.0)
        pair_key = (packet.agent_role, receiver_role)
        pair_w = self.receiver_weights.get(pair_key, 1.0)
        util = packet.utility if packet.utility > 0 else 0.5
        return role_w * pair_w * util

    def select(self, packets, receiver_role=None):
        n = len(packets)
        k = self._target_count(n)
        if k >= n:
            return list(packets)

        scored = [(i, self._score_packet(p, receiver_role))
                  for i, p in enumerate(packets)]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scored[:k]])
        return [packets[i] for i in selected_indices]

    def update_weights(self, role_weights: Optional[Dict] = None,
                       receiver_weights: Optional[Dict] = None):
        if role_weights:
            self.role_weights.update(role_weights)
        if receiver_weights:
            self.receiver_weights.update(receiver_weights)


class OracleStrategy(CommunicationStrategy):
    """
    Oracle selection: pick top-k packets by pre-computed utility.
    Used as an upper-bound reference (requires utility labels).
    """

    def select(self, packets, receiver_role=None):
        n = len(packets)
        k = self._target_count(n)
        if k >= n:
            return list(packets)

        scored = [(i, p.utility) for i, p in enumerate(packets)]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scored[:k]])
        return [packets[i] for i in selected_indices]


def get_strategy(strategy: CommStrategy, budget: float = 1.0,
                 **kwargs) -> CommunicationStrategy:
    """Factory function to create a communication strategy."""
    mapping = {
        CommStrategy.FULL: lambda: FullStrategy(),
        CommStrategy.RECENCY: lambda: RecencyStrategy(budget),
        CommStrategy.FIXED_TRUNCATION: lambda: FixedTruncationStrategy(budget),
        CommStrategy.UNIFORM: lambda: UniformStrategy(budget),
        CommStrategy.RANDOM: lambda: RandomStrategy(budget),
        CommStrategy.ROLE_AWARE: lambda: RoleAwareStrategy(budget, **kwargs),
        CommStrategy.ORACLE: lambda: OracleStrategy(budget),
    }
    factory = mapping.get(strategy)
    if factory is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    return factory()

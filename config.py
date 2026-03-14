"""
Global configuration for the multi-agent communication compression experiments.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class AgentRole(Enum):
    PLANNER = "planner"
    CRITIC = "critic"
    REFINER = "refiner"
    JUDGER = "judger"


class CommStrategy(Enum):
    FULL = "full"
    RECENCY = "recency"
    FIXED_TRUNCATION = "fixed_truncation"
    UNIFORM = "uniform"
    RANDOM = "random"
    ROLE_AWARE = "role_aware"
    ORACLE = "oracle"


class TaskType(Enum):
    GSM8K = "gsm8k"
    MATH = "math"
    MBPP_PLUS = "mbpp_plus"
    HUMANEVAL_PLUS = "humaneval_plus"
    MEDQA = "medqa"


TASK_MAX_TOKENS: Dict[TaskType, int] = {
    TaskType.MEDQA: 128,
    TaskType.GSM8K: 512,
    TaskType.MATH: 768,
    TaskType.MBPP_PLUS: 512,
    TaskType.HUMANEVAL_PLUS: 512,
}


def get_max_tokens(task: Optional[TaskType] = None) -> int:
    """Return max_tokens for a given task, falling back to default 512."""
    if task is not None:
        return TASK_MAX_TOKENS.get(task, 512)
    return 512


@dataclass
class ModelConfig:
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    use_local: bool = False
    local_model_path: Optional[str] = None


@dataclass
class CommunicationConfig:
    strategy: CommStrategy = CommStrategy.FULL
    budget: float = 1.0  # ratio in (0, 1]
    packet_size: int = 8  # tokens per packet
    use_adapter: bool = False
    adapter_hidden_dim: int = 256
    adapter_lr: float = 1e-4
    adapter_epochs: int = 5


@dataclass
class PipelineConfig:
    roles: List[AgentRole] = field(default_factory=lambda: [
        AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.REFINER, AgentRole.JUDGER
    ])
    num_rounds: int = 1
    comm_config: CommunicationConfig = field(default_factory=CommunicationConfig)


@dataclass
class ExperimentConfig:
    task: TaskType = TaskType.GSM8K
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    num_seeds: int = 3
    budgets: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    output_dir: str = "results"
    num_samples: Optional[int] = None  # None = use all
    difficulty_bins: int = 3  # easy / medium / hard


DEFAULT_TASKS = [
    TaskType.GSM8K,
    TaskType.MATH,
    TaskType.MBPP_PLUS,
    TaskType.HUMANEVAL_PLUS,
    TaskType.MEDQA,
]

BUDGET_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
BUDGET_SWEEP_MINIMAL = [0.1, 0.2, 0.4, 0.6, 1.0]

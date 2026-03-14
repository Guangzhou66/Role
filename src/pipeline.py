"""
Multi-agent pipeline with configurable communication strategies.

Orchestrates the flow:
  Planner → Critic → Refiner → Judger
with communication compression applied between agents.
"""
from __future__ import annotations
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from config import (
    AgentRole, CommStrategy, CommunicationConfig,
    ExperimentConfig, PipelineConfig,
)
from src.agents.base import BaseAgent, CommPacket, CommTrace
from src.agents import PlannerAgent, CriticAgent, RefinerAgent, JudgerAgent
from src.communication.strategy import CommunicationStrategy, get_strategy
from src.communication.adapter import ReceiverAdapter


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from a single pipeline run on one sample."""
    final_answer: str
    traces: Dict[str, CommTrace] = field(default_factory=dict)
    all_packets: List[CommPacket] = field(default_factory=list)
    compressed_packets: Dict[str, List[CommPacket]] = field(default_factory=dict)
    latency_ms: float = 0.0
    total_tokens_full: int = 0
    total_tokens_compressed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentPipeline:
    """
    Configurable multi-agent pipeline.

    Supports:
    - Variable number of agents (2, 3, 4)
    - Different communication strategies per edge
    - Receiver adaptation via adapter
    """

    def __init__(
        self,
        config: PipelineConfig,
        llm_fn: Optional[Callable] = None,
        adapter: Optional[ReceiverAdapter] = None,
    ):
        self.config = config
        self.llm_fn = llm_fn
        self.adapter = adapter

        self.agents: Dict[AgentRole, BaseAgent] = {}
        self._init_agents()
        self.strategy = get_strategy(
            config.comm_config.strategy,
            config.comm_config.budget,
        )

    def _init_agents(self):
        role_to_cls = {
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.CRITIC: CriticAgent,
            AgentRole.REFINER: RefinerAgent,
            AgentRole.JUDGER: JudgerAgent,
        }
        for role in self.config.roles:
            cls = role_to_cls[role]
            self.agents[role] = cls()

    def run(self, task_input: str) -> PipelineResult:
        """Run the full pipeline on a single task input."""
        start = time.time()
        for agent in self.agents.values():
            agent.reset()

        all_packets: List[CommPacket] = []
        compressed_per_agent: Dict[str, List[CommPacket]] = {}

        roles = self.config.roles
        final_answer = ""

        for idx, role in enumerate(roles):
            agent = self.agents[role]

            if idx == 0:
                received = []
            else:
                received = self.strategy.select(
                    all_packets, receiver_role=role,
                )
                compressed_per_agent[role.value] = list(received)

            response = agent.run(task_input, received, self.llm_fn)

            for pkt in agent.get_trace().packets:
                if pkt not in all_packets:
                    all_packets.append(pkt)

            if role == roles[-1]:
                final_answer = response

        elapsed = (time.time() - start) * 1000

        total_full = sum(p.token_length() for p in all_packets)
        total_comp = sum(
            sum(p.token_length() for p in pkts)
            for pkts in compressed_per_agent.values()
        )

        return PipelineResult(
            final_answer=final_answer,
            traces={r.value: self.agents[r].get_trace() for r in roles},
            all_packets=all_packets,
            compressed_packets=compressed_per_agent,
            latency_ms=elapsed,
            total_tokens_full=total_full,
            total_tokens_compressed=total_comp,
        )

    def run_batch(self, task_inputs: List[str]) -> List[PipelineResult]:
        """Run pipeline on multiple inputs, batching LLM calls per agent stage."""
        n = len(task_inputs)
        if n == 0:
            return []

        has_batch = (
            self.llm_fn is not None
            and hasattr(self.llm_fn, "batch")
            and callable(self.llm_fn.batch)
        )

        all_packets: List[List[CommPacket]] = [[] for _ in range(n)]
        compressed: List[Dict[str, List[CommPacket]]] = [{} for _ in range(n)]
        final_answers = [""] * n
        call_counts: List[Dict[AgentRole, int]] = [
            {r: 0 for r in self.config.roles} for _ in range(n)
        ]

        start = time.time()
        roles = self.config.roles

        for stage_idx, role in enumerate(roles):
            agent_template = self.agents[role]

            pairs: List[Tuple[str, str]] = []
            received_per_sample: List[List[CommPacket]] = []

            for si in range(n):
                if stage_idx == 0:
                    received = []
                else:
                    received = self.strategy.select(
                        all_packets[si], receiver_role=role,
                    )
                    compressed[si][role.value] = list(received)
                received_per_sample.append(received)

                system = agent_template.system_prompt()
                user = agent_template.build_user_prompt(
                    task_inputs[si], received)
                pairs.append((system, user))

            if self.llm_fn is None:
                responses = [
                    agent_template._default_response(task_inputs[si], [])
                    for si in range(n)
                ]
            elif has_batch:
                responses = self.llm_fn.batch(pairs)
            else:
                responses = [self.llm_fn(s, u) for s, u in pairs]

            for si in range(n):
                call_counts[si][role] += 1
                pkt = CommPacket(
                    agent_role=role,
                    step_idx=call_counts[si][role],
                    content=responses[si],
                    metadata={
                        "task_input_hash": hashlib.md5(
                            task_inputs[si].encode()).hexdigest()[:8]
                    },
                )
                all_packets[si].append(pkt)
                if role == roles[-1]:
                    final_answers[si] = responses[si]

        elapsed = (time.time() - start) * 1000

        results = []
        for si in range(n):
            total_full = sum(p.token_length() for p in all_packets[si])
            total_comp = sum(
                sum(p.token_length() for p in pkts)
                for pkts in compressed[si].values()
            )
            results.append(PipelineResult(
                final_answer=final_answers[si],
                all_packets=all_packets[si],
                compressed_packets=compressed[si],
                latency_ms=elapsed / n,
                total_tokens_full=total_full,
                total_tokens_compressed=total_comp,
            ))
        return results

    def run_with_leave_one_out(
        self, task_input: str, eval_fn: Callable[[str], float],
    ) -> List[Dict]:
        """
        Run leave-one-packet-out analysis for utility estimation (B1/B2).
        First runs full pipeline, then re-runs removing one packet at a time.
        """
        full_result = self.run(task_input)
        full_perf = eval_fn(full_result.final_answer)

        utility_results = []
        for i, pkt in enumerate(full_result.all_packets):
            remaining = (full_result.all_packets[:i]
                         + full_result.all_packets[i + 1:])

            # Re-run judger with remaining packets
            judger = self.agents.get(AgentRole.JUDGER)
            if judger is None:
                judger = self.agents[self.config.roles[-1]]
            judger.reset()
            answer = judger.run(task_input, remaining, self.llm_fn)
            perf_without = eval_fn(answer)

            utility_results.append({
                "packet_idx": i,
                "agent_role": pkt.agent_role.value,
                "step_idx": pkt.step_idx,
                "utility": full_perf - perf_without,
                "position_ratio": i / max(len(full_result.all_packets) - 1, 1),
            })
            pkt.utility = full_perf - perf_without

        return utility_results

    def run_compress_single_role(
        self, task_input: str, compress_role: AgentRole,
    ) -> PipelineResult:
        """
        Run pipeline compressing only one role's communication (B3).
        Other roles keep full communication.
        """
        start = time.time()
        for agent in self.agents.values():
            agent.reset()

        all_packets: List[CommPacket] = []
        roles = self.config.roles
        final_answer = ""

        for idx, role in enumerate(roles):
            agent = self.agents[role]
            if idx == 0:
                received = list(all_packets)
            else:
                pkts_to_compress = [
                    p for p in all_packets if p.agent_role == compress_role
                ]
                pkts_keep_full = [
                    p for p in all_packets if p.agent_role != compress_role
                ]
                compressed = self.strategy.select(
                    pkts_to_compress, receiver_role=role,
                )
                received = sorted(
                    pkts_keep_full + compressed,
                    key=lambda p: p.step_idx,
                )

            response = agent.run(task_input, received, self.llm_fn)
            for pkt in agent.get_trace().packets:
                if pkt not in all_packets:
                    all_packets.append(pkt)
            if role == roles[-1]:
                final_answer = response

        elapsed = (time.time() - start) * 1000
        return PipelineResult(
            final_answer=final_answer,
            traces={r.value: self.agents[r].get_trace() for r in roles},
            all_packets=all_packets,
            latency_ms=elapsed,
            metadata={"compressed_role": compress_role.value},
        )


def create_llm_fn(model_config) -> Callable:
    """Create an LLM callable from model configuration."""
    if model_config.get("use_local"):
        return _create_local_llm_fn(model_config)
    return _create_api_llm_fn(model_config)


def _create_api_llm_fn(config) -> Callable:
    """Create OpenAI-compatible API callable with mutable max_tokens."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.get("api_key", os.environ.get("OPENAI_API_KEY")),
            base_url=config.get("api_base"),
        )
    except ImportError:
        logger.warning("openai package not installed, using mock LLM")
        return _mock_llm_fn

    model_name = config.get("model_name", "gpt-4o-mini")
    temperature = config.get("temperature", 0.0)
    runtime = {"max_tokens": config.get("max_tokens", 512)}

    def llm_fn(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=runtime["max_tokens"],
        )
        return response.choices[0].message.content

    def _set_max_tokens(val: int):
        runtime["max_tokens"] = val

    llm_fn.set_max_tokens = _set_max_tokens
    return llm_fn


def _create_local_llm_fn(config) -> Callable:
    """Create local model callable with NF4 quantization and batch support."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        model_path = config.get("local_model_path", config.get("model_name"))

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        load_kwargs: dict = {}

        if device == "cuda":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = nf4_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device != "cuda":
            model = model.to(device)
        model.eval()
    except ImportError:
        logger.warning("transformers / bitsandbytes not installed, using mock LLM")
        return _mock_llm_fn

    temperature = config.get("temperature", 0.0)
    runtime = {"max_tokens": config.get("max_tokens", 512)}

    def _build_prompt(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return f"<|system|>{system}<|user|>{user}<|assistant|>"

    def llm_fn(system: str, user: str) -> str:
        prompt = _build_prompt(system, user)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=runtime["max_tokens"],
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)

    def batch_fn(pairs: List[Tuple[str, str]]) -> List[str]:
        """Batch inference: pairs = [(system, user), ...]."""
        prompts = [_build_prompt(s, u) for s, u in pairs]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=runtime["max_tokens"],
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        results = []
        for i, seq in enumerate(outputs):
            prompt_len = inputs.attention_mask[i].sum().item()
            results.append(
                tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
        return results

    def _set_max_tokens(val: int):
        runtime["max_tokens"] = val

    llm_fn.batch = batch_fn
    llm_fn.set_max_tokens = _set_max_tokens
    return llm_fn


def _mock_llm_fn(system: str, user: str) -> str:
    """Mock LLM for testing without API access."""
    return f"[Mock response to: {user[:200]}...]"

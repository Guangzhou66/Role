"""
Experiment D: Scaling / Generalization Validation

D1 - Length Scaling
D2 - Agent Count Scaling
D3 - Cross-task Transfer (optional)

Goal: Prove that the method advantage grows with communication length
and collaboration complexity, showing it addresses a fundamental
problem rather than exploiting a dataset-specific heuristic.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import (
    AgentRole, CommStrategy, CommunicationConfig,
    DEFAULT_TASKS, ExperimentConfig, PipelineConfig,
    BUDGET_SWEEP, get_max_tokens,
)
from src.pipeline import MultiAgentPipeline
from src.datasets.loader import get_loader, load_cached
from src.communication.adapter import ReceiverAdapter
from src.metrics.core import cheap_full_gap, aubc
from src.metrics.evaluator import task_accuracy, task_correct_list
from src.utils.helpers import save_json, set_seed, ensure_dir

logger = logging.getLogger(__name__)


def _set_task_max_tokens(llm_fn, task):
    if llm_fn is not None and hasattr(llm_fn, "set_max_tokens"):
        llm_fn.set_max_tokens(get_max_tokens(task))


def _build_pipeline(roles, strat, budget, llm_fn, use_adapter=False):
    comm_cfg = CommunicationConfig(strategy=strat, budget=budget,
                                   use_adapter=use_adapter)
    adapter = ReceiverAdapter() if use_adapter else None
    pipe = MultiAgentPipeline(
        PipelineConfig(roles=roles, comm_config=comm_cfg),
        llm_fn, adapter=adapter,
    )
    return pipe


# ════════════════════════════════════════════════
#  D1: Length Scaling
# ════════════════════════════════════════════════

class ExpD1_LengthScaling:
    """
    Test whether the method advantage grows with communication length.

    Approach: vary the number of reasoning rounds (each round adds
    more communication packets) and compare methods.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget
        self.round_counts = [1, 2, 3, 4]

    def run(self) -> Dict[str, Any]:
        results = {}
        methods = [
            ("full", CommStrategy.FULL, 1.0, False),
            ("recency", CommStrategy.RECENCY, self.budget, False),
            ("role_aware", CommStrategy.ROLE_AWARE, self.budget, False),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, self.budget, True),
        ]

        for task in DEFAULT_TASKS[:3]:
            logger.info(f"D1: Length scaling for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for n_rounds in self.round_counts:
                round_results = {}
                for name, strat, bgt, use_adapter in methods:
                    comm_cfg = CommunicationConfig(
                        strategy=strat, budget=bgt,
                        use_adapter=use_adapter)
                    adapter = ReceiverAdapter() if use_adapter else None
                    pipe_cfg = PipelineConfig(
                        roles=self.config.pipeline.roles,
                        num_rounds=n_rounds,
                        comm_config=comm_cfg,
                    )
                    pipe = MultiAgentPipeline(pipe_cfg, self.llm_fn, adapter)

                    preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                    perf = task_accuracy(task, preds, samples)
                    round_results[name] = {"accuracy": perf}

                # Compute gaps
                full_perf = round_results["full"]["accuracy"]
                for name in round_results:
                    if name != "full":
                        round_results[name]["gap_to_full"] = cheap_full_gap(
                            full_perf, round_results[name]["accuracy"])
                        round_results[name]["gap_to_recency"] = (
                            round_results[name]["accuracy"]
                            - round_results["recency"]["accuracy"]
                        )

                task_results[f"rounds_{n_rounds}"] = round_results

            # Scaling slope: does method advantage grow with length?
            advantage_curve = {}
            for name in ["role_aware", "role_aware+adapter"]:
                advantages = []
                for n_rounds in self.round_counts:
                    key = f"rounds_{n_rounds}"
                    adv = task_results[key].get(name, {}).get(
                        "gap_to_recency", 0)
                    advantages.append(adv)
                advantage_curve[name] = {
                    "advantages": advantages,
                    "scaling_positive": all(
                        advantages[i] <= advantages[i + 1]
                        for i in range(len(advantages) - 1)
                    ) if len(advantages) > 1 else False,
                }

            task_results["scaling_analysis"] = advantage_curve
            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  D2: Agent Count Scaling
# ════════════════════════════════════════════════

class ExpD2_AgentCountScaling:
    """
    Test whether the method scales to different numbers of agents.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}

        topologies = {
            "2-agent": [AgentRole.PLANNER, AgentRole.JUDGER],
            "3-agent": [
                AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.JUDGER],
            "4-agent": [
                AgentRole.PLANNER, AgentRole.CRITIC,
                AgentRole.REFINER, AgentRole.JUDGER],
        }

        methods = [
            ("full", CommStrategy.FULL, 1.0, False),
            ("recency", CommStrategy.RECENCY, self.budget, False),
            ("role_aware", CommStrategy.ROLE_AWARE, self.budget, False),
            ("recency+adapter", CommStrategy.RECENCY, self.budget, True),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, self.budget, True),
        ]

        for task in DEFAULT_TASKS[:3]:
            logger.info(f"D2: Agent count scaling for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for topo_name, roles in topologies.items():
                topo_results = {}
                for name, strat, bgt, use_adapter in methods:
                    pipe = _build_pipeline(
                        roles, strat, bgt, self.llm_fn, use_adapter)
                    preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                    perf = task_accuracy(task, preds, samples)
                    topo_results[name] = {"accuracy": perf}

                # Compute relative metrics
                full_perf = topo_results["full"]["accuracy"]
                rec_perf = topo_results["recency"]["accuracy"]
                for name in topo_results:
                    if name not in ("full",):
                        topo_results[name]["gap_to_full"] = cheap_full_gap(
                            full_perf, topo_results[name]["accuracy"])
                    if name not in ("full", "recency"):
                        topo_results[name]["gap_to_recency"] = (
                            topo_results[name]["accuracy"] - rec_perf)

                task_results[topo_name] = topo_results

            # Does role-aware advantage grow with agent count?
            advantage_by_count = {}
            for name in ["role_aware", "role_aware+adapter"]:
                advs = []
                for topo_name in topologies:
                    adv = task_results[topo_name].get(name, {}).get(
                        "gap_to_recency", 0)
                    advs.append(adv)
                advantage_by_count[name] = advs

            task_results["scaling_analysis"] = {
                "advantage_by_agent_count": advantage_by_count,
                "agent_counts": [2, 3, 4],
            }
            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  D3: Cross-task Transfer
# ════════════════════════════════════════════════

class ExpD3_CrossTaskTransfer:
    """
    Test whether selector/adapter learned on one task domain
    transfers to another domain.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}

        train_tasks = DEFAULT_TASKS[:2]  # e.g. GSM8K, MATH
        test_tasks = DEFAULT_TASKS[2:]   # e.g. MBPP+, HumanEval+, MedQA

        # Phase 1: Train role weights on train tasks
        # (simplified: collect role sensitivities and use as weights)
        logger.info("D3: Learning role weights from train tasks")
        role_weights = {
            AgentRole.PLANNER: 0.0,
            AgentRole.CRITIC: 0.0,
            AgentRole.REFINER: 0.0,
        }

        for task in train_tasks:
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=min(
                self.config.num_samples or 30, 30))

            full_pipe = _build_pipeline(
                self.config.pipeline.roles,
                CommStrategy.FULL, 1.0, self.llm_fn)
            full_preds = [r.final_answer for r in full_pipe.run_batch([s.input for s in samples])]
            full_perf = task_accuracy(task, full_preds, samples)

            for role in [AgentRole.PLANNER, AgentRole.CRITIC,
                         AgentRole.REFINER]:
                comp_cfg = CommunicationConfig(
                    strategy=CommStrategy.RECENCY, budget=self.budget)
                comp_pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comp_cfg),
                    self.llm_fn,
                )
                comp_preds = []
                for s in samples:
                    res = comp_pipe.run_compress_single_role(s.input, role)
                    comp_preds.append(res.final_answer)
                comp_perf = task_accuracy(task, comp_preds, samples)
                role_weights[role] += (full_perf - comp_perf)

        # Normalize weights
        total = sum(role_weights.values()) or 1.0
        role_weights = {r: v / total for r, v in role_weights.items()}

        results["learned_role_weights"] = {
            r.value: w for r, w in role_weights.items()
        }

        # Phase 2: Evaluate on test tasks with transferred weights
        logger.info("D3: Evaluating transfer on test tasks")
        for task in test_tasks:
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            methods = {
                "recency": (CommStrategy.RECENCY, None),
                "role_aware_transferred": (CommStrategy.ROLE_AWARE,
                                           role_weights),
            }

            task_results = {}
            for name, (strat, weights) in methods.items():
                from src.communication.strategy import RoleAwareStrategy
                if weights:
                    strategy_obj = RoleAwareStrategy(
                        budget=self.budget, role_weights=weights)
                    comm_cfg = CommunicationConfig(
                        strategy=strat, budget=self.budget)
                else:
                    comm_cfg = CommunicationConfig(
                        strategy=strat, budget=self.budget)

                pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg),
                    self.llm_fn,
                )
                preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                perf = task_accuracy(task, preds, samples)
                task_results[name] = {"accuracy": perf}

            if "role_aware_transferred" in task_results and "recency" in task_results:
                task_results["transfer_gain"] = (
                    task_results["role_aware_transferred"]["accuracy"]
                    - task_results["recency"]["accuracy"]
                )

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════

def run_experiment_d(config: ExperimentConfig, llm_fn: Callable = None,
                     output_dir: str = "results/exp_d") -> Dict[str, Any]:
    """Run all Experiment D sub-experiments."""
    ensure_dir(output_dir)
    all_results = {}

    experiments = [
        ("d1", "Length Scaling", ExpD1_LengthScaling),
        ("d2", "Agent Count Scaling", ExpD2_AgentCountScaling),
        ("d3", "Cross-task Transfer", ExpD3_CrossTaskTransfer),
    ]

    for key, name, cls in experiments:
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT {key.upper()}: {name}")
        logger.info("=" * 60)
        exp = cls(config, llm_fn)
        res = exp.run()
        save_json(res, os.path.join(
            output_dir, f"{key}_{name.lower().replace(' ', '_').replace('-', '_')}.json"))
        all_results[key] = res

    return all_results

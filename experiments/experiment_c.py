"""
Experiment C: Method Validation (提出并验证方法)

C1 - Method-to-Problem Alignment
C2 - Failure Case Recovery
C3 - Main Results under Fixed Budgets
C4 - Budget-Performance Pareto
C5 - Oracle Upper Bound
C6 - Statistical Reliability
C7 - Case Studies
C8 - Generalization (model & topology)

Goal: Prove that role-aware selection fixes "what to send" and
receiver adaptation fixes "how to read", and together they
produce stable improvements.
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
from src.metrics.core import (
    cheap_full_gap, normalized_gap,
    selection_gain, adaptation_gain, joint_gain, synergy,
    teacher_consistency,
    oracle_gap, aubc, mean_std_ci,
    efficiency_aware_score,
)
from src.metrics.evaluator import task_accuracy, task_correct_list, get_evaluator
from src.utils.helpers import save_json, set_seed, ensure_dir

logger = logging.getLogger(__name__)


def _set_task_max_tokens(llm_fn, task):
    if llm_fn is not None and hasattr(llm_fn, "set_max_tokens"):
        llm_fn.set_max_tokens(get_max_tokens(task))


def _build_pipeline(config, strat, budget, llm_fn, use_adapter=False):
    """Helper to build a pipeline with specific strategy."""
    comm_cfg = CommunicationConfig(strategy=strat, budget=budget,
                                   use_adapter=use_adapter)
    adapter = ReceiverAdapter() if use_adapter else None
    pipe = MultiAgentPipeline(
        PipelineConfig(roles=config.pipeline.roles, comm_config=comm_cfg),
        llm_fn, adapter=adapter,
    )
    return pipe


def _evaluate(pipe, samples):
    """Run pipeline on samples using batch, return predictions."""
    return [r.final_answer for r in pipe.run_batch([s.input for s in samples])]


# ════════════════════════════════════════════════
#  C1: Method-to-Problem Alignment
# ════════════════════════════════════════════════

class ExpC1_MethodAlignment:
    """
    Compare 5 configurations to show each method module
    addresses its target problem:
      1. Full communication
      2. Recency truncation
      3. Role-aware selection only
      4. Recency + adapter
      5. Role-aware + adapter

    Compute G_sel, G_adp, G_joint, S (synergy).
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}
        configs = [
            ("full", CommStrategy.FULL, 1.0, False),
            ("recency", CommStrategy.RECENCY, self.budget, False),
            ("role_aware", CommStrategy.ROLE_AWARE, self.budget, False),
            ("recency+adapter", CommStrategy.RECENCY, self.budget, True),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, self.budget, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C1: Method alignment for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            perfs = {}
            all_preds = {}

            for name, strat, bgt, adapter in configs:
                pipe = _build_pipeline(
                    self.config, strat, bgt, self.llm_fn, adapter)
                preds = _evaluate(pipe, samples)
                perf = task_accuracy(task, preds, samples)
                perfs[name] = perf
                all_preds[name] = preds

            # Compute gains
            p_rec = perfs["recency"]
            p_role = perfs["role_aware"]
            p_rec_adp = perfs["recency+adapter"]
            p_role_adp = perfs["role_aware+adapter"]

            task_results = {
                "performances": perfs,
                "G_sel": selection_gain(p_role, p_rec),
                "G_adp": adaptation_gain(p_rec_adp, p_rec),
                "G_joint": joint_gain(p_role_adp, p_rec),
                "synergy": synergy(p_role_adp, p_role, p_rec_adp, p_rec),
                "teacher_consistency": {
                    name: teacher_consistency(
                        all_preds[name], all_preds["full"])
                    for name in all_preds if name != "full"
                },
            }
            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C2: Failure Case Recovery
# ════════════════════════════════════════════════

class ExpC2_FailureRecovery:
    """
    On samples where full is correct but cheap is wrong,
    measure recovery rate of each method.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}
        methods = [
            ("role_aware", CommStrategy.ROLE_AWARE, False),
            ("recency+adapter", CommStrategy.RECENCY, True),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C2: Failure recovery for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            # Get full and cheap predictions
            full_pipe = _build_pipeline(
                self.config, CommStrategy.FULL, 1.0, self.llm_fn)
            cheap_pipe = _build_pipeline(
                self.config, CommStrategy.RECENCY, self.budget, self.llm_fn)

            full_preds = _evaluate(full_pipe, samples)
            cheap_preds = _evaluate(cheap_pipe, samples)

            # Identify failure set via task-aware matching
            full_correct = task_correct_list(task, full_preds, samples)
            cheap_correct = task_correct_list(task, cheap_preds, samples)
            failure_indices = [
                i for i in range(len(samples))
                if full_correct[i] and not cheap_correct[i]
            ]

            task_results = {
                "n_total": len(samples),
                "n_failure_set": len(failure_indices),
                "failure_fraction": len(failure_indices) / max(len(samples), 1),
            }

            for name, strat, use_adapter in methods:
                pipe = _build_pipeline(
                    self.config, strat, self.budget, self.llm_fn, use_adapter)
                method_preds = _evaluate(pipe, samples)

                method_correct = task_correct_list(task, method_preds, samples)
                n_recovered = (
                    sum(method_correct[i] for i in failure_indices)
                    if failure_indices else 0
                )
                rec_rate = (
                    n_recovered / len(failure_indices)
                    if failure_indices else 0.0
                )
                task_results[name] = {
                    "recovery_rate": rec_rate,
                    "n_recovered": n_recovered,
                }

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C3: Main Results under Fixed Budgets
# ════════════════════════════════════════════════

class ExpC3_MainResults:
    """
    Comprehensive comparison table at fixed budgets (20%, 40%, 100%).
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn
        self.budgets = [0.2, 0.4, 1.0]

    def run(self) -> Dict[str, Any]:
        results = {}

        all_methods = [
            ("full", CommStrategy.FULL, 1.0, False),
            ("recency", CommStrategy.RECENCY, None, False),
            ("random", CommStrategy.RANDOM, None, False),
            ("uniform", CommStrategy.UNIFORM, None, False),
            ("role_aware", CommStrategy.ROLE_AWARE, None, False),
            ("recency+adapter", CommStrategy.RECENCY, None, True),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, None, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C3: Main results for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for budget in self.budgets:
                budget_results = {}
                for name, strat, fixed_bgt, use_adapter in all_methods:
                    bgt = fixed_bgt if fixed_bgt else budget
                    pipe = _build_pipeline(
                        self.config, strat, bgt, self.llm_fn, use_adapter)

                    import time
                    start = time.time()
                    batch_res = pipe.run_batch([s.input for s in samples])
                    elapsed = (time.time() - start) * 1000
                    preds = [r.final_answer for r in batch_res]

                    perf = task_accuracy(task, preds, samples)
                    budget_results[name] = {
                        "accuracy": perf,
                        "avg_latency_ms": elapsed / max(len(samples), 1),
                    }

                task_results[f"budget_{budget}"] = budget_results

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C4: Budget-Performance Pareto
# ════════════════════════════════════════════════

class ExpC4_BudgetPareto:
    """
    Sweep budgets and draw Pareto curves for each method.
    Compute AUBC for each.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn
        self.budgets = BUDGET_SWEEP

    def run(self) -> Dict[str, Any]:
        results = {}
        methods = [
            ("recency", CommStrategy.RECENCY, False),
            ("uniform", CommStrategy.UNIFORM, False),
            ("role_aware", CommStrategy.ROLE_AWARE, False),
            ("recency+adapter", CommStrategy.RECENCY, True),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C4: Pareto curves for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for name, strat, use_adapter in methods:
                curve = {}
                perfs_list = []
                for b in self.budgets:
                    pipe = _build_pipeline(
                        self.config, strat, b, self.llm_fn, use_adapter)
                    preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                    perf = task_accuracy(task, preds, samples)
                    curve[str(b)] = perf
                    perfs_list.append(perf)

                area = aubc(self.budgets, perfs_list)
                task_results[name] = {
                    "budget_curve": curve,
                    "aubc": area,
                }

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C5: Oracle Upper Bound
# ════════════════════════════════════════════════

class ExpC5_OracleUpperBound:
    """
    Compare methods against oracle (utility-based) selection
    to quantify remaining headroom.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}
        methods = [
            ("recency", CommStrategy.RECENCY),
            ("role_aware", CommStrategy.ROLE_AWARE),
            ("oracle", CommStrategy.ORACLE),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C5: Oracle upper bound for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for name, strat in methods:
                pipe = _build_pipeline(
                    self.config, strat, self.budget, self.llm_fn)
                preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                perf = task_accuracy(task, preds, samples)
                task_results[name] = {"accuracy": perf}

            oracle_perf = task_results["oracle"]["accuracy"]
            for name in ["recency", "role_aware"]:
                task_results[name]["gap_to_oracle"] = oracle_gap(
                    oracle_perf, task_results[name]["accuracy"])

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C6: Statistical Reliability
# ════════════════════════════════════════════════

class ExpC6_StatisticalReliability:
    """
    Run main experiments across multiple seeds and report
    mean ± std with confidence intervals.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 n_seeds: int = 3, budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.n_seeds = n_seeds
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}
        methods = [
            ("recency", CommStrategy.RECENCY, False),
            ("role_aware", CommStrategy.ROLE_AWARE, False),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"C6: Statistical reliability for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for name, strat, use_adapter in methods:
                seed_perfs = []
                for seed in range(self.n_seeds):
                    set_seed(seed * 42 + 7)
                    pipe = _build_pipeline(
                        self.config, strat, self.budget,
                        self.llm_fn, use_adapter)
                    preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                    perf = task_accuracy(task, preds, samples)
                    seed_perfs.append(perf)

                mu, sigma, (ci_lo, ci_hi) = mean_std_ci(seed_perfs)
                task_results[name] = {
                    "mean": mu,
                    "std": sigma,
                    "ci_95": [ci_lo, ci_hi],
                    "per_seed": seed_perfs,
                }

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  C7: Case Studies
# ════════════════════════════════════════════════

class ExpC7_CaseStudies:
    """
    Select 3-5 representative failure cases and show:
    - What cheap baseline got wrong
    - What role-aware selection chose
    - What the adapter recovered
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 n_cases: int = 5, budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.n_cases = n_cases
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS[:2]:
            logger.info(f"C7: Case studies for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            method_configs = {
                "full": (CommStrategy.FULL, 1.0, False),
                "recency": (CommStrategy.RECENCY, self.budget, False),
                "role_aware": (CommStrategy.ROLE_AWARE, self.budget, False),
                "role_aware+adapter": (CommStrategy.ROLE_AWARE, self.budget, True),
            }

            evaluator = get_evaluator(task)
            all_results_per_sample = [
                {"idx": i, "input": s.input[:500], "reference": s.reference}
                for i, s in enumerate(samples)
            ]

            for name, (strat, bgt, adapter) in method_configs.items():
                pipe = _build_pipeline(
                    self.config, strat, bgt, self.llm_fn, adapter)
                batch_res = pipe.run_batch([s.input for s in samples])
                for i, result in enumerate(batch_res):
                    all_results_per_sample[i][f"{name}_answer"] = result.final_answer
                    all_results_per_sample[i][f"{name}_correct"] = evaluator.evaluate_single(
                        result.final_answer, samples[i])
                    if name == "role_aware":
                        all_results_per_sample[i]["selected_packets"] = [
                            {
                                "role": p.agent_role.value,
                                "step": p.step_idx,
                                "content": p.content[:200],
                            }
                            for p in result.all_packets
                        ]

            # Select interesting cases: full correct, cheap wrong, method correct
            interesting = [
                s for s in all_results_per_sample
                if s.get("full_correct") and not s.get("recency_correct")
            ]
            selected = interesting[:self.n_cases]
            if len(selected) < self.n_cases:
                selected = all_results_per_sample[:self.n_cases]

            results[task.value] = {
                "cases": selected,
                "n_interesting_found": len(interesting),
            }

        return results


# ════════════════════════════════════════════════
#  C8: Generalization
# ════════════════════════════════════════════════

class ExpC8_Generalization:
    """
    Test generalization across:
    - Different agent counts (2, 3, 4)
    - Different base models (if configured)
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}

        # Topology generalization
        topologies = {
            "2-agent": [AgentRole.PLANNER, AgentRole.JUDGER],
            "3-agent": [AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.JUDGER],
            "4-agent": [AgentRole.PLANNER, AgentRole.CRITIC,
                        AgentRole.REFINER, AgentRole.JUDGER],
        }

        methods = [
            ("full", CommStrategy.FULL, 1.0, False),
            ("recency", CommStrategy.RECENCY, self.budget, False),
            ("role_aware", CommStrategy.ROLE_AWARE, self.budget, False),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, self.budget, True),
        ]

        for task in DEFAULT_TASKS[:3]:
            logger.info(f"C8: Generalization for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for topo_name, roles in topologies.items():
                topo_results = {}
                for name, strat, bgt, use_adapter in methods:
                    comm_cfg = CommunicationConfig(
                        strategy=strat, budget=bgt,
                        use_adapter=use_adapter)
                    adapter = ReceiverAdapter() if use_adapter else None
                    pipe = MultiAgentPipeline(
                        PipelineConfig(roles=roles, comm_config=comm_cfg),
                        self.llm_fn, adapter=adapter,
                    )
                    preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                    perf = task_accuracy(task, preds, samples)
                    topo_results[name] = {"accuracy": perf}

                task_results[topo_name] = topo_results

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════

def run_experiment_c(config: ExperimentConfig, llm_fn: Callable = None,
                     output_dir: str = "results/exp_c") -> Dict[str, Any]:
    """Run all Experiment C sub-experiments."""
    ensure_dir(output_dir)
    all_results = {}

    experiments = [
        ("c1", "Method-to-Problem Alignment", ExpC1_MethodAlignment),
        ("c2", "Failure Case Recovery", ExpC2_FailureRecovery),
        ("c3", "Main Results", ExpC3_MainResults),
        ("c4", "Budget-Performance Pareto", ExpC4_BudgetPareto),
        ("c5", "Oracle Upper Bound", ExpC5_OracleUpperBound),
        ("c6", "Statistical Reliability", ExpC6_StatisticalReliability),
        ("c7", "Case Studies", ExpC7_CaseStudies),
        ("c8", "Generalization", ExpC8_Generalization),
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

"""
Experiment A: Failure Observation (观察到失败现象)

A1 - Cheap-vs-Full Gap Profiling
A2 - Budget Sensitivity Sweep
A3 - Difficulty-Stratified Failure Analysis

Goal: Prove that cheap communication has systematic deficiencies,
especially on complex tasks and under low budgets.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import (
    BUDGET_SWEEP, CommStrategy, DEFAULT_TASKS,
    ExperimentConfig, PipelineConfig, CommunicationConfig, TaskType,
    get_max_tokens,
)
from src.pipeline import MultiAgentPipeline, PipelineResult
from src.datasets.loader import Sample, get_loader, load_cached
from src.metrics.core import (
    cheap_full_gap, normalized_gap,
    degradation_slope, aubc,
    compute_difficulty_scores, stratify_by_difficulty,
    failure_concentration_ratio,
    hard_vs_easy_amplification, mean_std_ci,
)
from src.metrics.evaluator import (
    task_accuracy, task_correct_list, task_fail_rate_group,
)
from src.utils.helpers import save_json, set_seed, ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def _set_task_max_tokens(llm_fn, task: TaskType):
    """Dynamically adjust max_tokens on the llm_fn for the current task."""
    if llm_fn is not None and hasattr(llm_fn, "set_max_tokens"):
        llm_fn.set_max_tokens(get_max_tokens(task))


def _run_pipeline_on_samples(pipeline, samples):
    """Run pipeline on all samples; uses batch when available."""
    results = pipeline.run_batch([s.input for s in samples])
    return [r.final_answer for r in results]


def _precompute_full_results(
    config: ExperimentConfig, llm_fn: Callable,
) -> Dict[str, Dict]:
    """Run full-communication pipeline once per task and cache predictions + latencies."""
    cache: Dict[str, Dict] = {}
    for task in DEFAULT_TASKS:
        logger.info(f"[Pre-compute] Full pipeline for {task.value}")
        _set_task_max_tokens(llm_fn, task)
        samples = load_cached(task, max_samples=config.num_samples)

        full_cfg = CommunicationConfig(strategy=CommStrategy.FULL, budget=1.0)
        pipe = MultiAgentPipeline(
            PipelineConfig(roles=config.pipeline.roles, comm_config=full_cfg),
            llm_fn,
        )

        batch_results = pipe.run_batch([s.input for s in samples])
        predictions = [r.final_answer for r in batch_results]
        latencies = [r.latency_ms for r in batch_results]

        perf = task_accuracy(task, predictions, samples)
        cache[task.value] = {
            "samples": samples,
            "predictions": predictions,
            "latencies": latencies,
            "accuracy": perf,
        }
    return cache


# ════════════════════════════════════════════════
#  A1: Cheap-vs-Full Gap Profiling
# ════════════════════════════════════════════════

class ExpA1_CheapFullGap:
    """
    Compare full communication vs cheap baselines across tasks.
    Output: Δ_t^{cheap-full} and normalized Δ̃_t for each task.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 full_cache: Optional[Dict[str, Dict]] = None):
        self.config = config
        self.llm_fn = llm_fn
        self.strategies = [
            CommStrategy.FULL,
            CommStrategy.RECENCY,
            CommStrategy.FIXED_TRUNCATION,
        ]
        self.tasks = DEFAULT_TASKS
        self.full_cache = full_cache or {}

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in self.tasks:
            logger.info(f"A1: Running task {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)
            task_results = {}

            cached = self.full_cache.get(task.value)

            for strategy in self.strategies:
                if strategy == CommStrategy.FULL and cached:
                    task_results[strategy.value] = {
                        "accuracy": cached["accuracy"],
                        "avg_latency_ms": float(np.mean(cached["latencies"])),
                        "n_samples": len(samples),
                    }
                    continue

                budget = 0.4 if strategy != CommStrategy.FULL else 1.0
                comm_cfg = CommunicationConfig(strategy=strategy, budget=budget)
                pipe_cfg = PipelineConfig(
                    roles=self.config.pipeline.roles,
                    comm_config=comm_cfg,
                )
                pipeline = MultiAgentPipeline(pipe_cfg, self.llm_fn)

                batch_results = pipeline.run_batch([s.input for s in samples])
                predictions = [r.final_answer for r in batch_results]
                latencies = [r.latency_ms for r in batch_results]

                perf = task_accuracy(task, predictions, samples)
                task_results[strategy.value] = {
                    "accuracy": perf,
                    "avg_latency_ms": float(np.mean(latencies)),
                    "n_samples": len(samples),
                }

            full_perf = task_results[CommStrategy.FULL.value]["accuracy"]
            for strat in self.strategies:
                if strat == CommStrategy.FULL:
                    continue
                cheap_perf = task_results[strat.value]["accuracy"]
                task_results[strat.value]["gap"] = cheap_full_gap(
                    full_perf, cheap_perf)
                task_results[strat.value]["normalized_gap"] = normalized_gap(
                    full_perf, cheap_perf)

            results[task.value] = task_results

        summary = self._summarize(results)
        results["summary"] = summary
        return results

    def _summarize(self, results: Dict) -> Dict:
        gaps = {}
        for task_name, task_res in results.items():
            if task_name == "summary":
                continue
            for strat in [CommStrategy.RECENCY.value,
                          CommStrategy.FIXED_TRUNCATION.value]:
                if strat in task_res and "gap" in task_res[strat]:
                    gaps.setdefault(strat, {})[task_name] = task_res[strat]["gap"]
        return {
            "gaps_by_strategy": gaps,
            "conclusion": (
                "If gaps are larger on complex tasks (MATH, HumanEval+, MedQA) "
                "than simple ones (GSM8K), cheap communication failures are "
                "systematic, not random."
            ),
        }


# ════════════════════════════════════════════════
#  A2: Budget Sensitivity Sweep
# ════════════════════════════════════════════════

class ExpA2_BudgetSensitivity:
    """
    Sweep communication budget from 10% to 100% and measure
    performance curves for cheap baselines.
    Output: Perf(b), Δ^{full-gap}(b), S(b1,b2), AUBC.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 full_cache: Optional[Dict[str, Dict]] = None):
        self.config = config
        self.llm_fn = llm_fn
        self.budgets = config.budgets
        self.baselines = [CommStrategy.RECENCY, CommStrategy.FIXED_TRUNCATION]
        self.full_cache = full_cache or {}

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS:
            logger.info(f"A2: Running task {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)
            task_results = {}

            cached = self.full_cache.get(task.value)
            if cached:
                full_preds = cached["predictions"]
                full_perf = cached["accuracy"]
            else:
                full_cfg = CommunicationConfig(
                    strategy=CommStrategy.FULL, budget=1.0)
                full_pipe = MultiAgentPipeline(
                    PipelineConfig(roles=self.config.pipeline.roles,
                                   comm_config=full_cfg),
                    self.llm_fn,
                )
                full_preds = _run_pipeline_on_samples(full_pipe, samples)
                full_perf = task_accuracy(task, full_preds, samples)
            task_results["full"] = {"accuracy": full_perf}

            for baseline in self.baselines:
                budget_curve = {}
                for b in self.budgets:
                    comm_cfg = CommunicationConfig(strategy=baseline, budget=b)
                    pipe_cfg = PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg,
                    )
                    pipeline = MultiAgentPipeline(pipe_cfg, self.llm_fn)
                    preds = _run_pipeline_on_samples(pipeline, samples)
                    perf = task_accuracy(task, preds, samples)
                    gap = cheap_full_gap(full_perf, perf)
                    norm_gap = normalized_gap(full_perf, perf)
                    budget_curve[str(b)] = {
                        "accuracy": perf,
                        "gap_to_full": gap,
                        "normalized_gap": norm_gap,
                    }

                sorted_budgets = sorted(self.budgets)
                slopes = {}
                for i in range(len(sorted_budgets) - 1):
                    b_lo = sorted_budgets[i]
                    b_hi = sorted_budgets[i + 1]
                    p_lo = budget_curve[str(b_lo)]["accuracy"]
                    p_hi = budget_curve[str(b_hi)]["accuracy"]
                    slope = degradation_slope(p_hi, p_lo, b_hi, b_lo)
                    slopes[f"{b_lo}->{b_hi}"] = slope

                perfs = [budget_curve[str(b)]["accuracy"]
                         for b in sorted_budgets]
                area = aubc(sorted_budgets, perfs)

                task_results[baseline.value] = {
                    "budget_curve": budget_curve,
                    "degradation_slopes": slopes,
                    "aubc": area,
                }

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  A3: Difficulty-Stratified Failure Analysis
# ════════════════════════════════════════════════

class ExpA3_DifficultyStratified:
    """
    Stratify samples by difficulty and analyze where cheap
    communication fails most.
    Output: FailRate_d, Δ_d, FCR_hard, HFA.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 n_difficulty_seeds: int = 3,
                 full_cache: Optional[Dict[str, Dict]] = None):
        self.config = config
        self.llm_fn = llm_fn
        self.n_difficulty_seeds = n_difficulty_seeds
        self.full_cache = full_cache or {}

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS:
            logger.info(f"A3: Running task {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            cached = self.full_cache.get(task.value)

            # Step 1: compute difficulty via full communication
            correct_per_seed = []
            for seed in range(self.n_difficulty_seeds):
                set_seed(seed)
                if seed == 0 and cached:
                    preds = cached["predictions"]
                else:
                    full_cfg = CommunicationConfig(
                        strategy=CommStrategy.FULL, budget=1.0)
                    pipe = MultiAgentPipeline(
                        PipelineConfig(
                            roles=self.config.pipeline.roles,
                            comm_config=full_cfg),
                        self.llm_fn,
                    )
                    preds = _run_pipeline_on_samples(pipe, samples)
                correct = task_correct_list(task, preds, samples)
                correct_per_seed.append(correct)

            difficulty = compute_difficulty_scores(correct_per_seed)
            groups = stratify_by_difficulty(difficulty,
                                           self.config.difficulty_bins)

            # Step 2: evaluate cheap baselines per difficulty group
            baselines = [CommStrategy.RECENCY, CommStrategy.FIXED_TRUNCATION]
            task_results = {"difficulty_distribution": {
                g: len(idx) for g, idx in groups.items()
            }}

            if cached:
                full_preds = cached["predictions"]
            else:
                full_preds = _run_pipeline_on_samples(
                    MultiAgentPipeline(
                        PipelineConfig(
                            roles=self.config.pipeline.roles,
                            comm_config=CommunicationConfig(
                                strategy=CommStrategy.FULL, budget=1.0)),
                        self.llm_fn),
                    samples)
            full_perf_per_group = {}
            for g, indices in groups.items():
                g_preds = [full_preds[i] for i in indices]
                g_samples = [samples[i] for i in indices]
                full_perf_per_group[g] = task_accuracy(task, g_preds, g_samples)
            task_results["full_perf_by_group"] = full_perf_per_group

            for baseline in baselines:
                comm_cfg = CommunicationConfig(strategy=baseline, budget=0.4)
                pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg),
                    self.llm_fn,
                )
                cheap_preds = _run_pipeline_on_samples(pipe, samples)

                group_metrics = {}
                for g, indices in groups.items():
                    fr = task_fail_rate_group(task, cheap_preds, samples, indices.tolist())
                    g_preds = [cheap_preds[i] for i in indices]
                    g_samples = [samples[i] for i in indices]
                    g_perf = task_accuracy(task, g_preds, g_samples)
                    delta = full_perf_per_group[g] - g_perf
                    group_metrics[g] = {
                        "fail_rate": fr,
                        "accuracy": g_perf,
                        "gap_to_full": delta,
                    }

                cheap_correct = task_correct_list(task, cheap_preds, samples)
                total_errors = sum(not c for c in cheap_correct)
                hard_idx = groups.get("hard", np.array([]))
                hard_errors = sum(
                    not cheap_correct[i] for i in hard_idx
                ) if len(hard_idx) > 0 else 0
                fcr = hard_errors / total_errors if total_errors > 0 else 0.0

                fr_hard = group_metrics.get("hard", {}).get("fail_rate", 0)
                fr_easy = group_metrics.get("easy", {}).get("fail_rate", 1e-6)
                hfa = hard_vs_easy_amplification(fr_hard, fr_easy)

                task_results[baseline.value] = {
                    "group_metrics": group_metrics,
                    "fcr_hard": fcr,
                    "hfa": hfa,
                }

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════

def run_experiment_a(config: ExperimentConfig, llm_fn: Callable = None,
                     output_dir: str = "results/exp_a") -> Dict[str, Any]:
    """Run all Experiment A sub-experiments with shared full-pipeline cache."""
    ensure_dir(output_dir)
    all_results = {}

    logger.info("=" * 60)
    logger.info("PRE-COMPUTE: Running full pipeline for all tasks (shared cache)")
    logger.info("=" * 60)
    full_cache = _precompute_full_results(config, llm_fn)

    logger.info("=" * 60)
    logger.info("EXPERIMENT A1: Cheap-vs-Full Gap Profiling")
    logger.info("=" * 60)
    a1 = ExpA1_CheapFullGap(config, llm_fn, full_cache=full_cache)
    a1_res = a1.run()
    save_json(a1_res, os.path.join(output_dir, "a1_cheap_full_gap.json"))
    all_results["a1"] = a1_res

    logger.info("=" * 60)
    logger.info("EXPERIMENT A2: Budget Sensitivity Sweep")
    logger.info("=" * 60)
    a2 = ExpA2_BudgetSensitivity(config, llm_fn, full_cache=full_cache)
    a2_res = a2.run()
    save_json(a2_res, os.path.join(output_dir, "a2_budget_sensitivity.json"))
    all_results["a2"] = a2_res

    logger.info("=" * 60)
    logger.info("EXPERIMENT A3: Difficulty-Stratified Failure Analysis")
    logger.info("=" * 60)
    a3 = ExpA3_DifficultyStratified(config, llm_fn, full_cache=full_cache)
    a3_res = a3.run()
    save_json(a3_res, os.path.join(output_dir, "a3_difficulty_stratified.json"))
    all_results["a3"] = a3_res

    return all_results

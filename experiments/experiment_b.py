"""
Experiment B: Failure Mechanism Analysis (分析失败机制)

B1 - Temporal Utility Distribution
B2 - Tail Dominance Test
B3 - Role Sensitivity Matrix
B4 - Sender-Receiver Utility Asymmetry
B5 - Receiver Brittleness Test
B6 - Communication Distribution Shift Test
B7 - Recoverability under Adaptation

Goal: Explain WHY cheap communication fails — recent-only assumption
is wrong, role-insensitive selection is unreasonable, and receivers
are brittle to compressed input distributions.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Callable, Dict, List

import numpy as np

from config import (
    AgentRole, CommStrategy, CommunicationConfig,
    DEFAULT_TASKS, ExperimentConfig, PipelineConfig, TaskType,
    get_max_tokens,
)
from src.pipeline import MultiAgentPipeline
from src.datasets.loader import get_loader, load_cached
from src.communication.adapter import ReceiverAdapter
from src.metrics.core import (
    teacher_consistency, adaptation_recovery_ratio,
)
from src.metrics.evaluator import task_accuracy, get_evaluator
from src.metrics.analysis import (
    temporal_utility_summary,
    compute_receiver_brittleness, compute_distribution_shift,
)
from src.utils.helpers import save_json, ensure_dir

logger = logging.getLogger(__name__)


def _set_task_max_tokens(llm_fn, task):
    if llm_fn is not None and hasattr(llm_fn, "set_max_tokens"):
        llm_fn.set_max_tokens(get_max_tokens(task))


# ════════════════════════════════════════════════
#  B1: Temporal Utility Distribution
# ════════════════════════════════════════════════

class ExpB1_TemporalUtility:
    """
    Leave-one-packet-out analysis to determine where high-utility
    packets are located temporally.

    U_p = Perf(full) - Perf(full \\ p)
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS:
            logger.info(f"B1: Analyzing temporal utility for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            full_cfg = CommunicationConfig(
                strategy=CommStrategy.FULL, budget=1.0)
            pipe = MultiAgentPipeline(
                PipelineConfig(
                    roles=self.config.pipeline.roles,
                    comm_config=full_cfg),
                self.llm_fn,
            )

            evaluator = get_evaluator(task)
            all_utility = []
            for sample in samples:
                def eval_fn(answer, _s=sample):
                    return 1.0 if evaluator.evaluate_single(answer, _s) else 0.0

                utility_data = pipe.run_with_leave_one_out(
                    sample.input, eval_fn)
                all_utility.extend(utility_data)

            summary = temporal_utility_summary(all_utility)

            position_bins = {"head": [], "mid": [], "tail": []}
            for item in all_utility:
                pos = item["position_ratio"]
                if pos < 1 / 3:
                    position_bins["head"].append(item["utility"])
                elif pos < 2 / 3:
                    position_bins["mid"].append(item["utility"])
                else:
                    position_bins["tail"].append(item["utility"])

            high_utility_positions = [
                item["position_ratio"]
                for item in all_utility
                if item["utility"] > np.percentile(
                    [x["utility"] for x in all_utility], 75)
            ]

            results[task.value] = {
                "segment_avg_utility": summary,
                "position_distribution_of_high_utility": {
                    "head_frac": sum(
                        1 for p in high_utility_positions if p < 1/3
                    ) / max(len(high_utility_positions), 1),
                    "mid_frac": sum(
                        1 for p in high_utility_positions if 1/3 <= p < 2/3
                    ) / max(len(high_utility_positions), 1),
                    "tail_frac": sum(
                        1 for p in high_utility_positions if p >= 2/3
                    ) / max(len(high_utility_positions), 1),
                },
                "n_packets_analyzed": len(all_utility),
            }

        return results


# ════════════════════════════════════════════════
#  B2: Tail Dominance Test
# ════════════════════════════════════════════════

class ExpB2_TailDominance:
    """
    Compare tail-only vs uniform vs oracle packet selection
    at the same budget.

    R_k^tail = P_k^oracle - P_k^tail
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None,
                 budget: float = 0.4):
        self.config = config
        self.llm_fn = llm_fn
        self.budget = budget

    def run(self) -> Dict[str, Any]:
        results = {}
        strategies_to_test = {
            "tail": CommStrategy.RECENCY,
            "uniform": CommStrategy.UNIFORM,
            "oracle": CommStrategy.ORACLE,
        }

        for task in DEFAULT_TASKS:
            logger.info(f"B2: Tail dominance test for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            for name, strat in strategies_to_test.items():
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

            oracle_perf = task_results["oracle"]["accuracy"]
            tail_perf = task_results["tail"]["accuracy"]
            task_results["tail_regret"] = oracle_perf - tail_perf
            task_results["uniform_regret"] = (
                oracle_perf - task_results["uniform"]["accuracy"])

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  B3: Role Sensitivity Matrix
# ════════════════════════════════════════════════

class ExpB3_RoleSensitivity:
    """
    Compress one role at a time, measure Δ_r^role.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn
        self.compress_roles = [
            AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.REFINER,
        ]

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS:
            logger.info(f"B3: Role sensitivity for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            # Full performance reference
            full_cfg = CommunicationConfig(
                strategy=CommStrategy.FULL, budget=1.0)
            full_pipe = MultiAgentPipeline(
                PipelineConfig(
                    roles=self.config.pipeline.roles,
                    comm_config=full_cfg),
                self.llm_fn,
            )
            full_preds = [r.final_answer for r in full_pipe.run_batch([s.input for s in samples])]
            full_perf = task_accuracy(task, full_preds, samples)

            sensitivity = {}
            for role in self.compress_roles:
                comp_cfg = CommunicationConfig(
                    strategy=CommStrategy.RECENCY, budget=0.4)
                comp_pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comp_cfg),
                    self.llm_fn,
                )
                comp_preds = []
                for s in samples:
                    result = comp_pipe.run_compress_single_role(
                        s.input, role)
                    comp_preds.append(result.final_answer)
                comp_perf = task_accuracy(task, comp_preds, samples)
                sensitivity[role.value] = {
                    "accuracy": comp_perf,
                    "delta_role": full_perf - comp_perf,
                }

            results[task.value] = {
                "full_accuracy": full_perf,
                "role_sensitivity": sensitivity,
                "sensitivity_vector": [
                    sensitivity[r.value]["delta_role"]
                    for r in self.compress_roles
                ],
            }

        return results


# ════════════════════════════════════════════════
#  B4: Sender-Receiver Utility Asymmetry
# ════════════════════════════════════════════════

class ExpB4_SenderReceiverUtility:
    """
    Compute U_p^{i→j} for each packet and receiver pair.
    Measure variance across receivers.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn

    def run(self) -> Dict[str, Any]:
        results = {}

        for task in DEFAULT_TASKS[:2]:  # expensive, limit tasks
            logger.info(f"B4: Sender-receiver utility for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=min(
                self.config.num_samples or 50, 50))
            evaluator = get_evaluator(task)

            full_cfg = CommunicationConfig(
                strategy=CommStrategy.FULL, budget=1.0)
            pipe = MultiAgentPipeline(
                PipelineConfig(
                    roles=self.config.pipeline.roles,
                    comm_config=full_cfg),
                self.llm_fn,
            )

            all_variances = []
            batch_results = pipe.run_batch([s.input for s in samples])
            for sample, result in zip(samples, batch_results):
                packets = result.all_packets

                for i, pkt in enumerate(packets):
                    remaining = packets[:i] + packets[i + 1:]

                    receiver_utilities = {}
                    for recv_role in [AgentRole.CRITIC, AgentRole.REFINER,
                                      AgentRole.JUDGER]:
                        if recv_role == pkt.agent_role:
                            continue
                        agent = pipe.agents.get(recv_role)
                        if agent is None:
                            continue
                        agent.reset()
                        answer_full = agent.run(
                            sample.input, packets, self.llm_fn)
                        agent.reset()
                        answer_without = agent.run(
                            sample.input, remaining, self.llm_fn)

                        score_full = 1.0 if evaluator.evaluate_single(answer_full, sample) else 0.0
                        score_without = 1.0 if evaluator.evaluate_single(answer_without, sample) else 0.0
                        receiver_utilities[recv_role.value] = score_full - score_without

                    if receiver_utilities:
                        vals = list(receiver_utilities.values())
                        all_variances.append(float(np.var(vals)))

            results[task.value] = {
                "mean_receiver_variance": float(np.mean(all_variances)) if all_variances else 0.0,
                "median_receiver_variance": float(np.median(all_variances)) if all_variances else 0.0,
                "n_packets_analyzed": len(all_variances),
            }

        return results


# ════════════════════════════════════════════════
#  B5: Receiver Brittleness Test
# ════════════════════════════════════════════════

class ExpB5_ReceiverBrittleness:
    """
    Test judger accuracy under different compression strategies
    without adaptation.

    B^recv = Perf(full) - Perf(compressed)
    C^teacher = consistency with full output
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn

    def run(self) -> Dict[str, Any]:
        results = {}
        strategies = [
            ("full", CommStrategy.FULL, 1.0),
            ("recency", CommStrategy.RECENCY, 0.4),
            ("role_aware", CommStrategy.ROLE_AWARE, 0.4),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"B5: Receiver brittleness for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            task_results = {}
            full_preds = None

            for name, strat, budget in strategies:
                comm_cfg = CommunicationConfig(strategy=strat, budget=budget)
                pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg),
                    self.llm_fn,
                )
                preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                perf = task_accuracy(task, preds, samples)

                entry = {"accuracy": perf}
                if name == "full":
                    full_preds = preds
                elif full_preds is not None:
                    entry["brittleness"] = compute_receiver_brittleness(
                        task_results["full"]["accuracy"],
                        perf,
                        preds, full_preds,
                    )

                task_results[name] = entry

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  B6: Communication Distribution Shift Test
# ════════════════════════════════════════════════

class ExpB6_DistributionShift:
    """
    Compare embedding distributions between full and compressed
    communications fed to the judger.

    Uses real text embeddings (sentence-transformers or TF-IDF)
    computed from the actual communication content that each
    strategy delivers to the receiver.
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn

    def run(self) -> Dict[str, Any]:
        from src.utils.embeddings import get_embeddings

        results = {}

        for task in DEFAULT_TASKS[:3]:
            logger.info(f"B6: Distribution shift for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            strategies = {
                "full": (CommStrategy.FULL, 1.0),
                "recency": (CommStrategy.RECENCY, 0.4),
                "role_aware": (CommStrategy.ROLE_AWARE, 0.4),
            }

            comm_texts: Dict[str, List[str]] = {}
            for name, (strat, budget) in strategies.items():
                comm_cfg = CommunicationConfig(strategy=strat, budget=budget)
                pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg),
                    self.llm_fn,
                )
                batch_res = pipe.run_batch([s.input for s in samples])
                texts = [
                    "\n".join(p.content for p in r.all_packets)
                    for r in batch_res
                ]
                comm_texts[name] = texts

            embeddings = {}
            for name, texts in comm_texts.items():
                embeddings[name] = get_embeddings(texts)

            full_emb = embeddings["full"]
            task_results = {}
            for name in ["recency", "role_aware"]:
                shift = compute_distribution_shift(
                    full_emb, embeddings[name], use_mmd=True)
                task_results[name] = shift

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  B7: Recoverability under Adaptation
# ════════════════════════════════════════════════

class ExpB7_Recoverability:
    """
    Test whether receiver adaptation can recover performance
    lost to communication compression.

    R^adapt = (P_adapt - P_cmp) / (P_full - P_cmp + ε)
    """

    def __init__(self, config: ExperimentConfig, llm_fn: Callable = None):
        self.config = config
        self.llm_fn = llm_fn

    def run(self) -> Dict[str, Any]:
        results = {}
        test_configs = [
            ("recency", CommStrategy.RECENCY, False),
            ("recency+adapter", CommStrategy.RECENCY, True),
            ("role_aware", CommStrategy.ROLE_AWARE, False),
            ("role_aware+adapter", CommStrategy.ROLE_AWARE, True),
        ]

        for task in DEFAULT_TASKS:
            logger.info(f"B7: Recoverability for {task.value}")
            _set_task_max_tokens(self.llm_fn, task)
            samples = load_cached(task, max_samples=self.config.num_samples)

            # Full reference
            full_cfg = CommunicationConfig(
                strategy=CommStrategy.FULL, budget=1.0)
            full_pipe = MultiAgentPipeline(
                PipelineConfig(
                    roles=self.config.pipeline.roles,
                    comm_config=full_cfg),
                self.llm_fn,
            )
            full_preds = [r.final_answer for r in full_pipe.run_batch([s.input for s in samples])]
            full_perf = task_accuracy(task, full_preds, samples)

            task_results = {"full_accuracy": full_perf}

            for name, strat, use_adapter in test_configs:
                comm_cfg = CommunicationConfig(
                    strategy=strat, budget=0.4, use_adapter=use_adapter)
                adapter = ReceiverAdapter() if use_adapter else None
                pipe = MultiAgentPipeline(
                    PipelineConfig(
                        roles=self.config.pipeline.roles,
                        comm_config=comm_cfg),
                    self.llm_fn,
                    adapter=adapter,
                )
                preds = [r.final_answer for r in pipe.run_batch([s.input for s in samples])]
                perf = task_accuracy(task, preds, samples)
                consist = teacher_consistency(preds, full_preds)

                entry = {
                    "accuracy": perf,
                    "teacher_consistency": consist,
                }

                # Recovery ratio for adapter variants
                base_name = name.replace("+adapter", "")
                if "+adapter" in name and base_name in task_results:
                    base_perf = task_results[base_name]["accuracy"]
                    entry["recovery_ratio"] = adaptation_recovery_ratio(
                        perf, base_perf, full_perf)

                task_results[name] = entry

            results[task.value] = task_results

        return results


# ════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════

def run_experiment_b(config: ExperimentConfig, llm_fn: Callable = None,
                     output_dir: str = "results/exp_b") -> Dict[str, Any]:
    """Run all Experiment B sub-experiments."""
    ensure_dir(output_dir)
    all_results = {}

    experiments = [
        ("b1", "Temporal Utility Distribution", ExpB1_TemporalUtility),
        ("b2", "Tail Dominance Test", ExpB2_TailDominance),
        ("b3", "Role Sensitivity Matrix", ExpB3_RoleSensitivity),
        ("b4", "Sender-Receiver Utility Asymmetry", ExpB4_SenderReceiverUtility),
        ("b5", "Receiver Brittleness Test", ExpB5_ReceiverBrittleness),
        ("b6", "Distribution Shift Test", ExpB6_DistributionShift),
        ("b7", "Recoverability under Adaptation", ExpB7_Recoverability),
    ]

    for key, name, cls in experiments:
        logger.info("=" * 60)
        logger.info(f"EXPERIMENT {key.upper()}: {name}")
        logger.info("=" * 60)
        exp = cls(config, llm_fn)
        res = exp.run()
        save_json(res, os.path.join(output_dir, f"{key}_{name.lower().replace(' ', '_')}.json"))
        all_results[key] = res

    return all_results

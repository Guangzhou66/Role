"""
Main experiment runner.

Usage:
    python scripts/run_experiments.py --exp all
    python scripts/run_experiments.py --exp a
    python scripts/run_experiments.py --exp b1 b3 b5
    python scripts/run_experiments.py --exp c --budget 0.4 --samples 100
    python scripts/run_experiments.py --visualize
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ExperimentConfig, ModelConfig, PipelineConfig,
    CommunicationConfig, CommStrategy, AgentRole,
    BUDGET_SWEEP, DEFAULT_TASKS,
)
from src.pipeline import create_llm_fn, _mock_llm_fn
from src.utils.helpers import setup_logging, save_json, ensure_dir

from experiments.experiment_a import run_experiment_a
from experiments.experiment_b import run_experiment_b
from experiments.experiment_c import run_experiment_c
from experiments.experiment_d import run_experiment_d


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-agent communication compression experiments"
    )
    parser.add_argument(
        "--exp", nargs="+", default=["all"],
        help="Which experiments to run: all, a, b, c, d, or specific (a1, b3, ...)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model name for LLM calls"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL for compatible endpoints"
    )
    parser.add_argument(
        "--local-model", type=str, default=None,
        help="Path to local model (uses transformers instead of API)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM (for testing pipeline without API)"
    )
    parser.add_argument(
        "--budget", type=float, default=0.4,
        help="Default communication budget"
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Max samples per task (None = all)"
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of random seeds for reliability tests"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate figures from existing results"
    )
    parser.add_argument(
        "--agents", type=int, default=4,
        choices=[2, 3, 4],
        help="Number of agents in pipeline"
    )
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    model_cfg = ModelConfig(
        model_name=args.model,
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
        api_base=args.api_base,
        use_local=args.local_model is not None,
        local_model_path=args.local_model,
    )

    role_map = {
        2: [AgentRole.PLANNER, AgentRole.JUDGER],
        3: [AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.JUDGER],
        4: [AgentRole.PLANNER, AgentRole.CRITIC,
            AgentRole.REFINER, AgentRole.JUDGER],
    }

    comm_cfg = CommunicationConfig(
        strategy=CommStrategy.FULL,
        budget=args.budget,
    )
    pipe_cfg = PipelineConfig(
        roles=role_map[args.agents],
        comm_config=comm_cfg,
    )

    return ExperimentConfig(
        model=model_cfg,
        pipeline=pipe_cfg,
        num_seeds=args.seeds,
        budgets=BUDGET_SWEEP,
        output_dir=args.output_dir,
        num_samples=args.samples,
    )


def main():
    args = parse_args()
    logger = setup_logging("INFO")

    if args.visualize:
        from scripts.visualize import generate_all_figures
        generate_all_figures(args.output_dir, "figures")
        return

    config = build_config(args)

    if args.mock:
        llm_fn = _mock_llm_fn
        logger.info("Using MOCK LLM (no API calls)")
    else:
        llm_fn = create_llm_fn(vars(config.model))

    exps = set(args.exp)
    run_all = "all" in exps

    ensure_dir(config.output_dir)
    start_total = time.time()

    if run_all or "a" in exps or any(e.startswith("a") for e in exps):
        logger.info("\n" + "=" * 70)
        logger.info("  EXPERIMENT GROUP A: Failure Observation")
        logger.info("=" * 70)
        start = time.time()
        results_a = run_experiment_a(
            config, llm_fn,
            os.path.join(config.output_dir, "exp_a"))
        logger.info(f"Experiment A completed in {time.time() - start:.1f}s")

    if run_all or "b" in exps or any(e.startswith("b") for e in exps):
        logger.info("\n" + "=" * 70)
        logger.info("  EXPERIMENT GROUP B: Failure Mechanism Analysis")
        logger.info("=" * 70)
        start = time.time()
        results_b = run_experiment_b(
            config, llm_fn,
            os.path.join(config.output_dir, "exp_b"))
        logger.info(f"Experiment B completed in {time.time() - start:.1f}s")

    if run_all or "c" in exps or any(e.startswith("c") for e in exps):
        logger.info("\n" + "=" * 70)
        logger.info("  EXPERIMENT GROUP C: Method Validation")
        logger.info("=" * 70)
        start = time.time()
        results_c = run_experiment_c(
            config, llm_fn,
            os.path.join(config.output_dir, "exp_c"))
        logger.info(f"Experiment C completed in {time.time() - start:.1f}s")

    if run_all or "d" in exps or any(e.startswith("d") for e in exps):
        logger.info("\n" + "=" * 70)
        logger.info("  EXPERIMENT GROUP D: Scaling / Generalization")
        logger.info("=" * 70)
        start = time.time()
        results_d = run_experiment_d(
            config, llm_fn,
            os.path.join(config.output_dir, "exp_d"))
        logger.info(f"Experiment D completed in {time.time() - start:.1f}s")

    total_time = time.time() - start_total
    logger.info(f"\nAll experiments completed in {total_time:.1f}s")
    logger.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

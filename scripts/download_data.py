"""
Download and validate all datasets before running experiments.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --task gsm8k math
    python scripts/download_data.py --check-only
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TaskType, DEFAULT_TASKS
from src.datasets.loader import get_loader
from src.utils.helpers import setup_logging


TASK_NAME_MAP = {
    "gsm8k": TaskType.GSM8K,
    "math": TaskType.MATH,
    "mbpp": TaskType.MBPP_PLUS,
    "mbpp_plus": TaskType.MBPP_PLUS,
    "humaneval": TaskType.HUMANEVAL_PLUS,
    "humaneval_plus": TaskType.HUMANEVAL_PLUS,
    "medqa": TaskType.MEDQA,
}


def download_task(task: TaskType, max_samples: int = None) -> dict:
    """Download one dataset and return summary info."""
    start = time.time()
    loader = get_loader(task, max_samples=max_samples)
    try:
        samples = loader.load()
        elapsed = time.time() - start
        first = samples[0] if samples else None
        return {
            "task": task.value,
            "status": "OK",
            "n_samples": len(samples),
            "time_sec": round(elapsed, 1),
            "sample_input_preview": first.input[:120] + "..." if first else "",
            "sample_reference_preview": first.reference[:80] if first else "",
        }
    except Exception as e:
        return {
            "task": task.value,
            "status": f"FAILED: {e}",
            "n_samples": 0,
            "time_sec": round(time.time() - start, 1),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Download and validate experiment datasets")
    parser.add_argument("--task", nargs="+", default=None,
                        help="Specific tasks to download (default: all)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check, don't print previews")
    parser.add_argument("--samples", type=int, default=5,
                        help="Max samples to load for validation")
    args = parser.parse_args()

    logger = setup_logging("INFO")

    if args.task:
        tasks = []
        for name in args.task:
            t = TASK_NAME_MAP.get(name.lower())
            if t is None:
                print(f"Unknown task: {name}. Available: {list(TASK_NAME_MAP.keys())}")
                sys.exit(1)
            tasks.append(t)
    else:
        tasks = DEFAULT_TASKS

    print("=" * 70)
    print("  Dataset Download & Validation")
    print("=" * 70)
    print()

    all_ok = True
    for task in tasks:
        print(f"[{task.value}] Downloading...", end=" ", flush=True)
        info = download_task(task, max_samples=args.samples)

        if info["status"] == "OK":
            print(f"OK ({info['n_samples']} samples, {info['time_sec']}s)")
            if not args.check_only:
                print(f"  Input:     {info['sample_input_preview']}")
                print(f"  Reference: {info['sample_reference_preview']}")
        else:
            print(f"FAILED")
            print(f"  Error: {info['status']}")
            all_ok = False
        print()

    print("=" * 70)
    if all_ok:
        print("All datasets loaded successfully!")
    else:
        print("Some datasets failed. Check logs above.")
        print("Ensure 'datasets' package is installed: pip install datasets")
    print("=" * 70)


if __name__ == "__main__":
    main()

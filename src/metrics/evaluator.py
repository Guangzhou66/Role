"""
Task-aware evaluator that dispatches the correct comparison method
for each dataset type.

Replaces the naive `accuracy(predictions, references)` used throughout
the experiments with evaluation methods matched to each dataset's
actual answer format.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from config import TaskType
from src.datasets.loader import Sample
from src.utils.answer_utils import (
    match_gsm8k, match_math, match_medqa,
    extract_gsm8k_answer, extract_math_answer,
    extract_code, extract_humaneval_completion,
    extract_medqa_option,
)
from src.utils.code_executor import evaluate_mbpp, evaluate_humaneval

logger = logging.getLogger(__name__)


class TaskEvaluator:
    """
    Evaluates LLM predictions against ground truth using the
    appropriate method for each task type.
    """

    def __init__(self, task: TaskType, code_timeout: int = 10):
        self.task = task
        self.code_timeout = code_timeout
        self._dispatch = {
            TaskType.GSM8K: self._eval_gsm8k,
            TaskType.MATH: self._eval_math,
            TaskType.MBPP_PLUS: self._eval_mbpp,
            TaskType.HUMANEVAL_PLUS: self._eval_humaneval,
            TaskType.MEDQA: self._eval_medqa,
        }

    # ── public API ───────────────────────────────

    def evaluate_single(self, prediction: str, sample: Sample) -> bool:
        """Check if a single prediction is correct."""
        fn = self._dispatch[self.task]
        return fn(prediction, sample)

    def evaluate_batch(
        self, predictions: List[str], samples: List[Sample],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions.
        Returns dict with accuracy and per-sample results.
        """
        assert len(predictions) == len(samples)
        correct = []
        for pred, sample in zip(predictions, samples):
            c = self.evaluate_single(pred, sample)
            correct.append(c)

        n = len(predictions)
        acc = sum(correct) / n if n > 0 else 0.0
        return {
            "accuracy": acc,
            "n_correct": sum(correct),
            "n_total": n,
            "per_sample": correct,
        }

    def accuracy(self, predictions: List[str],
                 samples: List[Sample]) -> float:
        """Shorthand: return just the accuracy float."""
        return self.evaluate_batch(predictions, samples)["accuracy"]

    # ── per-task evaluation ──────────────────────

    def _eval_gsm8k(self, prediction: str, sample: Sample) -> bool:
        return match_gsm8k(prediction, sample.reference)

    def _eval_math(self, prediction: str, sample: Sample) -> bool:
        return match_math(prediction, sample.reference)

    def _eval_mbpp(self, prediction: str, sample: Sample) -> bool:
        code = extract_code(prediction)
        test_list = sample.metadata.get("test_list", [])
        test_imports = sample.metadata.get("test_imports", "")
        if not test_list:
            logger.warning("MBPP sample has no test_list, falling back to string match")
            return code.strip() == sample.reference.strip()
        result = evaluate_mbpp(code, test_list, test_imports,
                               timeout=self.code_timeout)
        return result["passed"]

    def _eval_humaneval(self, prediction: str, sample: Sample) -> bool:
        prompt = sample.input
        code = extract_humaneval_completion(prediction, prompt)
        test_code = sample.metadata.get("test", "")
        entry_point = sample.metadata.get("entry_point", "")
        if not test_code or not entry_point:
            logger.warning("HumanEval sample missing test/entry_point")
            return code.strip() == sample.reference.strip()
        result = evaluate_humaneval(
            code, prompt, test_code, entry_point,
            timeout=self.code_timeout,
        )
        return result["passed"]

    def _eval_medqa(self, prediction: str, sample: Sample) -> bool:
        return match_medqa(prediction, sample.reference)


def get_evaluator(task: TaskType, **kwargs) -> TaskEvaluator:
    """Factory for task evaluators."""
    return TaskEvaluator(task, **kwargs)


# ── convenience functions for experiment code ────

def task_accuracy(task: TaskType, predictions: List[str],
                  samples: List[Sample]) -> float:
    """One-liner: compute task-aware accuracy."""
    return get_evaluator(task).accuracy(predictions, samples)


def task_correct_list(task: TaskType, predictions: List[str],
                      samples: List[Sample]) -> List[bool]:
    """Per-sample correctness list (for difficulty scoring etc.)."""
    ev = get_evaluator(task)
    return [ev.evaluate_single(p, s) for p, s in zip(predictions, samples)]


def task_fail_rate_group(task: TaskType, predictions: List[str],
                         samples: List[Sample],
                         indices: List[int]) -> float:
    """Failure rate in a subset of indices, using task-aware matching."""
    if not indices:
        return 0.0
    ev = get_evaluator(task)
    errors = sum(
        not ev.evaluate_single(predictions[i], samples[i]) for i in indices
    )
    return errors / len(indices)

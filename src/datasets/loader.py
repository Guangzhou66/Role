"""
Dataset loaders for all evaluation tasks.

Supports: GSM8K, MATH/AIME, MBPP+, HumanEval+, MedQA.
Each loader downloads real data from HuggingFace Hub and returns a
unified format:
  [Sample(input=str, reference=str, metadata=dict), ...]

If HuggingFace loading fails, falls back to local JSONL files under
data_dir.  An explicit error is raised when *no* data can be found.
"""
from __future__ import annotations
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config import TaskType

logger = logging.getLogger(__name__)

_dataset_cache: dict = {}


class Sample:
    """A single evaluation sample."""
    __slots__ = ("input", "reference", "metadata")

    def __init__(self, input: str, reference: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.input = input
        self.reference = reference
        self.metadata = metadata or {}

    def to_dict(self):
        return {"input": self.input, "reference": self.reference,
                "metadata": self.metadata}


class DatasetLoader(ABC):
    """Base dataset loader."""

    def __init__(self, data_dir: str = "data", split: str = "test",
                 max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples

    @abstractmethod
    def load(self) -> List[Sample]:
        """Load and return samples."""

    def _maybe_truncate(self, samples: List[Sample]) -> List[Sample]:
        if self.max_samples and len(samples) > self.max_samples:
            return samples[:self.max_samples]
        return samples

    def _validate(self, samples: List[Sample], name: str) -> List[Sample]:
        if not samples:
            raise RuntimeError(
                f"[{name}] Loaded 0 samples. Check network connection or "
                f"place data files under '{self.data_dir}/{name.lower()}/{self.split}.jsonl'."
            )
        logger.info(f"[{name}] Loaded {len(samples)} samples (split={self.split})")
        return samples

    @staticmethod
    def _hf_load(repo: str, name: Optional[str] = None,
                 split: str = "test", **kwargs):
        """Thin wrapper around datasets.load_dataset with logging."""
        import os as _os
        if _os.environ.get("HF_DATASETS_OFFLINE", "0") == "1":
            raise RuntimeError("HF_DATASETS_OFFLINE=1, skipping network request")
        from datasets import load_dataset
        args = [repo]
        if name:
            args.append(name)
        logger.info(f"Downloading from HuggingFace: {repo} (config={name}, split={split})")
        return load_dataset(*args, split=split, **kwargs)

    def _load_local_jsonl(self, subdir: str) -> List[Dict]:
        path = os.path.join(self.data_dir, subdir, f"{self.split}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Local file not found: {path}")
            return []
        logger.info(f"Loading local data from {path}")
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def _load_local_samples(self, subdir: str) -> Optional[List["Sample"]]:
        """Load local JSONL and return Sample list if file uses {input, reference} format."""
        raw = self._load_local_jsonl(subdir)
        if not raw:
            return None
        # Detect pre-processed format
        if "input" in raw[0] and "reference" in raw[0]:
            samples = [
                Sample(
                    input=item["input"],
                    reference=item["reference"],
                    metadata=item.get("metadata", {}),
                )
                for item in raw
            ]
            return samples
        return None  # raw format, let caller handle field mapping


# ════════════════════════════════════════════════════════
#  GSM8K — Grade-School Math 8K
#  HuggingFace: openai/gsm8k  (config="main", split="test")
#  1319 test samples
# ════════════════════════════════════════════════════════

class GSM8KLoader(DatasetLoader):
    """GSM8K: grade-school math word problems."""

    def load(self) -> List[Sample]:
        ds = None

        # --- HuggingFace ---
        try:
            ds = self._hf_load("openai/gsm8k", "main", split=self.split)
        except Exception as e:
            logger.warning(f"HF gsm8k load failed: {e}")

        # --- Local fallback ---
        if ds is None or len(ds) == 0:
            local = self._load_local_samples("gsm8k")
            if local is not None:
                return self._maybe_truncate(self._validate(local, "GSM8K"))
            ds = self._load_local_jsonl("gsm8k")

        samples = []
        for item in ds:
            answer = self._extract_answer(item.get("answer", ""))
            samples.append(Sample(
                input=item["question"],
                reference=answer,
                metadata={
                    "full_answer": item.get("answer", ""),
                    "n_steps": item.get("answer", "").count("\n"),
                },
            ))
        return self._maybe_truncate(self._validate(samples, "GSM8K"))

    @staticmethod
    def _extract_answer(answer_str: str) -> str:
        match = re.search(r"####\s*(.+)", answer_str)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_str.strip().split("\n")[-1].strip()


# ════════════════════════════════════════════════════════
#  MATH — Competition-level math problems
#  Primary: hendrycks/competition_math  (may be DMCA'd)
#  Fallbacks: qwedsacf/competition_math, lighteval/MATH
# ════════════════════════════════════════════════════════

class MATHLoader(DatasetLoader):
    """MATH dataset (including AIME subset)."""

    # (repo, config, possible_splits)
    HF_REPOS = [
        ("hendrycks/competition_math", None, ["test", "train"]),
        ("qwedsacf/competition_math", None, ["test", "train"]),
        ("DigitalLearningGmbH/MATH-lighteval", None, ["test", "train"]),
        ("EleutherAI/hendrycks_math", None, ["test", "train"]),
    ]

    def __init__(self, subset: str = "all", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset

    def load(self) -> List[Sample]:
        ds = None

        for repo, config_name, splits_to_try in self.HF_REPOS:
            for split in [self.split] + [s for s in splits_to_try if s != self.split]:
                try:
                    ds = self._hf_load(repo, config_name, split=split)
                    if ds is not None and len(ds) > 0:
                        logger.info(f"MATH loaded from {repo} (split={split}): {len(ds)} samples")
                        break
                except Exception as e:
                    logger.warning(f"MATH from {repo} (split={split}) failed: {e}")
                    ds = None
            if ds is not None and len(ds) > 0:
                break

        if ds is None or len(ds) == 0:
            local = self._load_local_samples("math")
            if local is not None:
                return self._maybe_truncate(self._validate(local, "MATH"))
            ds = self._load_local_jsonl("math")

        samples = []
        for item in ds:
            item_type = item.get("type", item.get("subject", ""))
            if self.subset != "all" and item_type != self.subset:
                continue
            solution = item.get("solution", item.get("answer", ""))
            answer = self._extract_answer(solution)
            samples.append(Sample(
                input=item.get("problem", item.get("question", "")),
                reference=answer,
                metadata={
                    "level": item.get("level", ""),
                    "type": item_type,
                    "full_solution": solution,
                },
            ))
        return self._maybe_truncate(self._validate(samples, "MATH"))

    @staticmethod
    def _extract_answer(solution: str) -> str:
        match = re.search(r"\\boxed\{(.+)\}", solution)
        if match:
            inner = match.group(1).strip()
            # handle nested braces
            depth = 0
            end = 0
            for i, ch in enumerate(inner):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    if depth == 0:
                        end = i
                        break
                    depth -= 1
            if end > 0:
                return inner[:end].strip()
            return inner
        return solution.strip().split("\n")[-1].strip()


# ════════════════════════════════════════════════════════
#  MBPP — Mostly Basic Python Programming
#  HuggingFace: google-research-datasets/mbpp
#  config="sanitized" → 427 test,  config="full" → 974
# ════════════════════════════════════════════════════════

class MBPPPlusLoader(DatasetLoader):
    """MBPP+ code generation dataset (sanitized subset)."""

    def load(self) -> List[Sample]:
        ds = None

        try:
            ds = self._hf_load("google-research-datasets/mbpp",
                               "sanitized", split=self.split)
        except Exception as e1:
            logger.warning(f"MBPP sanitized failed: {e1}")
            try:
                ds = self._hf_load("google-research-datasets/mbpp",
                                   "full", split=self.split)
            except Exception as e2:
                logger.warning(f"MBPP full failed: {e2}")

        if ds is None or len(ds) == 0:
            local = self._load_local_samples("mbpp")
            if local is not None:
                return self._maybe_truncate(self._validate(local, "MBPP"))
            ds = self._load_local_jsonl("mbpp")

        samples = []
        for item in ds:
            prompt = item.get("prompt", item.get("text", ""))
            code = item.get("code", item.get("solution", ""))
            test_list = item.get("test_list", [])
            samples.append(Sample(
                input=prompt,
                reference=code,
                metadata={
                    "task_id": item.get("task_id", item.get("source_file", "")),
                    "test_list": test_list,
                    "test_imports": item.get("test_imports", ""),
                },
            ))
        return self._maybe_truncate(self._validate(samples, "MBPP"))


# ════════════════════════════════════════════════════════
#  HumanEval — OpenAI code completion benchmark
#  HuggingFace: openai/openai_humaneval  (split="test", 164 problems)
# ════════════════════════════════════════════════════════

class HumanEvalPlusLoader(DatasetLoader):
    """HumanEval+ code generation dataset."""

    def load(self) -> List[Sample]:
        ds = None

        try:
            ds = self._hf_load("openai/openai_humaneval", split=self.split)
        except Exception as e1:
            logger.warning(f"HumanEval from openai/openai_humaneval failed: {e1}")
            try:
                ds = self._hf_load("openai_humaneval", split=self.split)
            except Exception as e2:
                logger.warning(f"HumanEval from openai_humaneval failed: {e2}")

        if ds is None or len(ds) == 0:
            local = self._load_local_samples("humaneval")
            if local is not None:
                return self._maybe_truncate(self._validate(local, "HumanEval"))
            ds = self._load_local_jsonl("humaneval")

        samples = []
        for item in ds:
            samples.append(Sample(
                input=item.get("prompt", ""),
                reference=item.get("canonical_solution", ""),
                metadata={
                    "task_id": item.get("task_id", ""),
                    "entry_point": item.get("entry_point", ""),
                    "test": item.get("test", ""),
                },
            ))
        return self._maybe_truncate(self._validate(samples, "HumanEval"))


# ════════════════════════════════════════════════════════
#  MedQA — Medical question answering (USMLE 4-option)
#  Primary:  GBaker/MedQA-USMLE-4-options
#  Fallback: bigbio/med_qa
# ════════════════════════════════════════════════════════

class MedQALoader(DatasetLoader):
    """MedQA medical question answering dataset."""

    def load(self) -> List[Sample]:
        ds = None
        source = None

        # Try GBaker version (clean 4-option USMLE format)
        try:
            ds = self._hf_load("GBaker/MedQA-USMLE-4-options", split=self.split)
            source = "GBaker"
        except Exception as e1:
            logger.warning(f"MedQA GBaker failed: {e1}")
            try:
                ds = self._hf_load("bigbio/med_qa", split=self.split)
                source = "bigbio"
            except Exception as e2:
                logger.warning(f"MedQA bigbio failed: {e2}")

        if ds is None or len(ds) == 0:
            local = self._load_local_samples("medqa")
            if local is not None:
                return self._maybe_truncate(self._validate(local, "MedQA"))
            ds = self._load_local_jsonl("medqa")
            source = "local"

        samples = []
        for item in ds:
            question, options_str, options_dict, answer_idx, answer_text = \
                self._parse_item(item, source)
            full_input = f"{question}\n\nOptions:\n{options_str}"
            samples.append(Sample(
                input=full_input,
                reference=answer_idx,
                metadata={
                    "question": question,
                    "options": options_dict,
                    "answer_text": answer_text,
                    "source": source or "unknown",
                },
            ))
        return self._maybe_truncate(self._validate(samples, "MedQA"))

    @staticmethod
    def _parse_item(item: Dict, source: Optional[str]) -> tuple:
        """
        Parse a single MedQA item regardless of source format.
        Returns: (question, options_str, options_dict, answer_idx, answer_text)
        """
        question = item.get("question", item.get("sent1", ""))

        # Build options dict {letter: text}
        raw_options = item.get("options", item.get("choices",
                               item.get("ending0", None)))
        options_dict = {}
        if isinstance(raw_options, dict):
            options_dict = dict(raw_options)
        elif isinstance(raw_options, list):
            options_dict = {chr(65 + i): o for i, o in enumerate(raw_options)}
        elif raw_options is None:
            endings = []
            for i in range(10):
                e = item.get(f"ending{i}")
                if e is not None:
                    endings.append(e)
            options_dict = {chr(65 + i): e for i, e in enumerate(endings)}

        options_str = "\n".join(f"{k}: {v}" for k, v in options_dict.items())

        # Prefer answer_idx (letter); fall back to answer text
        answer_idx = item.get("answer_idx", "")
        answer_text = item.get("answer", "")

        if not answer_idx and answer_text:
            # Try to find which option matches the answer text
            for k, v in options_dict.items():
                if v.strip().lower() == answer_text.strip().lower():
                    answer_idx = k
                    break
            if not answer_idx:
                answer_idx = item.get("label", answer_text)

        if not answer_idx:
            answer_idx = answer_text

        return question, options_str, options_dict, str(answer_idx), str(answer_text)


# ════════════════════════════════════════════════════════
#  Factory
# ════════════════════════════════════════════════════════

def get_loader(task: TaskType, **kwargs) -> DatasetLoader:
    """Factory for dataset loaders."""
    mapping = {
        TaskType.GSM8K: GSM8KLoader,
        TaskType.MATH: MATHLoader,
        TaskType.MBPP_PLUS: MBPPPlusLoader,
        TaskType.HUMANEVAL_PLUS: HumanEvalPlusLoader,
        TaskType.MEDQA: MedQALoader,
    }
    cls = mapping.get(task)
    if cls is None:
        raise ValueError(f"Unknown task: {task}")
    return cls(**kwargs)


def load_cached(task: TaskType, **kwargs) -> List[Sample]:
    """Load dataset with module-level caching to avoid redundant parsing."""
    cache_key = (task, kwargs.get("max_samples"), kwargs.get("split", "test"))
    if cache_key in _dataset_cache:
        logger.info(f"[Cache hit] {task.value} (max_samples={kwargs.get('max_samples')})")
        return _dataset_cache[cache_key]
    loader = get_loader(task, **kwargs)
    samples = loader.load()
    _dataset_cache[cache_key] = samples
    return samples

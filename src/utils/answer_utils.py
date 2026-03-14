"""
Answer extraction and normalization for each task type.

Handles the critical gap between raw LLM output (free-form text)
and the ground-truth reference format per dataset.

Task-specific logic:
  GSM8K:        extract last number
  MATH:         extract \\boxed{...} or last math expression, then normalize
  MBPP/HumanEval: extract code block
  MedQA:        extract option letter (A/B/C/D)
"""
from __future__ import annotations
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  GSM8K: numeric answer extraction
# ──────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> str:
    """
    Extract numeric answer from GSM8K-style LLM response.
    Looks for patterns like:
      "#### 42", "the answer is 42", "= 42", or last number in text.
    """
    text = text.strip()

    # Pattern 1: #### delimiter (GSM8K standard)
    m = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 2: "the answer is X" / "answer: X"
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?([+-]?\d[\d,]*\.?\d*)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 3: "= X" at end of a line
    m = re.search(r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\s*\$?\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 4: last number in the entire text
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return text.strip()


def normalize_numeric(s: str) -> Optional[float]:
    """Try to parse a string as a number for comparison."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def match_gsm8k(prediction: str, reference: str) -> bool:
    """Check if GSM8K prediction matches reference numerically."""
    pred_str = extract_gsm8k_answer(prediction)
    pred_num = normalize_numeric(pred_str)
    ref_num = normalize_numeric(reference)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) < 1e-5
    return pred_str.strip() == reference.strip()


# ──────────────────────────────────────────────
#  MATH: symbolic answer extraction + normalization
# ──────────────────────────────────────────────

def extract_math_answer(text: str) -> str:
    """
    Extract answer from MATH-style LLM response.
    Looks for \\boxed{...} first, then common patterns.
    """
    text = text.strip()

    # Pattern 1: \boxed{...} with proper brace matching
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # Pattern 2: "the answer is X" / "answer: X"
    m = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*\$?(.+?)\$?(?:\.|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().strip("$").strip()

    # Pattern 3: last LaTeX inline expression $...$
    dollar_exprs = re.findall(r"\$([^$]+)\$", text)
    if dollar_exprs:
        return dollar_exprs[-1].strip()

    # Pattern 4: last "= expression" on a line
    m = re.search(r"=\s*(.+?)\s*$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().strip("$").strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].strip("$").strip()
    return text.strip()


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("\\boxed ")
        if idx == -1:
            return None
        return text[idx + 7:].strip().split()[0] if idx + 7 < len(text) else None

    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return text[start:].strip()


def normalize_math_expr(expr: str) -> str:
    """Normalize a math expression for comparison."""
    s = expr.strip()
    s = s.replace(" ", "")
    # remove surrounding $ or \( \)
    s = s.strip("$")
    s = re.sub(r"^\\[(\[]", "", s)
    s = re.sub(r"\\[)\]]$", "", s)
    # common LaTeX simplifications
    s = s.replace("\\frac", "frac")
    s = s.replace("\\dfrac", "frac")
    s = s.replace("\\tfrac", "frac")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\%", "%")
    s = s.replace("\\$", "$")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace("\\", "")
    return s.lower()


def match_math(prediction: str, reference: str) -> bool:
    """Check if MATH prediction matches reference."""
    pred = extract_math_answer(prediction)
    # Try numeric comparison first
    pred_num = normalize_numeric(pred)
    ref_num = normalize_numeric(reference)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) < 1e-5

    # Normalized string comparison
    return normalize_math_expr(pred) == normalize_math_expr(reference)


# ──────────────────────────────────────────────
#  Code: extract code block from LLM output
# ──────────────────────────────────────────────

def extract_code(text: str) -> str:
    """
    Extract Python code from LLM output.
    Handles markdown code blocks and plain code.
    """
    text = text.strip()

    # Pattern 1: ```python ... ``` or ``` ... ```
    blocks = re.findall(
        r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Pattern 2: ```...``` single-line style
    blocks = re.findall(r"```(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Pattern 3: looks like code already (has def/class/import)
    if re.search(r"^(def |class |import |from )", text, re.MULTILINE):
        return text

    return text


def extract_humaneval_completion(text: str, prompt: str) -> str:
    """
    Extract function completion for HumanEval.
    The prompt contains the function signature; we need just the body.
    """
    code = extract_code(text)
    # If the response includes the full function, strip the prompt part
    if prompt.strip() in code:
        code = code[code.index(prompt.strip()) + len(prompt.strip()):]
    return code


# ──────────────────────────────────────────────
#  MedQA: option letter extraction
# ──────────────────────────────────────────────

def extract_medqa_option(text: str) -> str:
    """
    Extract option letter (A/B/C/D) from LLM response.
    Handles formats like:
      "The answer is B", "B.", "(B)", "Option B", "B: ..."
    """
    text = text.strip()

    # Pattern 1: "The answer is X" / "answer: X"
    m = re.search(
        r"(?:the\s+)?(?:correct\s+)?(?:best\s+)?answer\s*(?:is|=|:)\s*\(?([A-Da-d])\)?",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Pattern 2: starts with option letter "A." or "(A)" or "A:"
    m = re.match(r"^\s*\(?([A-Da-d])\)?[\.\):\s]", text)
    if m:
        return m.group(1).upper()

    # Pattern 3: "Option X"
    m = re.search(r"option\s+\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 4: last standalone letter A-D
    matches = re.findall(r"\b([A-Da-d])\b", text)
    for letter in reversed(matches):
        if letter.upper() in "ABCD":
            return letter.upper()

    return text.strip()[:1].upper() if text.strip() else ""


def match_medqa(prediction: str, reference: str) -> bool:
    """Check if MedQA prediction matches reference option letter."""
    pred_letter = extract_medqa_option(prediction)
    ref_letter = reference.strip().upper()
    # reference could be letter or full text
    if len(ref_letter) == 1 and ref_letter in "ABCD":
        return pred_letter == ref_letter
    # If reference is full text, try to match the extracted letter concept
    return pred_letter == extract_medqa_option(reference)

"""
Safe code execution sandbox for evaluating MBPP+ and HumanEval+ outputs.

Uses subprocess with a timeout to execute generated Python code with test
assertions. Returns pass/fail and any error message.

Cross-platform: works on Windows (spawn) and Linux (fork).
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import tempfile
import os
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds per test execution


def execute_code_with_tests(
    code: str,
    test_code: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, object]:
    """
    Execute generated code together with test assertions in an
    isolated subprocess.

    Returns:
      {"passed": bool, "error": str or None, "timeout": bool}
    """
    full_code = code + "\n" + test_code
    return _run_in_subprocess(full_code, timeout)


def evaluate_mbpp(
    generated_code: str,
    test_list: list,
    test_imports: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, object]:
    """
    Evaluate MBPP-style code: run each assert statement.
    test_list: ["assert func(...) == ...", ...]
    """
    test_code = (test_imports + "\n") if test_imports else ""
    test_code += "\n".join(test_list)
    return execute_code_with_tests(generated_code, test_code, timeout)


def evaluate_humaneval(
    generated_code: str,
    prompt: str,
    test_code: str,
    entry_point: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, object]:
    """
    Evaluate HumanEval-style code: combine prompt + completion + test.
    """
    full_code = prompt + generated_code + "\n" + test_code
    full_code += f"\ncheck({entry_point})\n"
    return execute_code_with_tests(full_code, "", timeout)


def _run_in_subprocess(code: str, timeout: int) -> Dict[str, object]:
    """Run code in an isolated subprocess via a temp file."""
    fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="code_eval_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )

        if result.returncode == 0:
            return {"passed": True, "error": None, "timeout": False}
        else:
            stderr = result.stderr.strip()
            # Extract last meaningful error line
            lines = stderr.split("\n")
            err_msg = lines[-1] if lines else "Unknown error"
            return {"passed": False, "error": err_msg, "timeout": False}

    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout", "timeout": True}
    except Exception as e:
        return {"passed": False, "error": str(e), "timeout": False}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

"""Easy evaluation API for benchmark samples.

Evaluate any model on run-dir benchmark samples with one call::

    from benchmark.eval_api import EvalConfig, evaluate

    report = evaluate(EvalConfig(
        samples="archetype_guided_benchmark/data/runs/final-200-syc-luna-a",
        model_under_test="openai/gpt-5.6-luna",
        assistant_prompt="extreme-cold",
        output_dir="evals/luna-cold",
    ))
"""

from .config import EvalConfig, EvalReport, load_config
from .runner import evaluate

__all__ = ["EvalConfig", "EvalReport", "evaluate", "load_config"]

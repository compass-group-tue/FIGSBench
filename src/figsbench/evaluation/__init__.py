"""Evaluate a model on FIGS.

    from figsbench.evaluation import EvalConfig, evaluate
    report = evaluate(EvalConfig(samples="data/benchmark_500.jsonl",
                                 model_under_test="openai/gpt-5.6-luna",
                                 output_dir="runs/my-model"))
"""

from .config import EvalConfig, EvalReport, load_config
from .runner import evaluate

__all__ = ["EvalConfig", "EvalReport", "evaluate", "load_config"]

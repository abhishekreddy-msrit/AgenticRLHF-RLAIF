"""Evaluation helpers.

Keep this package init import-light to avoid circular imports with
`src.judge.agent` (which needs `src.eval.behavioral`).
Import symbols from submodules directly:

  from src.eval.behavioral import analyze_behavior
  from src.eval.benchmark import compare_runs, load_eval_set, run_model_on_benchmark
"""

__all__ = [
    "analyze_behavior",
    "compare_runs",
    "load_eval_set",
    "run_model_on_benchmark",
]


def __getattr__(name: str):
    if name == "analyze_behavior":
        from .behavioral import analyze_behavior as _fn

        return _fn
    if name in {"compare_runs", "load_eval_set", "run_model_on_benchmark"}:
        from . import benchmark as _bm

        return getattr(_bm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

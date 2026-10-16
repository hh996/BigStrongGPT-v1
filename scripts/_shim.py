"""从 scripts/ 根目录转发到 scripts/benchmark/。"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def run_benchmark(module: str) -> None:
    import importlib

    mod = importlib.import_module(f"benchmark.{module}")
    mod.main()

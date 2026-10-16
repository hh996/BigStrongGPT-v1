"""兼容入口 → scripts/benchmark/compare_qwen_sft.py"""
from _shim import run_benchmark

if __name__ == "__main__":
    run_benchmark("compare_qwen_sft")

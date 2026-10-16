"""兼容入口 → scripts/benchmark/compare_sft_ckpts.py"""
from _shim import run_benchmark

if __name__ == "__main__":
    run_benchmark("compare_sft_ckpts")

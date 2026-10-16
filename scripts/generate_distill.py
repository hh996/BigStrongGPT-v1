"""用量化 7B 老师批量生成 SFT 数据。不训练 7B。

示例（4070，约 1~5 万条可过夜）:
  python scripts/generate_distill.py --max-samples 20000

默认从 dataset/sft_merged.jsonl 取用户问题，写出 dataset/sft_distill_7b.jsonl。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Windows + 国内镜像：在 import transformers 前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= limit:
                break
            row = json.loads(line)
            conv = row.get("conversations") or []
            if not conv:
                continue
            text = (conv[0].get("content") or "").strip()
            if text:
                prompts.append(text)
    return prompts


def _bytes_under_model_cache(cache: Path) -> int:
    total = 0
    if not cache.is_dir():
        return 0
    for p in cache.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix in {".safetensors", ".incomplete"} or "safetensors" in p.name:
            total += p.stat().st_size
    return total


def _append_log(log_path: Path | None, line: str) -> None:
    if not log_path:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(line if line.endswith("\n") else line + "\n")
    except OSError:
        pass


def _watch_download(cache: Path, log_path: Path | None, stop: threading.Event) -> None:
    while not stop.is_set():
        gb = _bytes_under_model_cache(cache) / 1e9
        line = f"[{time.strftime('%H:%M:%S')}] 模型下载进度（约） {gb:.2f} / 15 GB\n"
        print(line, end="", flush=True)
        _append_log(log_path, line.rstrip("\n"))
        stop.wait(30)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompts", default=str(REPO / "dataset" / "sft_merged.jsonl"))
    parser.add_argument("--output", default=str(REPO / "dataset" / "sft_distill_7b.jsonl"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true", help="不用 bitsandbytes 4bit")
    parser.add_argument(
        "--model-dir",
        default="",
        help="已下载的本地模型目录（跳过在线拉取）",
    )
    parser.add_argument(
        "--use-modelscope",
        action="store_true",
        help="从 ModelScope 下载（国内推荐，HF 大文件易卡在 0%%）",
    )
    parser.add_argument(
        "--progress-log",
        default=str(REPO / "dataset" / "distill_progress.log"),
        help="追加进度（勿与 Tee-Object 同一文件，否则 -Wait 会锁文件）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有 output jsonl 行数继续（追加写入）",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(Path(args.prompts), args.max_samples)
    model_id = args.model
    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.is_dir():
            print(f"model-dir 不存在: {model_path}", file=sys.stderr)
            sys.exit(1)
        load_path = str(model_path)
        print(f"prompts={len(prompts)} local_model={load_path}")
    elif args.use_modelscope:
        from modelscope import snapshot_download

        cache = REPO / "models" / "qwen2.5-7b-instruct"
        log_path = Path(args.progress_log) if args.progress_log else None
        print(f"prompts={len(prompts)} modelscope={model_id} (~15GB)...", flush=True)
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_download,
            args=(cache, log_path, stop),
            daemon=True,
        )
        watcher.start()
        try:
            load_path = snapshot_download(model_id, cache_dir=str(cache))
        finally:
            stop.set()
        line = f"[{time.strftime('%H:%M:%S')}] 模型下载完成: {load_path}\n"
        print(line, end="", flush=True)
        _append_log(log_path, line.rstrip("\n"))
    else:
        from huggingface_hub import snapshot_download

        print(f"prompts={len(prompts)} model={model_id}")
        print("正在下载/校验模型权重（约 15GB；若长期 0%% 请改用 --use-modelscope）...", flush=True)
        load_path = snapshot_download(repo_id=model_id, resume_download=True, max_workers=4)
        print(f"模型目录: {load_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if not args.no_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs = {
                "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                "device_map": "auto",
            }
            print("使用 4bit 量化加载")
        except Exception as exc:
            print(f"4bit 不可用（{exc}），改为 bf16")
    model = AutoModelForCausalLM.from_pretrained(load_path, **kwargs)
    model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and out.is_file():
        with out.open(encoding="utf-8") as existing:
            start = sum(1 for _ in existing)
        print(f"resume: 已有 {start} 条，从第 {start + 1} 条继续", flush=True)
    mode = "a" if start > 0 else "w"
    log_path = Path(args.progress_log) if args.progress_log else None
    with out.open(mode, encoding="utf-8") as f:
        for i, prompt in enumerate(prompts[start:], start=start):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            answer = tokenizer.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            row = {"conversations": [{"content": prompt}, {"content": answer}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            msg = f"[{i + 1}/{len(prompts)}] {prompt[:40]} -> {len(answer)} chars"
            print(msg, flush=True)
            if (i + 1) % 10 == 0:
                _append_log(log_path, msg)
    print(f"写入 {out}")


if __name__ == "__main__":
    main()

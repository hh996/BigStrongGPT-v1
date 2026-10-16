"""同一组 prompt：Qwen2.5-7B vs sft v1 / sft v2。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "scripts"))

from benchmark.compare_sft_ckpts import BENCH, generate, load_model  # noqa: E402

DEFAULT_QWEN = (
    REPO / "models/qwen2.5-7b-instruct/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
)


def load_qwen(model_dir: Path):
    tok = AutoTokenizer.from_pretrained(model_dir)
    kwargs = {
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model.eval()
    return model, tok


def generate_qwen(model, tok, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", default=str(DEFAULT_QWEN))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--student-max-new-tokens", type=int, default=64)
    parser.add_argument("--out", default=str(REPO / "output/v1/compare_qwen_sft.txt"))
    args = parser.parse_args()

    qwen_dir = Path(args.qwen_dir)
    if not qwen_dir.is_dir():
        print(f"Qwen 目录不存在: {qwen_dir}", file=sys.stderr)
        sys.exit(1)

    with (REPO / "configs/v1/pretrain.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stu_tok = AutoTokenizer.from_pretrained(REPO / cfg["tokenizer_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lines: list[str] = []
    lines.append(
        f"Qwen: 贪心 max_new={args.max_new_tokens} | "
        f"学生(sft_v1/v2): 贪心 max_new={args.student_max_new_tokens}\n"
    )

    print("加载 Qwen 7B (4bit)...", flush=True)
    qwen, qtok = load_qwen(qwen_dir)
    qwen_replies = [generate_qwen(qwen, qtok, p, args.max_new_tokens) for p in BENCH]
    del qwen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ckpt_v1 = REPO / "output/v1/sft/last.pt"
    ckpt_v2 = REPO / "output/v1/sft_v2/last.pt"
    print("加载 sft v1...", flush=True)
    m1 = load_model(ckpt_v1, cfg, device)
    v1 = [generate(m1, stu_tok, p, device, args.student_max_new_tokens, True) for p in BENCH]
    del m1
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("加载 sft v2...", flush=True)
    m2 = load_model(ckpt_v2, cfg, device)
    v2 = [generate(m2, stu_tok, p, device, args.student_max_new_tokens, True) for p in BENCH]
    del m2

    for i, prompt in enumerate(BENCH):
        block = [
            "=" * 72,
            f"[{i + 1}/{len(BENCH)}] 用户: {prompt}",
            f"  [Qwen2.5-7B] {qwen_replies[i]}",
            f"  [sft_v1 87M] {v1[i]}",
            f"  [sft_v2 87M] {v2[i]}",
            "",
        ]
        lines.extend(block)
        print("\n".join(block))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()

"""对比两个 SFT checkpoint（相同 prompt、相同解码）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

from bigstrong.modeling import BigStrongConfig, BigStrongForCausalLLM  # noqa: E402

BENCH = [
    "你好，请做一下自我介绍。",
    "1+1 等于几？",
    "3 乘以 7 等于多少？",
    "用一句话解释什么是预训练。",
    "用一句话解释什么是机器学习。",
    "中国的首都是哪里？",
    "请用一句话介绍长城。",
    "把「谢谢你的帮助」翻译成英文。",
    "列举三种常见的水果。",
    "如果明天下雨，出门建议带什么？（一句话）",
    "写一句关于春天的短句。",
    "什么是光合作用？用一句话回答。",
]


def load_model(ckpt: Path, cfg: dict, device: str):
    mcfg = cfg["model"]
    config = BigStrongConfig(
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        vocab_size=mcfg["vocab_size"],
    )
    model = BigStrongForCausalLLM(config)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def generate(model, tok, prompt: str, device: str, max_new_tokens: int, greedy: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "do_sample": not greedy,
    }
    if not greedy:
        gen_kw.update(temperature=0.7, top_p=0.9, repetition_penalty=1.15)
    with torch.inference_mode():
        out = model.generate(ids, **gen_kw)
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="output/v1/sft/last.pt")
    parser.add_argument("--b", default="output/v1/sft_v2/last.pt")
    parser.add_argument("--label-a", default="sft_v1")
    parser.add_argument("--label-b", default="sft_v2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--no-greedy", action="store_true")
    args = parser.parse_args()
    greedy = args.greedy and not args.no_greedy

    with (REPO / "configs/v1/pretrain.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(REPO / cfg["tokenizer_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_a = REPO / args.a if not Path(args.a).is_absolute() else Path(args.a)
    ckpt_b = REPO / args.b if not Path(args.b).is_absolute() else Path(args.b)

    print(f"解码: {'贪心' if greedy else '采样'}  max_new_tokens={args.max_new_tokens}  device={device}\n")
    model_a = load_model(ckpt_a, cfg, device)
    replies_a = [generate(model_a, tok, p, device, args.max_new_tokens, greedy) for p in BENCH]
    del model_a
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    model_b = load_model(ckpt_b, cfg, device)
    replies_b = [generate(model_b, tok, p, device, args.max_new_tokens, greedy) for p in BENCH]
    del model_b

    for i, prompt in enumerate(BENCH):
        print("=" * 72)
        print(f"[{i + 1}/{len(BENCH)}] 用户: {prompt}")
        print(f"  [{args.label_a}] {replies_a[i]}")
        print(f"  [{args.label_b}] {replies_b[i]}")


if __name__ == "__main__":
    main()

"""v1 学生模型对话生成评测。"""
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

PROMPTS = [
    "你好，请做一下自我介绍。",
    "1+1 等于几？",
    "用一句话解释什么是预训练。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1/pretrain.yaml")
    parser.add_argument("--ckpt", default="output/v1/sft/last.pt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="（默认已是贪心，仅保留旧命令兼容）",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="采样解码（默认贪心，87M 推荐贪心）",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--prompt", action="append", help="额外测试句，可多次指定")
    args = parser.parse_args()

    prompts = list(PROMPTS)
    if args.prompt:
        prompts.extend(args.prompt)

    with (REPO / args.config).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tok = AutoTokenizer.from_pretrained(REPO / cfg["tokenizer_dir"])
    mcfg = cfg["model"]
    config = BigStrongConfig(
        hidden_size=mcfg["hidden_size"],
        num_hidden_layers=mcfg["num_hidden_layers"],
        num_attention_heads=mcfg["num_attention_heads"],
        num_key_value_heads=mcfg["num_key_value_heads"],
        vocab_size=mcfg["vocab_size"],
    )
    model = BigStrongForCausalLLM(config)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO / ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    gen_kw: dict = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
    }
    if args.sample:
        gen_kw.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        gen_kw["do_sample"] = False

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.inference_mode():
            out = model.generate(ids, **gen_kw)
        reply = tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
        print("---")
        print(f"用户: {prompt}")
        print(f"助手: {reply}")


if __name__ == "__main__":
    main()

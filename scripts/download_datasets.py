#!/usr/bin/env python3
"""
下载并转换训练数据为 BigStrongGPT 项目格式。

用法:
  python scripts/download_datasets.py --all
  python scripts/download_datasets.py --pretrain --sft --dpo --lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
OUT_DIR = ROOT / "dataset"


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  写入 {path} ({len(rows)} 条)")


def load_hf_dataset(name: str, config: str | None = None, split: str = "train"):
    from datasets import load_dataset

    if config:
        return load_dataset(name, config, split=split)
    return load_dataset(name, split=split)


def sample_rows(rows: list[dict], max_samples: int | None, seed: int = 42) -> list[dict]:
    if max_samples is None or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_samples)


# ---------------------------------------------------------------------------
# Pretrain
# ---------------------------------------------------------------------------

def download_pretrain_wikipedia(max_samples: int | None = 100_000) -> Path:
    """Wikipedia 中文，适合本地调试。"""
    print("\n[Pretrain] Wikipedia 中文...")
    try:
        ds = load_hf_dataset("wikimedia/wikipedia", "20231101.zh")
    except Exception:
        # 兼容旧版 datasets 的加载方式
        ds = load_hf_dataset("wikipedia", "20231101.zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        text = (item.get("text") or "").strip()
        if len(text) < 100:
            continue
        rows.append({"text": text})
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "pretrain_wikipedia.jsonl"
    write_jsonl(out, rows)
    return out


def download_pretrain_skypile(max_files: int = 2, max_samples: int | None = None) -> Path | None:
    """下载 SkyPile 前 N 个 zh head shard，流式写入 jsonl（正式预训练可 --skypile-files 12 --no-pretrain-cap）。"""
    print(f"\n[Pretrain] SkyPile-150B（{max_files} 个 shard）...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  跳过：需要 pip install huggingface_hub")
        return None

    shard_names = [f"data/2021-43_zh_head_{i:04d}.jsonl" for i in range(max_files)]
    out = OUT_DIR / "pretrain_skypile.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as out_f:
        for shard in shard_names:
            print(f"  下载 {shard} ...")
            try:
                local = hf_hub_download(
                    repo_id="Skywork/SkyPile-150B",
                    repo_type="dataset",
                    filename=shard,
                    local_dir=RAW_DIR / "skypile",
                )
                with open(local, encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        text = (data.get("text") or "").strip()
                        if len(text) < 50:
                            continue
                        out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        count += 1
                        if max_samples is not None and count >= max_samples:
                            break
            except Exception as e:
                print(f"  下载失败 {shard}: {e}")
                break
            print(f"  累计 {count} 条")
            if max_samples is not None and count >= max_samples:
                break

    if count == 0:
        print("  SkyPile 下载失败，请手动下载或改用 --pretrain-wiki")
        out.unlink(missing_ok=True)
        return None

    print(f"  写入 {out} ({count} 条)")
    return out


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def belle_to_conversations(instruction: str, inp: str, output: str) -> dict:
    user = instruction.strip()
    if inp and inp.strip():
        user = f"{user}\n{inp.strip()}"
    return {
        "conversations": [
            {"content": user},
            {"content": output.strip()},
        ]
    }


def download_sft_alpaca_zh(max_samples: int | None = None) -> Path:
    print("\n[SFT] shibing624/alpaca-zh ...")
    ds = load_hf_dataset("shibing624/alpaca-zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        inst = item.get("instruction") or ""
        inp = item.get("input") or ""
        out = item.get("output") or ""
        if not out.strip():
            continue
        rows.append(belle_to_conversations(inst, inp, out))
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "sft_alpaca_zh.jsonl"
    write_jsonl(out, rows)
    return out


def download_sft_belle(max_samples: int | None = 30_000) -> Path | None:
    print("\n[SFT] BelleGroup/train_0.5M_CN ...")
    try:
        ds = load_hf_dataset("BelleGroup/train_0.5M_CN")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        rows.append(
            belle_to_conversations(
                item.get("instruction") or "",
                item.get("input") or "",
                item.get("output") or "",
            )
        )
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "sft_belle.jsonl"
    write_jsonl(out, rows)
    return out


def merge_sft_files(out_name: str = "sft_merged.jsonl") -> Path:
    files = sorted(OUT_DIR.glob("sft_*.jsonl"))
    files = [f for f in files if f.name != out_name]
    if not files:
        print("  无 SFT 文件可合并")
        return OUT_DIR / out_name

    seen = set()
    merged = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(json.loads(line))

    out = OUT_DIR / out_name
    write_jsonl(out, merged)
    return out


def merge_sft_v2(out_name: str = "sft_merged_v2.jsonl") -> Path:
    """第一轮 sft_merged + 7B 蒸馏，去重后供第二轮 SFT。"""
    files = [OUT_DIR / "sft_merged.jsonl", OUT_DIR / "sft_distill_7b.jsonl"]
    files = [f for f in files if f.is_file()]
    if len(files) < 2:
        print("  merge_sft_v2 需要 sft_merged.jsonl 与 sft_distill_7b.jsonl")
        return OUT_DIR / out_name
    seen: set[str] = set()
    merged: list[dict] = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(json.loads(line))
    out = OUT_DIR / out_name
    write_jsonl(out, merged)
    return out


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def download_dpo_shibing(max_samples: int | None = 5_000) -> Path:
    print("\n[DPO] shibing624/DPO-En-Zh-20k-Preference (中文部分) ...")
    ds = load_hf_dataset("shibing624/DPO-En-Zh-20k-Preference", "zh")
    rows = []
    for item in tqdm(ds, desc="  转换"):
        # 优先中文字段；该数据集含 system/history/question
        q = item.get("question") or item.get("prompt") or ""
        chosen = item.get("response_chosen") or item.get("chosen") or ""
        rejected = item.get("response_rejected") or item.get("rejected") or ""
        system = item.get("system") or ""
        if system:
            q = f"{system}\n{q}".strip()
        if not q or not chosen or not rejected:
            continue
        # 简单过滤：中文占比（含 CJK 字符）
        if sum(1 for c in q + chosen if "\u4e00" <= c <= "\u9fff") < 5:
            continue
        rows.append({"prompt": q.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})

    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "dpo_zh.jsonl"
    write_jsonl(out, rows)
    return out


def download_dpo_small(max_samples: int | None = 5_000) -> Path | None:
    print("\n[DPO] Karsh-CAI/btfChinese-DPO-small ...")
    try:
        ds = load_hf_dataset("Karsh-CAI/btfChinese-DPO-small")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        prompt = item.get("prompt") or item.get("question") or ""
        chosen = item.get("chosen") or item.get("response_chosen") or ""
        rejected = item.get("rejected") or item.get("response_rejected") or ""
        if prompt and chosen and rejected:
            rows.append({"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "dpo_btf_small.jsonl"
    write_jsonl(out, rows)
    return out


def merge_dpo_files(out_name: str = "dpo_merged.jsonl") -> Path:
    files = [OUT_DIR / "dpo_zh.jsonl", OUT_DIR / "dpo_btf_small.jsonl"]
    files = [f for f in files if f.exists()]
    merged = []
    seen = set()
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() in seen:
                    continue
                seen.add(line.strip())
                merged.append(json.loads(line))
    out = OUT_DIR / out_name
    write_jsonl(out, merged)
    return out


# ---------------------------------------------------------------------------
# DPO clean（无 btf，可选 enthusiastic + 过滤）
# ---------------------------------------------------------------------------

_DPO_PROFANITY = __import__("re").compile(
    r"他妈|你妈的|混蛋|瞎逼|操你|傻逼|滚蛋|去死|逼逼"
)


def _dpo_chat_token_len(tokenizer, prompt: str, response: str) -> int:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(text).input_ids)


def filter_dpo_row(
    row: dict,
    *,
    min_chosen: int = 15,
    min_rejected: int = 5,
    max_chat_tokens: int = 480,
    tokenizer=None,
) -> bool:
    p = (row.get("prompt") or "").strip()
    c = (row.get("chosen") or "").strip()
    r = (row.get("rejected") or "").strip()
    if not p or not c or not r:
        return False
    if c == r:
        return False
    if len(c) < min_chosen or len(r) < min_rejected:
        return False
    if _DPO_PROFANITY.search(c) or _DPO_PROFANITY.search(r):
        return False
    if tokenizer is not None and max_chat_tokens > 0:
        if _dpo_chat_token_len(tokenizer, p, c) > max_chat_tokens:
            return False
    return True


def build_dpo_clean(
    out_name: str = "dpo_clean.jsonl",
    max_shibing: int | None = None,
    include_enthusiastic: bool = True,
    max_chat_tokens: int = 480,
) -> Path:
    """shibing 中文 +（可选）chinese-enthusiastic-dpo，过滤毒性/重复/过长。"""
    print("\n[DPO clean] 构建 dpo_clean（不含 btf）...")
    tokenizer = None
    tok_dir = ROOT / "tokenizer" / "v1"
    if tok_dir.is_dir() and max_chat_tokens > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        print(f"  使用词表过滤 >{max_chat_tokens} tokens 的样本")

    rows: list[dict] = []
    seen_prompt: set[str] = set()

    def add_row(prompt: str, chosen: str, rejected: str) -> None:
        row = {"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()}
        key = row["prompt"][:200]
        if key in seen_prompt:
            return
        if not filter_dpo_row(row, tokenizer=tokenizer, max_chat_tokens=max_chat_tokens):
            return
        seen_prompt.add(key)
        rows.append(row)

    print("  加载 shibing624/DPO-En-Zh-20k-Preference (zh)...")
    ds = load_hf_dataset("shibing624/DPO-En-Zh-20k-Preference", "zh")
    shibing_buf = []
    for item in tqdm(ds, desc="  shibing"):
        q = item.get("question") or item.get("prompt") or ""
        chosen = item.get("response_chosen") or item.get("chosen") or ""
        rejected = item.get("response_rejected") or item.get("rejected") or ""
        system = item.get("system") or ""
        if system:
            q = f"{system}\n{q}".strip()
        if sum(1 for c in q + chosen if "\u4e00" <= c <= "\u9fff") < 5:
            continue
        shibing_buf.append({"prompt": q.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()})
    if max_shibing is not None and len(shibing_buf) > max_shibing:
        shibing_buf = sample_rows(shibing_buf, max_shibing)
    for row in shibing_buf:
        add_row(row["prompt"], row["chosen"], row["rejected"])

    if include_enthusiastic:
        print("  加载 markyfsun/chinese-enthusiastic-dpo ...")
        try:
            eds = load_hf_dataset("markyfsun/chinese-enthusiastic-dpo")
            for item in tqdm(eds, desc="  enthusiastic"):
                add_row(
                    item.get("prompt") or "",
                    item.get("chosen") or "",
                    item.get("rejected") or "",
                )
        except Exception as e:
            print(f"  enthusiastic 跳过: {e}")

    out = OUT_DIR / out_name
    write_jsonl(out, rows)
    print(f"  过滤后保留 {len(rows)} 条 -> {out}")
    return out


# ---------------------------------------------------------------------------
# LoRA 医疗
# ---------------------------------------------------------------------------

def sharegpt_to_conversations(conversations: list) -> dict | None:
    """ShareGPT: from human/gpt 或 role/content."""
    msgs = []
    for turn in conversations:
        if isinstance(turn, dict):
            role = turn.get("from") or turn.get("role") or ""
            content = turn.get("value") or turn.get("content") or ""
            role = role.lower()
            if role in ("human", "user"):
                msgs.append({"content": content.strip()})
            elif role in ("gpt", "assistant", "bot"):
                msgs.append({"content": content.strip()})
    if len(msgs) >= 2 and len(msgs) % 2 == 0:
        return {"conversations": msgs}
    return None


def download_lora_huatuo(max_samples: int | None = 10_000) -> Path | None:
    print("\n[LoRA] shibing624/huatuo_medical_qa_sharegpt ...")
    try:
        ds = load_hf_dataset("shibing624/huatuo_medical_qa_sharegpt")
    except Exception as e:
        print(f"  加载失败: {e}")
        return None

    rows = []
    for item in tqdm(ds, desc="  转换"):
        conv = item.get("conversations") or item.get("conversation") or []
        parsed = sharegpt_to_conversations(conv)
        if parsed:
            rows.append(parsed)
    rows = sample_rows(rows, max_samples)
    out = OUT_DIR / "lora_medical.jsonl"
    write_jsonl(out, rows)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="下载并转换 BigStrongGPT 训练数据")
    parser.add_argument("--all", action="store_true", help="下载全部（抽样版）")
    parser.add_argument("--pretrain", action="store_true", help="预训练数据")
    parser.add_argument("--pretrain-wiki", action="store_true", help="仅 Wikipedia 中文（调试）")
    parser.add_argument("--pretrain-skypile", action="store_true", help="SkyPile shard（正式）")
    parser.add_argument("--skypile-files", type=int, default=2, help="SkyPile shard 数量")
    parser.add_argument("--sft", action="store_true", help="SFT 数据")
    parser.add_argument(
        "--merge-sft-v2",
        action="store_true",
        help="合并 sft_merged + sft_distill_7b -> sft_merged_v2.jsonl（第二轮 SFT）",
    )
    parser.add_argument("--dpo", action="store_true", help="DPO 数据（shibing + btf 合并，不推荐正式训练）")
    parser.add_argument(
        "--dpo-clean",
        action="store_true",
        help="干净 DPO：shibing 中文 + enthusiastic，过滤后写入 dpo_clean.jsonl",
    )
    parser.add_argument(
        "--no-dpo-btf",
        action="store_true",
        help="--dpo 时不下载 btf，仅 merge dpo_zh.jsonl",
    )
    parser.add_argument(
        "--no-dpo-enthusiastic",
        action="store_true",
        help="--dpo-clean 时不合并 chinese-enthusiastic-dpo",
    )
    parser.add_argument("--max-shibing-dpo", type=int, default=None, help="--dpo-clean 时 shibing 条数上限（默认全量）")
    parser.add_argument("--dpo-max-tokens", type=int, default=480, help="--dpo-clean ChatML 最大 token（0=不滤）")
    parser.add_argument("--lora", action="store_true", help="LoRA 医疗数据")
    parser.add_argument("--max-pretrain-samples", type=int, default=100_000, help="预训练条数上限（调试）")
    parser.add_argument(
        "--no-pretrain-cap",
        action="store_true",
        help="SkyPile 预训练不截断条数（正式预训练）",
    )
    parser.add_argument("--max-sft-samples", type=int, default=30_000)
    parser.add_argument("--max-dpo-samples", type=int, default=5_000)
    parser.add_argument("--max-lora-samples", type=int, default=10_000)
    args = parser.parse_args()

    if args.all:
        args.pretrain = args.sft = args.dpo = args.lora = True
        args.pretrain_wiki = True

    if not any(
        [
            args.pretrain,
            args.pretrain_wiki,
            args.pretrain_skypile,
            args.sft,
            args.merge_sft_v2,
            args.dpo,
            args.dpo_clean,
            args.lora,
        ]
    ):
        parser.print_help()
        return

    ensure_dirs()
    print(f"输出目录: {OUT_DIR}")

    if args.merge_sft_v2:
        merge_sft_v2()
        if not any(
            [args.pretrain, args.pretrain_wiki, args.pretrain_skypile, args.sft, args.dpo, args.dpo_clean, args.lora]
        ):
            return

    if args.pretrain or args.pretrain_wiki:
        download_pretrain_wikipedia(max_samples=args.max_pretrain_samples)

    pretrain_cap = None if args.no_pretrain_cap else args.max_pretrain_samples
    if args.pretrain_skypile:
        download_pretrain_skypile(max_files=args.skypile_files, max_samples=pretrain_cap)
    elif args.pretrain:
        download_pretrain_skypile(max_files=args.skypile_files, max_samples=pretrain_cap)

    if args.pretrain and not args.pretrain_wiki and not args.pretrain_skypile:
        download_pretrain_wikipedia(max_samples=args.max_pretrain_samples)

    if args.sft:
        download_sft_alpaca_zh(max_samples=min(args.max_sft_samples, 20_000))
        download_sft_belle(max_samples=args.max_sft_samples)
        merge_sft_files()

    if args.dpo_clean:
        build_dpo_clean(
            include_enthusiastic=not args.no_dpo_enthusiastic,
            max_shibing=args.max_shibing_dpo,
            max_chat_tokens=args.dpo_max_tokens,
        )

    if args.dpo:
        download_dpo_shibing(max_samples=args.max_dpo_samples)
        if not args.no_dpo_btf:
            download_dpo_small(max_samples=args.max_dpo_samples)
        merge_dpo_files()

    if args.lora:
        download_lora_huatuo(max_samples=args.max_lora_samples)

    print("\n完成！请查看 dataset/ 目录。")
    print("训练示例:")
    print("  cd legacy/v0_25m/train")
    print("  python train_pretrain.py")
    print("  python train_full_sft.py")
    print("  python train_dpo.py")
    print("  python train_lora.py")


if __name__ == "__main__":
    main()

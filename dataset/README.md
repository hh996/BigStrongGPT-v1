# 训练数据说明（本机只留主链路）

主链路：**Pretrain → SFT**；对话用 **`output/v1/sft/last.pt`**。

| 阶段 | 本目录数据文件 | 权重 |
|------|----------------|------|
| 预训练 | `pretrain_skypile.jsonl`（SkyPile 中文小样；云上同源、量更大） | `output/v1/pretrain/last.pt` |
| SFT | **`sft_merged.jsonl`**（Belle 3 万 + alpaca-zh 2 万合并去重，≈5 万条） | **`output/v1/sft/last.pt`** |

原料 `sft_belle` / `sft_alpaca_zh`、蒸馏 `sft_distill_7b`、`sft_merged_v2`、`dpo_clean`、维基预训练等 **本机已删**（曾做实验或已合并进 `sft_merged`）。需要可从 `scripts/download_datasets.py` 重新下。

**看样例**：`samples/01_pretrain/`、`samples/02_sft/merged_head20.jsonl`（每行 `conversations: [用户, 助手]`）。

预训练每行：`{"text": "..."}`。

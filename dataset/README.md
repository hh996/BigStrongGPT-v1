# 训练数据说明（本机保留 = 各 checkpoint 用到的）

| 权重目录 | 训练时读的数据 | 本机 `dataset/` 文件 |
|----------|----------------|----------------------|
| `output/v1/pretrain/` | SkyPile 中文 | `pretrain_skypile.jsonl` |
| **`output/v1/sft/`**（主用） | 第一轮 SFT | **`sft_merged.jsonl`** |
| `output/v1/sft_v2/` | 第二轮 SFT | `sft_merged_v2.jsonl`（= merged + 蒸馏合并） |
| `output/v1/dpo_clean/` | 干净 DPO | `dpo_clean.jsonl` |

**第二轮 SFT 原料链**（不单独训模型，但生成 `sft_merged_v2` 需要）：

- `sft_distill_7b.jsonl`：从 `sft_merged` 抽问题，Qwen2.5-7B 写答案（`generate_distill.py`）

**不保留**：Belle/alpaca 单文件（已并进 `sft_merged`）、脏 `dpo_merged`、维基预训练、smoke 等。

**看样例**：`samples/01_pretrain/`、`02_sft/`、`03_dpo/`。

若 `sft_distill_7b` / `sft_merged_v2` 缺失：先 `.\scripts\run_distill_local.ps1`，再 `python scripts/download_datasets.py --merge-sft-v2`。

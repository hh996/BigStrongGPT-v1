# 数据清单（运维向）

**按阶段看懂训练用了啥，请先看 [README.md](README.md)。**

下文只列文件大小与路径，便于核对本机是否拷全。

## 预训练

| 文件 | 行数 | 说明 |
|------|------|------|
| `pretrain_skypile.jsonl` | 100 000 | 本地 SkyPile 小样；云上同源、shard 更多 |
| `pretrain_wikipedia.jsonl` | 见本机 | 调试集，非云上主路径 |

## SFT

| 文件 | 行数 | 说明 |
|------|------|------|
| `sft_belle.jsonl` | 30 000 | 原料 |
| `sft_alpaca_zh.jsonl` | 20 000 | 原料 |
| `sft_merged.jsonl` | 49 995 | 第一轮 SFT |
| `sft_distill_7b.jsonl` | 20 000 | 7B 蒸馏 |
| `sft_merged_v2.jsonl` | 69 941 | 第二轮 SFT |

## DPO

| 文件 | 行数 | 说明 |
|------|------|------|
| ~~`dpo_merged.jsonl`~~ | — | 脏数据，**本机已删** |
| `dpo_clean.jsonl` | 12 091 | 本地干净 DPO |

## 样本目录

`samples/01_pretrain` … `04_distill`：各阶段可打开的小 jsonl。

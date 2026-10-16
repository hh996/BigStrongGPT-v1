# 本机数据清单（v1 训练用了什么）

`*.jsonl` 不进 git，仅本地备份；用文本编辑器打开 **`samples/`** 下小文件即可快速感受格式与内容。

## 预训练（Pretrain）

| 文件 | 行数（约） | 大小 | 说明 |
|------|------------|------|------|
| `pretrain_skypile.jsonl` | **100 000** | ~274 MB | 本地脚本从 **SkyPile-150B** 中文 head **前 2 个 shard** 流式写出，并受 `--max-pretrain-samples 100000` 上限（见 `download_datasets.py`）。**不是**全量 150B。 |
| `pretrain_wikipedia.jsonl` | （调试集） | ~218 MB | Wikipedia 中文，本地调试/备用；**云上正式 pretrain** 以 SkyPile 为主（`configs/v1/pretrain.yaml` → `pretrain_skypile.jsonl`）。 |
| `samples/skypile_raw_shard0000_head30.jsonl` | 30 行 | 极小 | 原始 shard `2021-43_zh_head_0000` 前几行，字段含 `text`。完整 shard 在旧目录 `dataset/raw/skypile/`（约 2 GB/个），**未复制到 v1**（太大）。 |

云上 `output/v1/pretrain/last.pt`（step **818670**）是在 **更完整 SkyPile 流程** 上训的；本机 `pretrain_skypile.jsonl` 主要用于复现/理解数据源，不等同于云上见过的全部 token。

若要本地加大预训练集：

```powershell
python scripts/download_datasets.py --pretrain-skypile --skypile-files 12 --no-pretrain-cap
```

## 监督微调（SFT）

| 文件 | 行数（约） | 配置 | 说明 |
|------|------------|------|------|
| `sft_merged.jsonl` | **49 995** | `configs/v1/sft.yaml` | 第一轮 SFT（云上）；合并 Belle 等 |
| `sft_distill_7b.jsonl` | **20 000** | — | Qwen2.5-7B 蒸馏回复，`generate_distill.py` 生成 |
| `sft_merged_v2.jsonl` | **69 941** | `configs/v1/sft_v2.yaml` | `sft_merged` + 蒸馏合并后第二轮 SFT |
| `sft_belle.jsonl` | — | 子集 | Belle 源（合并前），便于对照 |

**推荐对话权重**：`output/v1/sft/last.pt`（第一轮 SFT），优于 `sft_v2`。

## DPO

| 文件 | 行数（约） | 配置 | 说明 |
|------|------------|------|------|
| `dpo_clean.jsonl` | **12 091** | `dpo_clean.yaml` | 无 btf 毒性 preferred；本地干净 DPO |
| `dpo_merged.jsonl` | — | 旧 `dpo.yaml` | **含 btf，已弃用**；仅留档对照，勿再训 |

## 权重（`output/v1/`，不进 git）

| 路径 | 用途 |
|------|------|
| `pretrain/last.pt` | 预训练基座 |
| `sft/last.pt` | **主用对话** |
| `sft_v2/last.pt` | 第二轮 SFT（实验） |
| `dpo_clean/last.pt` | 干净 DPO（实验） |
| `dpo/last.pt` | 旧脏 DPO（勿用） |
| `smoke_*` | 冒烟测试权重 |

## 快速查看样本

```powershell
Get-Content dataset\samples\sft_merged_head30.jsonl -TotalCount 3
Get-Content dataset\samples\pretrain_skypile_head80.jsonl -TotalCount 2
```

SFT/DPO 每行一般为 JSON（`messages` 或 `prompt`/`chosen`/`rejected`）；预训练每行 `{"text": "..."}`。

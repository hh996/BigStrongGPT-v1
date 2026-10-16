# 脚本说明

## 数据

| 脚本 | 作用 |
|------|------|
| `download_datasets.py` | 下载/转换 Pretrain、SFT、DPO；`--merge-sft-v2`、`--dpo-clean` |

## 训练入口（Python）

```powershell
$env:PYTHONPATH = "src;."
python -m bigstrong.train --config configs/v1/<stage>.yaml
```

## 蒸馏（7B 老师）

| 脚本 | 作用 |
|------|------|
| `generate_distill.py` | 生成 `dataset/sft_distill_7b.jsonl` |
| `local/run_distill_local.ps1` | Windows 一键续跑（`--resume`） |

## 评测

| 脚本 | 作用 |
|------|------|
| `benchmark/eval_v1.py` | 单模型对话（默认贪心、64 token） |
| `benchmark/compare_sft_ckpts.py` | 两个学生 ckpt 对比 |
| `benchmark/compare_qwen_sft.py` | Qwen 7B vs 学生 |

根目录下的 `eval_v1.py` 等为**兼容入口**，转发到 `benchmark/`。

## 本地一键（Windows）

| 脚本 | 作用 |
|------|------|
| `local/run_distill_local.ps1` | 7B 蒸馏 |
| `local/run_sft_v2_local.ps1` | 合并蒸馏 + 第二轮 SFT |
| `local/run_dpo_clean_local.ps1` | 干净 DPO（实验） |

`scripts/run_*.ps1` 会转发到 `local/`。

## 词表

| 脚本 | 作用 |
|------|------|
| `train_tokenizer_v1.py` | 训练 `tokenizer/v1/`（已生成可跳过） |

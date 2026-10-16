# 本地工作流（v1 87M）

## 环境

```powershell
conda activate BigStrongGPT
cd <repo>
$env:PYTHONPATH = "src;."
```

## 推荐权重（当前）

| 用途 | 路径 |
|------|------|
| **对话 / Demo** | `output/v1/sft/last.pt`（云上第一轮 SFT） |
| Pretrain 基座 | `output/v1/pretrain/last.pt` |
| 实验对比 | `output/v1/sft_v2/last.pt` |
| DPO 实验（可选） | `output/v1/dpo_clean/last.pt`（见下） |
| 勿用 | `output/v1/dpo/last.pt`（云上脏 DPO）；`smoke_*` 冒烟权重本机不保留 |

权重与完整 `dataset/*.jsonl` 不进 git；**预训练 / SFT / DPO / 蒸馏各用了啥**见 [dataset/README.md](../dataset/README.md)。

## 评测

```powershell
python scripts/eval_v1.py --ckpt output/v1/sft/last.pt
# 默认贪心 + 64 token；要采样：加 --sample
```

```powershell
python scripts/compare_sft_ckpts.py
python scripts/compare_qwen_sft.py
```

（实现位于 `scripts/benchmark/`，根目录同名脚本为兼容入口。）

## 蒸馏 → 第二轮 SFT（可选）

```powershell
.\scripts\run_distill_local.ps1
# 或 .\scripts\local\run_distill_local.ps1
.\scripts\run_sft_v2_local.ps1
```

## 数据

```powershell
python scripts/download_datasets.py --merge-sft-v2
python scripts/download_datasets.py --dpo-clean
```

详见 [DATA_SOURCES.md](DATA_SOURCES.md)。

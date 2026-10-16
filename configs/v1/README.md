# v1 训练配置

| 文件 | 阶段 | 说明 |
|------|------|------|
| `pretrain.yaml` | Pretrain | `dataset/pretrain_skypile.jsonl` |
| `sft.yaml` | SFT 第一轮 | `sft_merged.jsonl`，从 pretrain 初始化 |
| `sft_v2.yaml` | SFT 第二轮 | `sft_merged_v2.jsonl`，从 `output/v1/sft/last.pt` |
| `dpo_clean.yaml` | DPO（实验） | `dpo_clean.jsonl`，无 btf |
| `dpo.yaml` | DPO（弃用） | `dpo_merged` 含 btf，勿用 |
| `smoke_*.yaml` | 冒烟 | 小数据、少 step |

统一入口：

```powershell
python -m bigstrong.train --config configs/v1/<file>.yaml
```

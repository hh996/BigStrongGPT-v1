# BigStrongGPT v1

**仓库**：<https://github.com/hh996/BigStrongGPT-v1>（v1 唯一维护线）

中文 Decoder-Only 小语言模型（约 **87M**：768×12、GQA、词表 16K）。  
训练链路：**Pretrain → SFT →（可选）7B 蒸馏 / 第二轮 SFT**。

## 目录结构

```
├── src/bigstrong/          # 模型与训练（python -m bigstrong.train）
├── configs/v1/             # pretrain / sft / sft_v2 / smoke / dpo_clean
├── tokenizer/v1/           # ChatML 词表（进 git）
├── dataset/lm_dataset.py   # Dataset 定义
│   ├── README.md           # **训练各阶段用了啥数据（先看这个）**
│   ├── samples/            # 按阶段拆开的小样（可进 git）
│   └── *.jsonl             # 完整数据本机备份，不进 git
│   └── *.jsonl             # 完整数据本地复制，不进 git
├── scripts/
│   ├── download_datasets.py
│   ├── generate_distill.py
│   ├── benchmark/          # eval、对比 ckpt、对比 Qwen
│   └── local/              # Windows 一键脚本
├── docs/LOCAL.md           # 本地工作流
└── output/v1/              # 权重（gitignore，自行备份）
```

## 环境

```powershell
conda create -n BigStrongGPT python=3.11
conda activate BigStrongGPT
pip install -r requirements.txt
# 另需 PyTorch（CUDA）、bitsandbytes（7B 蒸馏 4bit）
```

```powershell
cd <repo>
$env:PYTHONPATH = "src;."
```

## 推荐权重

| 用途 | 路径 |
|------|------|
| **对话 / 评测** | `output/v1/sft/last.pt` |
| Pretrain | `output/v1/pretrain/last.pt` |

评测（默认贪心、64 token）：

```powershell
python scripts/eval_v1.py --ckpt output/v1/sft/last.pt
```

## 训练

```powershell
python -m bigstrong.train --config configs/v1/pretrain.yaml
python -m bigstrong.train --config configs/v1/sft.yaml
```

数据：`python scripts/download_datasets.py --help`  
蒸馏：`.\scripts\run_distill_local.ps1`  
详见 [docs/LOCAL.md](docs/LOCAL.md)、[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)、[docs/REPOSITORY.md](docs/REPOSITORY.md)（迁数据与旧仓说明）。

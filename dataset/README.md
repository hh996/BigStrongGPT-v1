# 训练数据说明（看这个）

`dataset/` 里大文件 `*.jsonl` 是**本机备份**（不进 Git）。  
想**肉眼看一眼**每条长什么样 → 打开 **`samples/`** 下对应子目录的小文件。

---

## 一张表：阶段 → 用了啥 → 哪个文件 → 哪个权重

| 阶段 | 在哪里训的 | 实际用的数据（来源） | 本目录完整文件 | 对应 checkpoint |
|------|------------|----------------------|----------------|-----------------|
| **预训练** | **云上 AutoDL** | [SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) 中文 head，多 shard、远多于本地 | `pretrain_skypile.jsonl`（**小样**，见下） | `output/v1/pretrain/last.pt` |
| **SFT 第一轮** | **云上** | Belle 3 万 + alpaca-zh 2 万 → 合并去重 | `sft_merged.jsonl`（≈5 万） | **`output/v1/sft/last.pt`（主用）** |
| **DPO（云上）** | **云上** | shibing 中文偏好 5k + **btf 中文 DPO 5k**（脏） | `dpo_merged.jsonl`（1 万） | `output/v1/dpo/last.pt`（**不推荐**） |
| **蒸馏** | **本地** | 从 `sft_merged` 抽用户问题，**Qwen2.5-7B-Instruct** 生成回答 | `sft_distill_7b.jsonl`（2 万） | 不单独出 ckpt |
| **SFT 第二轮** | **本地** | `sft_merged` + `sft_distill_7b` 合并去重 | `sft_merged_v2.jsonl`（≈7 万） | `output/v1/sft_v2/last.pt`（不如 v1 SFT） |
| **DPO 干净** | **本地** | shibing 中文 + enthusiastic，**无 btf**，过滤脏话/过长 | `dpo_clean.jsonl`（≈1.2 万） | `output/v1/dpo_clean/last.pt`（实验，见下） |

**`dpo_clean` 是啥？** 在 **`sft/last.pt` 上**做的一轮 **DPO（偏好对齐）**：数据是 `dpo_clean.jsonl`（没有云上那种 btf 辱骂当「好答案」）。目的是让模型更倾向 `chosen`、远离 `rejected`。你这边实测 **不如直接用 SFT**，所以 Demo 仍用 `sft/last.pt`；`dpo_clean/` 只是留档对比。

**`smoke_*`**：`configs/v1/smoke_*.yaml` 跑通的测试权重，**本机 output 不保留**（省约 2.5GB）。

本地调试预训练还可看 `pretrain_wikipedia.jsonl`（**云上正式 pretrain 不用它**）。

---

## ① 预训练（云上很大，本地只是一口「同味道」）

**云上做了什么**

- 数据：**SkyPile-150B** 里中文 head 的 jsonl shard（`2021-43_zh_head_*.jsonl`），按文档计划是 **多个 shard（约 5～10+）**，一直训到 `pretrain/last.pt` 里约 **step 818670**。
- 配置：`configs/v1/pretrain.yaml` → 读 `dataset/pretrain_skypile.jsonl`（云上会把下载好的 SkyPile 合成这个文件名或等价路径）。
- 格式：一行一个 JSON，`{"text": "正文..."}`。

**本地 `pretrain_skypile.jsonl` 是啥**

- 同一个 HF 数据集，脚本只下了 **前 2 个 shard**，并且 **最多 10 万条**（调试上限）。
- 和云上**种类一样**（都是 SkyPile 中文网页文本），**量差很多**；理解「预训练在吃啥」看本地这份 + `samples/01_pretrain/` 就够。
- 原始单 shard 约 2GB，没整包拷进 v1；只有 `samples/01_pretrain/skypile_hf_shard_raw_head10.jsonl` 几行原始格式。

**怎么看**

```text
samples/01_pretrain/skypile_head50.jsonl          # 前 50 条（与 pretrain_skypile 相同）
samples/01_pretrain/skypile_random200.jsonl       # 随机 200 条（更有代表性）
samples/01_pretrain/skypile_hf_shard_raw_head10.jsonl  # HF 原始 shard 字段
```

---

## ② SFT 第一轮（云上，当前最好对话模型）

**配方**

1. **BelleGroup/train_0.5M_CN** 抽 **30 000** 条 → `sft_belle.jsonl`
2. **shibing624/alpaca-zh** 全量约 **20 000** 条 → `sft_alpaca_zh.jsonl`
3. 脚本 `merge_sft_files()`：所有 `sft_*.jsonl`（除 `sft_merged` 自己）合并、**按整行去重** → **`sft_merged.jsonl`（49 995 条）**

**训练**

- `configs/v1/sft.yaml` → `dataset/sft_merged.jsonl`
- 从 `pretrain/last.pt` 继续，**2 epoch**

**注意**：第一轮云上 SFT **没有**用 7B 蒸馏；蒸馏是后来在本地做的。

**怎么看**

```text
sft_belle.jsonl / sft_alpaca_zh.jsonl   # 两个原料
sft_merged.jsonl                        # 真正喂给第一轮 SFT 的
samples/02_sft/belle_head15.jsonl
samples/02_sft/alpaca_zh_head15.jsonl
samples/02_sft/merged_head20.jsonl
```

每行格式：`{"conversations": [{"content": "用户"}, {"content": "助手"}]}`

---

## ③ DPO

### 云上（脏，已弃用）

1. **shibing624/DPO-En-Zh-20k-Preference**（zh）→ **5 000** 条 → `dpo_zh.jsonl`
2. **Karsh-CAI/btfChinese-DPO-small** → **5 000** 条 → `dpo_btf_small.jsonl`
3. 合并 → **`dpo_merged.jsonl`（10 000 条）** → 云上 `configs/v1/dpo.yaml` → `output/v1/dpo/last.pt`

btf 里不少 **chosen 带辱骂**，小模型容易被带偏，**不要再用这条线**。

### 本地（干净，实验）

- `python scripts/download_datasets.py --dpo-clean` → **`dpo_clean.jsonl`**
- 来源：**shibing 中文偏好** + **markyfsun/chinese-enthusiastic-dpo**，过滤脏话、重复、过长
- `configs/v1/dpo_clean.yaml` → `output/v1/dpo_clean/last.pt`

**怎么看**

```text
samples/03_dpo/dpo_cloud_dirty_head15.jsonl   # 云上那条（对照）
samples/03_dpo/dpo_clean_head15.jsonl       # 本地干净 DPO
```

每行：`{"prompt": "...", "chosen": "...", "rejected": "..."}`

---

## ④ 蒸馏（给第二轮 SFT 用，不是云上 SFT）

| 项目 | 内容 |
|------|------|
| 脚本 | `scripts/generate_distill.py`（或 `run_distill_local.ps1`） |
| 老师 | **Qwen/Qwen2.5-7B-Instruct**（本地 4bit） |
| 输入 | `sft_merged.jsonl` 里每条对话的**用户问题**（本机跑了 **20 000** 条） |
| 输出 | **`sft_distill_7b.jsonl`**：用户问题 + 7B 写的助手回复 |

**怎么看**：`samples/04_distill/distill_7b_head15.jsonl`

---

## ⑤ 第二轮 SFT（本地可选）

- `python scripts/download_datasets.py --merge-sft-v2`
- **`sft_merged.jsonl` + `sft_distill_7b.jsonl`** 去重 → **`sft_merged_v2.jsonl`（69 941 条）**
- `configs/v1/sft_v2.yaml` → `output/v1/sft_v2/last.pt`  
- 实测不如第一轮 **`sft/last.pt`**，演示仍用第一轮。

---

## 根目录每个 `.jsonl` 速查

| 文件 | 一句话 |
|------|--------|
| `pretrain_skypile.jsonl` | SkyPile 中文，本地 10 万条小样（云上同源、更大量） |
| `pretrain_wikipedia.jsonl` | 维基中文，本地调试用 |
| `sft_belle.jsonl` | SFT 原料：Belle 3 万 |
| `sft_alpaca_zh.jsonl` | SFT 原料：alpaca-zh 2 万 |
| `sft_merged.jsonl` | **第一轮 SFT 训练集** |
| `sft_distill_7b.jsonl` | **7B 蒸馏 2 万** |
| `sft_merged_v2.jsonl` | 第二轮 SFT 训练集 |
| `dpo_merged.jsonl` | **云上 DPO（含 btf，弃用）** |
| `dpo_clean.jsonl` | **本地干净 DPO** |

更细的字段说明见 [DATA_MANIFEST.md](DATA_MANIFEST.md)（偏运维清单）。

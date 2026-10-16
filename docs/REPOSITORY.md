# 仓库说明（仅维护本仓）

**正式仓库**：<https://github.com/hh996/BigStrongGPT-v1>

旧 monorepo `hh996/BigStrongGPT`（含 `legacy/v0_25m`）不再作为 v1 开发主线；新改动只在本仓提交、推送。

## 日常开发

```powershell
git clone https://github.com/hh996/BigStrongGPT-v1.git
cd BigStrongGPT-v1
conda activate BigStrongGPT
$env:PYTHONPATH = "src;."
```

PyCharm / Cursor：**打开本目录**为项目根，不要再用旧 `BigStrongGPT` 路径做 git 提交。

## 从旧本机目录迁数据（一次性）

若你之前在 `PycharmProjects\BigStrongGPT` 里已有权重和数据，**复制**到本仓即可（不进 git）：

| 源（旧目录） | 目标（本仓） |
|--------------|--------------|
| `output/v1/` | `output/v1/` |
| `dataset/*.jsonl` | `dataset/` |
| `.env`（若有 SwanLab） | `.env` |

旧目录可保留作备份，或改名 `BigStrongGPT-archive`；不必再 `git push` 到 `BigStrongGPT`。

## 提交前检查

```powershell
git status
```

勿提交：`.env`、`output/`、`models/`、`*.pt`、`dataset/*.jsonl`、`dataset/*.log`、`_gh/`。

## 旧仓 `export_v1_repo.ps1`

`scripts/export_v1_repo.ps1` 仍保留：若有人本地还握着旧大仓，可导出到空目录再与本远程对比。你已是本仓克隆时**不需要**再导出。

## 旧 GitHub 仓库怎么处理（可选）

在 <https://github.com/hh996/BigStrongGPT> 任选其一：

1. **Archive**（Settings → Archive）：只读，历史保留。  
2. **改 README**：顶部写一句「v1 见 [BigStrongGPT-v1](https://github.com/hh996/BigStrongGPT-v1)」。  
3. 什么都不做：远程停在旧提交，本机只推 v1 仓。

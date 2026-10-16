# 仓库怎么放、要不要新建仓库

## 方案 A：继续用本仓库 `BigStrongGPT`（推荐）

单仓多代，边界已整理：

| 目录 | 内容 |
|------|------|
| `src/bigstrong/` | v1 正式代码 |
| `scripts/benchmark/` | 评测与对比（实现） |
| `scripts/local/` | Windows 一键 |
| `scripts/*.py` 根下短文件 | **兼容入口**，转发到 `benchmark/` |
| `legacy/v0_25m/` | 25M，不扩展 |

大文件仍在 `.gitignore`：`output/`、`models/`、`dataset/*.jsonl`、`.env`、`笔记.md`。

## 方案 B：导出 v1 到新目录 / 新 Git 仓库

适合：对外只展示 v1、去掉 legacy/RAG。

```powershell
cd C:\Users\haohuang\PycharmProjects\BigStrongGPT
.\scripts\export_v1_repo.ps1
# 默认: ..\BigStrongGPT-v1

# 或指定路径
.\scripts\export_v1_repo.ps1 -Dest D:\repos\BigStrongGPT-v1
```

导出内容：**不含** `legacy/`、`笔记.md`、权重与 jsonl；**含** `README.v1.md` → 复制为 `README.md`。

新目录初始化：

```powershell
cd ..\BigStrongGPT-v1
git init -b main
git add -A
git status   # 确认无 .env、无 *.pt
git commit -m "Initial commit: BigStrongGPT v1 pipeline"
git remote add origin <你的远程 URL>
git push -u origin main
```

本仓库可保留，在 README 里写一句：「v1 独立仓：xxx」。

## 提交前检查

```powershell
git status
```

勿提交：`.env`、`笔记.md`、`output/`、`*.pt`、`dataset/*.jsonl`。

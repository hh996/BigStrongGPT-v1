# 仓库根目录 + Python 环境（Windows 本地一键脚本共用）
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$script:BigStrongPy = "C:\Users\haohuang\miniconda3\envs\BigStrongGPT\python.exe"
if (-not (Test-Path $script:BigStrongPy)) {
    $script:BigStrongPy = "python"
}

$env:PYTHONPATH = "src;."
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
chcp 65001 | Out-Null

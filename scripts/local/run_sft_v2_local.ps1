. "$PSScriptRoot\_common.ps1"

$sftCkpt = Join-Path $RepoRoot "output\v1\sft\last.pt"
if (-not (Test-Path $sftCkpt)) {
    Write-Host "缺少 $sftCkpt（第一轮 SFT 权重）"
    exit 1
}

Write-Host "=== 1/2 合并 sft_merged_v2.jsonl ==="
& $BigStrongPy scripts\download_datasets.py --merge-sft-v2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 2/2 第二轮 SFT ==="
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "output\v1\sft_v2") | Out-Null
& $BigStrongPy -m bigstrong.train --config configs/v1/sft_v2.yaml 2>&1 |
    Tee-Object -FilePath (Join-Path $RepoRoot "output\v1\sft_v2\train.log")
exit $LASTEXITCODE

. "$PSScriptRoot\_common.ps1"

$sftCkpt = Join-Path $RepoRoot "output\v1\sft\last.pt"
$sftDownload = "C:\Users\haohuang\Downloads\last (1).pt"
if (-not (Test-Path $sftCkpt)) {
    if (Test-Path $sftDownload) {
        New-Item -ItemType Directory -Force -Path (Split-Path $sftCkpt) | Out-Null
        Copy-Item $sftDownload $sftCkpt
        Write-Host "已复制 SFT 权重 -> $sftCkpt"
    } else {
        Write-Host "缺少 $sftCkpt"
        exit 1
    }
}

Write-Host "=== 1/2 构建 dpo_clean.jsonl ==="
& $BigStrongPy scripts\download_datasets.py --dpo-clean
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 2/2 DPO clean ==="
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "output\v1\dpo_clean") | Out-Null
& $BigStrongPy -m bigstrong.train --config configs/v1/dpo_clean.yaml 2>&1 |
    Tee-Object -FilePath (Join-Path $RepoRoot "output\v1\dpo_clean\train.log")
exit $LASTEXITCODE

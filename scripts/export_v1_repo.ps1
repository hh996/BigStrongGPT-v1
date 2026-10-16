# Export v1-only tree for a new git repo (no legacy, no large artifacts).
param(
    [string]$Dest = (Join-Path (Split-Path $PSScriptRoot -Parent) "BigStrongGPT-v1")
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dest = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Dest)

if (Test-Path $Dest) {
    Write-Host "Dest already exists: $Dest"
    Write-Host "Remove it or pass -Dest to another path."
    exit 1
}

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Copy-Tree($Rel) {
    $src = Join-Path $RepoRoot $Rel
    $dst = Join-Path $Dest $Rel
    if (-not (Test-Path $src)) { return }
    $parent = Split-Path $dst -Parent
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    Copy-Item -Path $src -Destination $dst -Recurse -Force
}

$paths = @(
    "src",
    "configs\v1",
    "tokenizer\v1",
    "scripts\benchmark",
    "scripts\local",
    "scripts\download_datasets.py",
    "scripts\generate_distill.py",
    "scripts\train_tokenizer_v1.py",
    "scripts\_shim.py",
    "scripts\eval_v1.py",
    "scripts\compare_sft_ckpts.py",
    "scripts\compare_qwen_sft.py",
    "scripts\run_distill_local.ps1",
    "scripts\run_sft_v2_local.ps1",
    "scripts\run_dpo_clean_local.ps1",
    "scripts\export_v1_repo.ps1",
    "scripts\README.md",
    "docs\LOCAL.md",
    "docs\DATA_SOURCES.md",
    "docs\REPOSITORY.md",
    "requirements.txt"
)
foreach ($rel in $paths) { Copy-Tree $rel }

New-Item -ItemType Directory -Force -Path (Join-Path $Dest "dataset") | Out-Null
Copy-Item (Join-Path $RepoRoot "dataset\lm_dataset.py") (Join-Path $Dest "dataset\lm_dataset.py") -Force
if (-not (Test-Path (Join-Path $Dest "dataset\__init__.py"))) {
    New-Item -ItemType File -Path (Join-Path $Dest "dataset\__init__.py") -Force | Out-Null
}

Copy-Item (Join-Path $RepoRoot ".gitignore") (Join-Path $Dest ".gitignore") -Force
$readmeV1 = Join-Path $RepoRoot "README.v1.md"
if (Test-Path $readmeV1) {
    Copy-Item $readmeV1 (Join-Path $Dest "README.md") -Force
} else {
    Copy-Item (Join-Path $RepoRoot "README.md") (Join-Path $Dest "README.md") -Force
}

Write-Host "Exported to: $Dest"

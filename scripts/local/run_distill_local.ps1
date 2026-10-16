. "$PSScriptRoot\_common.ps1"

$modelDir = "models\qwen2.5-7b-instruct\models\Qwen--Qwen2.5-7B-Instruct\snapshots\master"
$consoleLog = Join-Path $RepoRoot "dataset\distill_gen.log"

& $BigStrongPy scripts\generate_distill.py `
  --model-dir $modelDir `
  --max-samples 20000 `
  --resume `
  *>&1 | Tee-Object -FilePath $consoleLog -Append

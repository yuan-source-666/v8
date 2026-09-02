# v8 一键启动脚本（Windows PowerShell）：先 prepare 后 train
# 用法（在项目根目录右键“使用 PowerShell 运行”，或命令行）：
#   powershell -ExecutionPolicy Bypass -File scripts/run_mini.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$config = $env:V8_CONFIG
if (-not $config) { $config = "config/mini.yaml" }
$profile = $env:V8_PROFILE
if (-not $profile) { $profile = "mini" }

# 已安装依赖检查
python -c "import torch, tiktoken, numpy, yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[run_mini] 首次运行：安装依赖（CPU 版 torch）..." -ForegroundColor Cyan
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { python -m pip install torch }
    python -m pip install tiktoken numpy pyyaml intel-extension-for-pytorch
}

Write-Host "[run_mini] —— 第 1 步：数据管线（prepare）——" -ForegroundColor Cyan
python data/prepare.py --config $config --profile $profile --demo
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[run_mini] —— 第 2 步：预训练（train，启用 drives）——" -ForegroundColor Cyan
python train/train.py --config $config --profile $profile --use-drives

Write-Host "[run_mini] 完成。可继续：SFT / DPO / 采样" -ForegroundColor Green

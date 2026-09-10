# Headroom Experiment — two arms, run sequentially
# From repo root: .\dev_cut_lagrangian\run_headroom_exp.ps1
#
# Arm 1: plain LR in-loop, initial UB = floor(LR_200_cold)
# Arm 2: cutLR_triple in-loop, initial UB = floor(LR_200_cold)
#
# Expected runtime: ~1.5-2h per arm (3 workers, 54 instances)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "=== Headroom Experiment ===" -ForegroundColor Cyan
Write-Host "Subset: data\stratified54_list.csv (54 instances)"
Write-Host "Initial UB: floor(LR_200_cold) from data\headroom_weak_ubs.csv"
Write-Host ""

Write-Host "--- ARM 1: LR in-loop ---" -ForegroundColor Yellow
python run_adaptive_full_cutlag_exp_monotone.py `
    --subset data/stratified54_list.csv `
    --weak-ub-csv data/headroom_weak_ubs.csv `
    --bound-mode lag `
    --save-suffix "_headroom_lr" `
    --tag "headroom_lr"

Write-Host ""
Write-Host "--- ARM 2: cutLR_triple in-loop ---" -ForegroundColor Yellow
python run_adaptive_full_cutlag_exp_monotone.py `
    --subset data/stratified54_list.csv `
    --weak-ub-csv data/headroom_weak_ubs.csv `
    --bound-mode cutlag_triple `
    --save-suffix "_headroom_cutlag" `
    --tag "headroom_cutlag"

Write-Host ""
Write-Host "=== Both arms complete ===" -ForegroundColor Green
Write-Host "Run analysis with:"
Write-Host "  python dev_cut_lagrangian/analyze_headroom_exp.py \"
Write-Host "    --lr-csv  results/adaptive_full_*_headroom_lr.csv \"
Write-Host "    --cut-csv results/adaptive_full_*_headroom_cutlag.csv"

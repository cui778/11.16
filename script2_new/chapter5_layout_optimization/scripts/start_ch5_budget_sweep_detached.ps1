param(
    [string]$Manifest = "E:\11.16\script2_new\chapter5_layout_optimization\outputs\budget_sweep_formal\CH5_budget_sweep_manifest_s42.csv",
    [int]$SleepSeconds = 60
)

$ErrorActionPreference = "Stop"

$python = "D:\conda3\envs\swmm_gpu\python.exe"
$runner = "E:\11.16\script2_new\chapter5_layout_optimization\scripts\run_ch5_budget_sweep.py"
$outputDir = Split-Path -Parent $Manifest
$launcherLogDir = Join-Path $outputDir "launcher_logs"
New-Item -ItemType Directory -Force -Path $launcherLogDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $launcherLogDir "budget_sweep_$timestamp.stdout.log"
$stderrLog = Join-Path $launcherLogDir "budget_sweep_$timestamp.stderr.log"
$pidFile = Join-Path $launcherLogDir "budget_sweep_$timestamp.pid"

$argumentList = @(
    "`"$runner`"",
    "--manifest", "`"$Manifest`"",
    "--sleep", "$SleepSeconds"
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $argumentList `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

Set-Content -Path $pidFile -Value $process.Id -Encoding ASCII

Write-Host "Started detached budget sweep."
Write-Host "PID: $($process.Id)"
Write-Host "stdout: $stdoutLog"
Write-Host "stderr: $stderrLog"
Write-Host "per-job logs: $(Join-Path $outputDir 'logs')"

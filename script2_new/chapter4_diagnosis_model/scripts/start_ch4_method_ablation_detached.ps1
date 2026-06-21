param(
    [ValidateSet("smoke", "seed42", "confirm", "all-seeds", "summarize")]
    [string]$Mode = "seed42",
    [int]$CooldownSeconds = 60
)

$ErrorActionPreference = "Stop"

$python = "D:\conda3\envs\swmm_gpu\python.exe"
$launcher = "E:\11.16\script2_new\chapter4_diagnosis_model\scripts\launch_ch4_method_ablation_detached.py"

& $python $launcher `
    --mode $Mode `
    --cooldown-seconds $CooldownSeconds

if ($LASTEXITCODE -ne 0) {
    throw "Detached launcher failed with exit code $LASTEXITCODE"
}

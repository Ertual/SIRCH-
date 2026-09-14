param(
    [string]$TrainingRoot = "C:\SIRCH_ENV\models\revision\exp13_grouped_retraining",
    [string]$EvaluationCacheRoot = "C:\SIRCH_ENV\models\revision\exp14_grouped_factorial_evaluation",
    [string]$Python = "C:\SIRCH_ENV\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TrainingStatus = Join-Path $TrainingRoot "pipeline_status.json"
$QueueStatus = Join-Path $EvaluationCacheRoot "queued_evaluation_status.json"
$EvaluationScript = Join-Path $PSScriptRoot "run_grouped_evaluation.py"

New-Item -ItemType Directory -Path $EvaluationCacheRoot -Force | Out-Null

function Write-QueueStatus {
    param([hashtable]$Payload)
    $temporary = "$QueueStatus.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -Path $temporary -Encoding utf8
    Move-Item -Path $temporary -Destination $QueueStatus -Force
}

Write-QueueStatus @{
    status = "waiting_for_training"
    started_utc = [DateTime]::UtcNow.ToString("o")
    training_status = $TrainingStatus
    evaluation_script = $EvaluationScript
    pid = $PID
}

while ($true) {
    if (-not (Test-Path $TrainingStatus)) {
        Start-Sleep -Seconds 30
        continue
    }
    $training = Get-Content $TrainingStatus -Raw | ConvertFrom-Json
    if ($training.status -eq "complete") {
        break
    }
    if ($training.status -eq "error") {
        Write-QueueStatus @{
            status = "blocked_by_training_error"
            updated_utc = [DateTime]::UtcNow.ToString("o")
            training_error_type = $training.error_type
            training_error = $training.error
        }
        exit 2
    }
    $trainingProcess = Get-Process -Id ([int]$training.pid) -ErrorAction SilentlyContinue
    if ($null -eq $trainingProcess) {
        Write-QueueStatus @{
            status = "blocked_by_stopped_training_process"
            updated_utc = [DateTime]::UtcNow.ToString("o")
            training_pid = $training.pid
            training_pipeline_status = $training.status
        }
        exit 3
    }
    Start-Sleep -Seconds 30
}

Write-QueueStatus @{
    status = "running_evaluation"
    started_utc = [DateTime]::UtcNow.ToString("o")
    training_status = "complete"
    evaluation_script = $EvaluationScript
    pid = $PID
}

$env:SIRCH_EXP13_MODEL_DIR = $TrainingRoot
$env:SIRCH_EXP14_CACHE_DIR = $EvaluationCacheRoot
Push-Location $ProjectRoot
try {
    & $Python $EvaluationScript
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation exited with code $LASTEXITCODE"
    }
    Write-QueueStatus @{
        status = "complete"
        completed_utc = [DateTime]::UtcNow.ToString("o")
        evaluation_script = $EvaluationScript
    }
}
catch {
    Write-QueueStatus @{
        status = "error"
        updated_utc = [DateTime]::UtcNow.ToString("o")
        error = $_.Exception.Message
    }
    throw
}
finally {
    Pop-Location
}

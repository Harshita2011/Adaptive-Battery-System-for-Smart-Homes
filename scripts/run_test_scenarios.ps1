param(
    [string]$BaseUrl = "http://localhost:8000",
    [int]$Repeats = 5,
    [string[]]$ControllerModes = @("rule_only", "fuzzy_only", "rnn_fuzzy", "rnn_fuzzy_ga"),
    [int]$NormalSteps = 40,
    [int]$HeatwaveSteps = 40,
    [int]$SpikeLowSteps = 20,
    [int]$SpikeHighSteps = 12,
    [int]$SpikeRecoverSteps = 20,
    [int]$StaleSteps = 20
)

$ErrorActionPreference = "Stop"

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path "data\\eval" $runId
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$csvPathAbs = Join-Path (Resolve-Path $outDir) "scenario_log.csv"

$rows = New-Object System.Collections.Generic.List[object]
$gridPrices = @(0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.3,0.4,0.5,0.6,0.8,0.7,0.4,0.2,0.1,0.1)

function Get-PriceNow {
    $h = (Get-Date).Hour
    return [double]$gridPrices[$h]
}

function Get-GitCommit {
    try {
        return (git rev-parse --short HEAD 2>$null).Trim()
    } catch {
        return "unknown"
    }
}

function Get-ApplianceKW([object]$statusObj) {
    $total = 0.0
    if ($statusObj.appliances) {
        foreach ($a in $statusObj.appliances) {
            $v = 0.0
            [void][double]::TryParse([string]$a.usage, [ref]$v)
            $total += $v
        }
    }
    return [Math]::Round($total, 4)
}

function Run-Step {
    param(
        [string]$ExperimentId,
        [int]$RepeatId,
        [int]$RunSeed,
        [string]$ControllerMode,
        [string]$ScenarioName,
        [int]$Step,
        [double]$ManualLoad,
        [switch]$SendTelemetry,
        [double]$TelemetryTemp = 0.0
    )

    if ($SendTelemetry) {
        $body = @{
            current = $ManualLoad
            temperature = $TelemetryTemp
            soh = 0.985
            timestamp = (Get-Date).ToUniversalTime().ToString("o")
        } | ConvertTo-Json
        Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/telemetry" -ContentType "application/json" -Body $body | Out-Null
    }

    $status = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/system-status?manual_load=$ManualLoad"
    $dispatch = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/control/dispatch" -ContentType "application/json" -Body '{"dry_run":true,"optimization_enabled":true}'

    $dispatchCount = 0
    $turnOffCount = 0
    if ($dispatch.commands) {
        $dispatchCount = @($dispatch.commands).Count
        foreach ($c in @($dispatch.commands)) {
            if ($c.command -eq "turn_off") { $turnOffCount++ }
        }
    }

    $tempC = [double]$status.metrics.temp
    $health = [double]$status.metrics.health
    $limitText = [string]$status.metrics.limit
    $limitNum = [double]($limitText -replace "%", "")
    $applianceKW = Get-ApplianceKW -statusObj $status
    $priceNow = Get-PriceNow
    $costIndex = [Math]::Round($applianceKW * $priceNow, 6)

    $rows.Add([PSCustomObject]@{
        run_id = $runId
        experiment_id = $ExperimentId
        repeat_id = $RepeatId
        run_seed = $RunSeed
        controller_mode = $ControllerMode
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        scenario = $ScenarioName
        step = $Step
        mode = [string]$status.mode
        source = [string]$status.source
        telemetry_fresh = [bool]$status.telemetry_fresh
        load_a = [double]($status.metrics.load -replace "A","")
        temp_c = $tempC
        health = $health
        status = [string]$status.metrics.status
        fuzzy_limit_pct = $limitNum
        dispatch_commands = $dispatchCount
        turn_off_commands = $turnOffCount
        appliance_total_kw = $applianceKW
        grid_price = $priceNow
        energy_cost_index = $costIndex
        action_log_size = @($status.recent_actions).Count
    })
}

function Run-ScenarioSet {
    param(
        [string]$ExperimentId,
        [int]$RepeatId,
        [int]$RunSeed,
        [string]$ControllerMode
    )

    Write-Host "[scenario][$ControllerMode][repeat=$RepeatId] Normal load"
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=SIM" | Out-Null
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/set-scenario?name=NORMAL" | Out-Null
    for ($i=0; $i -lt $NormalSteps; $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "normal_load" -Step $i -ManualLoad 18.0
        Start-Sleep -Milliseconds 200
    }

    Write-Host "[scenario][$ControllerMode][repeat=$RepeatId] Heatwave high temperature"
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=SIM" | Out-Null
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/set-scenario?name=HEATWAVE" | Out-Null
    for ($i=0; $i -lt $HeatwaveSteps; $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "heatwave_high_temp" -Step $i -ManualLoad 22.0
        Start-Sleep -Milliseconds 200
    }

    Write-Host "[scenario][$ControllerMode][repeat=$RepeatId] Sudden load spike (LIVE telemetry)"
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=LIVE" | Out-Null
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/set-scenario?name=NORMAL" | Out-Null
    for ($i=0; $i -lt $SpikeLowSteps; $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "sudden_spike" -Step $i -ManualLoad 14.0 -SendTelemetry -TelemetryTemp 36.0
        Start-Sleep -Milliseconds 200
    }
    for ($i=$SpikeLowSteps; $i -lt ($SpikeLowSteps + $SpikeHighSteps); $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "sudden_spike" -Step $i -ManualLoad 30.0 -SendTelemetry -TelemetryTemp 54.0
        Start-Sleep -Milliseconds 200
    }
    for ($i=($SpikeLowSteps + $SpikeHighSteps); $i -lt ($SpikeLowSteps + $SpikeHighSteps + $SpikeRecoverSteps); $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "sudden_spike" -Step $i -ManualLoad 16.0 -SendTelemetry -TelemetryTemp 38.0
        Start-Sleep -Milliseconds 200
    }

    Write-Host "[scenario][$ControllerMode][repeat=$RepeatId] Missing / stale telemetry fallback"
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=LIVE" | Out-Null
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/set-scenario?name=NORMAL" | Out-Null
    Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "missing_stale_telemetry" -Step 0 -ManualLoad 20.0 -SendTelemetry -TelemetryTemp 41.0
    for ($i=1; $i -lt $StaleSteps; $i++) {
        Run-Step -ExperimentId $ExperimentId -RepeatId $RepeatId -RunSeed $RunSeed -ControllerMode $ControllerMode -ScenarioName "missing_stale_telemetry" -Step $i -ManualLoad 20.0
        Start-Sleep -Milliseconds 750
    }
}

$gitCommit = Get-GitCommit
Write-Host "[info] run_id=$runId commit=$gitCommit repeats=$Repeats modes=$($ControllerModes -join ',')"

foreach ($mode in $ControllerModes) {
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/control/mode?name=$mode" | Out-Null

    for ($r=1; $r -le $Repeats; $r++) {
        $currentExp = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/experiments/current"
        if ($currentExp.active_experiment_id) {
            Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/experiments/end?status=aborted" | Out-Null
        }

        $seed = 1000 + $r
        $experimentId = "$runId`_$mode`_r$('{0:D2}' -f $r)"
        $startPayload = @{
            experiment_id = $experimentId
            run_seed = $seed
            metadata = @{
                run_id = $runId
                repeat_id = $r
                controller_mode = $mode
                git_commit = $gitCommit
            }
        } | ConvertTo-Json -Depth 6

        Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/experiments/start" -ContentType "application/json" -Body $startPayload | Out-Null
        try {
            Run-ScenarioSet -ExperimentId $experimentId -RepeatId $r -RunSeed $seed -ControllerMode $mode
            Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/experiments/end?status=completed" | Out-Null
        } catch {
            Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/experiments/end?status=failed" | Out-Null
            throw
        }
    }
}

$rows.ToArray() | Export-Csv -Path $csvPathAbs -NoTypeInformation

$meta = [PSCustomObject]@{
    run_id = $runId
    git_commit = $gitCommit
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    repeats = $Repeats
    controller_modes = ($ControllerModes -join ",")
    normal_steps = $NormalSteps
    heatwave_steps = $HeatwaveSteps
    spike_low_steps = $SpikeLowSteps
    spike_high_steps = $SpikeHighSteps
    spike_recover_steps = $SpikeRecoverSteps
    stale_steps = $StaleSteps
}
$metaPath = Join-Path (Resolve-Path $outDir) "run_metadata.json"
$meta | ConvertTo-Json -Depth 5 | Set-Content $metaPath

Write-Host "[done] Scenario log written: $csvPathAbs"
Write-Host "[done] Metadata written: $metaPath"
Write-Host "[next] Run: python scripts/evaluate_scenarios.py --input `"$csvPathAbs`""

param(
    [string]$BaseUrl = "http://localhost:8000",
    [double]$Current = 18.0,
    [double]$Temperature = 40.0,
    [double]$Voltage = 48.0,
    [double]$Soc = 0.75,
    [double]$Soh = 0.995,
    [int]$IntervalMs = 1000,
    [switch]$Randomize,
    [double]$CurrentJitter = 1.5,
    [double]$TempJitter = 1.2,
    [switch]$SetLiveMode
)

$ErrorActionPreference = "Stop"

function Get-RandomFloat([double]$Center, [double]$Amplitude) {
    return $Center + ((Get-Random -Minimum -1000 -Maximum 1001) / 1000.0) * $Amplitude
}

if ($SetLiveMode) {
    try {
        Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=LIVE" | Out-Null
        Write-Host "[info] Switched backend mode to LIVE"
    }
    catch {
        Write-Warning "Could not switch to LIVE mode: $($_.Exception.Message)"
    }
}

Write-Host "[info] Sending telemetry to $BaseUrl/api/telemetry every $IntervalMs ms"
Write-Host "[info] Press Ctrl+C to stop"

while ($true) {
    try {
        $now = (Get-Date).ToUniversalTime().ToString("o")

        if ($Randomize) {
            $sendCurrent = [Math]::Max(0.0, (Get-RandomFloat -Center $Current -Amplitude $CurrentJitter))
            $sendTemp = [Math]::Max(0.0, (Get-RandomFloat -Center $Temperature -Amplitude $TempJitter))
        }
        else {
            $sendCurrent = $Current
            $sendTemp = $Temperature
        }

        $payload = @{
            current = [Math]::Round($sendCurrent, 3)
            temperature = [Math]::Round($sendTemp, 3)
            voltage = $Voltage
            soc = $Soc
            soh = $Soh
            timestamp = $now
        } | ConvertTo-Json

        $resp = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/telemetry" -ContentType "application/json" -Body $payload
        Write-Host "[$(Get-Date -Format HH:mm:ss)] sent current=$([Math]::Round($sendCurrent,2))A temp=$([Math]::Round($sendTemp,2))C mode=$($resp.mode)"
    }
    catch {
        Write-Warning "Telemetry push failed: $($_.Exception.Message)"
    }

    Start-Sleep -Milliseconds $IntervalMs
}

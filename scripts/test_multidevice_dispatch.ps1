param(
    [string]$BaseUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Loading demo devices..."
$demo = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/devices/load-demo"
$demo | ConvertTo-Json -Depth 6

Write-Host "[2/5] Switching to LIVE mode..."
Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/mode?name=LIVE" | Out-Null

Write-Host "[3/5] Pushing hot telemetry (to trigger safety override)..."
Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/telemetry" -ContentType "application/json" -Body '{"current":26,"temperature":52,"soh":0.98}' | Out-Null

Write-Host "[4/5] Dispatching control with dry_run=false..."
$dispatch = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/control/dispatch" -ContentType "application/json" -Body '{"dry_run":false,"optimization_enabled":true}'
$dispatch | ConvertTo-Json -Depth 8

Write-Host "[5/5] Reading command outbox..."
$outbox = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/control/outbox"
$outbox | ConvertTo-Json -Depth 8


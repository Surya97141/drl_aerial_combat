# ============================================================
# Aerial Combat AI — Demo Startup Script
# Run from repo root:  .\demo\run_demo.ps1
# ============================================================

$python = "C:\Users\hp\AppData\Local\Programs\Python\Python313\python.exe"
$demoDir = Split-Path $MyInvocation.MyCommand.Path -Parent

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Aerial Combat AI — Interactive Demo" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# 1. Generate heightmap if missing
$heightmap = "$demoDir\static\assets\heightmap.png"
if (-not (Test-Path $heightmap)) {
    Write-Host "  Generating terrain heightmap..." -ForegroundColor Yellow
    & $python "$demoDir\generate_heightmap.py"
}

# 2. Install demo dependencies if needed
Write-Host "  Checking demo dependencies..." -ForegroundColor Yellow
& $python -m pip install -q -r "$demoDir\requirements.txt" --no-warn-script-location

# 3. Launch the FastAPI server
Write-Host "`n  Starting demo server on http://localhost:8000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop`n" -ForegroundColor Green

Set-Location $demoDir
& $python server.py

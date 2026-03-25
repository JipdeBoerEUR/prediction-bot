# watchdog.ps1 — Run this ONCE on your Home PC. It loops forever:
#   1) git pull  2) start main.py  3) wait for exit  4) restart after 5s
#
# REMOTE KILL: create a file named 'stop.txt' in this folder to stop the loop.
#   Laptop: New-Item stop.txt; .\deploy.ps1
#
# Usage: .\watchdog.ps1

$ErrorActionPreference = "SilentlyContinue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonExe = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"   # fallback to system python
}

Write-Host ""
Write-Host "  =====================================" -ForegroundColor Magenta
Write-Host "    Watchdog started. CTRL+C to force  " -ForegroundColor Magenta
Write-Host "    stop, or drop 'stop.txt' here.     " -ForegroundColor Magenta
Write-Host "  =====================================" -ForegroundColor Magenta
Write-Host ""

while ($true) {

    # --- Remote Kill Check (pre-pull) ---
    if (Test-Path (Join-Path $ScriptDir "stop.txt")) {
        Write-Host "  [WATCHDOG] stop.txt detected — exiting." -ForegroundColor Red
        break
    }

    # --- Pull latest code from GitHub ---
    Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Pulling latest code..." -ForegroundColor Cyan
    git pull origin main

    # --- Remote Kill Check (post-pull, catches stop.txt pushed from laptop) ---
    if (Test-Path (Join-Path $ScriptDir "stop.txt")) {
        Write-Host "  [WATCHDOG] stop.txt detected after pull — exiting." -ForegroundColor Red
        break
    }

    # --- Launch main.py and wait for it to finish/crash ---
    Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] Starting main.py..." -ForegroundColor Green
    $process = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList "main.py" `
        -WorkingDirectory $ScriptDir `
        -NoNewWindow `
        -PassThru

    $process.WaitForExit()
    $exitCode = $process.ExitCode

    Write-Host "  [$(Get-Date -Format 'HH:mm:ss')] main.py exited (code $exitCode)." -ForegroundColor Yellow

    # --- Remote Kill Check (post-exit) ---
    if (Test-Path (Join-Path $ScriptDir "stop.txt")) {
        Write-Host "  [WATCHDOG] stop.txt detected — exiting." -ForegroundColor Red
        break
    }

    Write-Host "  [WATCHDOG] Restarting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

Write-Host "  [WATCHDOG] Stopped cleanly." -ForegroundColor Red

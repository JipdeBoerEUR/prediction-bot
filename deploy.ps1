# deploy.ps1 — Run this on your laptop to push code to GitHub.
# Usage: .\deploy.ps1

$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host ""
Write-Host "  Staging all changes..." -ForegroundColor Cyan
git add .

Write-Host "  Committing: deploy $timestamp" -ForegroundColor Cyan
git commit -m "deploy: $timestamp"

Write-Host "  Pushing to origin/main..." -ForegroundColor Cyan
git push origin main

Write-Host ""
Write-Host "  🚀 Code Deployed to GitHub. Home PC will sync shortly." -ForegroundColor Green
Write-Host ""

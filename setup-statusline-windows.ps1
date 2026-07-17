<#
.SYNOPSIS
  Show the router's live online/offline model state in the Claude Code desktop
  app's status line (next to the model chip).

.DESCRIPTION
  1. Installs claude_status_line.py into %USERPROFILE%\.claude\tools\.
  2. Merges a statusLine block into %USERPROFILE%\.claude\settings.json
     (non-destructive - every other setting is preserved).
  3. Reminds you to restart the Claude Code app.

  After this, whenever router.py dispatches, it writes
  %USERPROFILE%\.claude\.routing-status.json, and the status line shows a live
  badge for ~10 minutes: online (Ollama Cloud), online (Claude), or offline.

.EXAMPLE
  .\setup-statusline-windows.ps1
#>
param([int]$RefreshInterval = 10)

$ErrorActionPreference = "Stop"
$claudeDir = Join-Path $env:USERPROFILE ".claude"
$toolsDir  = Join-Path $claudeDir "tools"
$src       = Join-Path $PSScriptRoot ".claude\tools\claude_status_line.py"
$dest      = Join-Path $toolsDir "claude_status_line.py"
$settings  = Join-Path $claudeDir "settings.json"

if (!(Test-Path $src)) { throw "status-line tool not found at $src - run 'git pull' first." }
New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
Copy-Item $src $dest -Force
Write-Host "installed $dest" -ForegroundColor Green

# Merge statusLine into settings.json without clobbering anything else.
$cfg = if (Test-Path $settings) {
    try { Get-Content $settings -Raw | ConvertFrom-Json } catch { [pscustomobject]@{} }
} else { [pscustomobject]@{} }

$statusLine = [pscustomobject]@{
    type            = "command"
    command         = "python `"$dest`""
    refreshInterval = $RefreshInterval
}
$cfg | Add-Member -NotePropertyName statusLine -NotePropertyValue $statusLine -Force

$cfg | ConvertTo-Json -Depth 20 | Set-Content $settings -Encoding UTF8
Write-Host "wired statusLine into $settings" -ForegroundColor Green

# python on PATH? the status line silently hides if the command fails.
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: 'python' is not on PATH - the status line will be blank until it is." -ForegroundColor Yellow
}

Write-Host "`nDone. Fully quit and reopen the Claude Code app to load the status line." -ForegroundColor Cyan
Write-Host "Then run any router dispatch (e.g. router --tier glm `"hi`") and watch the badge appear."

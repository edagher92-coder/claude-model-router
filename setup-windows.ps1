<#
.SYNOPSIS
  One-command client setup for the claude-model-router Ollama bridge (v5.1).

.DESCRIPTION
  Persists the router's bridge environment variables for the current user and
  runs `python router.py --doctor` to verify. Run it on each machine that will
  dispatch through the router (NOT needed on the routing server itself — the
  daemon there just needs "Expose Ollama to the network" enabled).

.EXAMPLE
  # Client PC pointing at a tailnet routing server, with Ollama Cloud backstop:
  .\setup-windows.ps1 -RoutingServer "http://100.122.28.89:11434" -OllamaApiKey "<key>"

.EXAMPLE
  # Local-daemon-only machine (no args): just verifies localhost:11434.
  .\setup-windows.ps1
#>
param(
    [string]$RoutingServer = "",   # e.g. http://<tailscale-ip-or-host>:11434
    [string]$SecondServer = "",    # optional second daemon in the chain
    [string]$OllamaApiKey = "",    # optional: Ollama Cloud backstop
    [string]$GlmTag = ""           # optional: override glm-5.2:cloud
)

$ErrorActionPreference = "Stop"

function Set-UserEnv([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, "User")
    Set-Item -Path "Env:$Name" -Value $Value
    Write-Host "  set $Name" -ForegroundColor Green
}

$chain = @()
if ($RoutingServer) { $chain += $RoutingServer.TrimEnd("/") }
if ($SecondServer)  { $chain += $SecondServer.TrimEnd("/") }
$chain += "http://localhost:11434"
$chain = $chain | Select-Object -Unique

Write-Host "Configuring the Ollama bridge chain:" -ForegroundColor Cyan
Set-UserEnv "CLAUDE_ROUTER_OLLAMA_URL" ($chain -join ",")
if ($OllamaApiKey) { Set-UserEnv "OLLAMA_API_KEY" $OllamaApiKey }
if ($GlmTag)       { Set-UserEnv "GLM_OLLAMA_TAG" $GlmTag }

Write-Host "`nRunning the setup check:" -ForegroundColor Cyan
python "$PSScriptRoot\router.py" --doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDoctor reported no ready engine — follow the fix lines above." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "`nBridge ready. Live smoke test:" -ForegroundColor Cyan
Write-Host '  python router.py --tier glm "reply with the word ready"'

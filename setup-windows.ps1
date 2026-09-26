<#
.SYNOPSIS
  One-command client setup for the claude-model-router Ollama bridge (v5.1).

.DESCRIPTION
  Persists the router's bridge environment variables for the current user and
  runs `python router.py --doctor` to verify. Run it on each machine that will
  dispatch through the router (NOT needed on the routing server itself - the
  daemon there just needs "Expose Ollama to the network" enabled).

.EXAMPLE
  # Client PC pointing at a tailnet routing server, with Ollama Cloud backstop:
  .\setup-windows.ps1 -RoutingServer "http://100.122.28.89:11434" -OllamaApiKey "<key>"

.EXAMPLE
  # Local-daemon-only machine (no args): just verifies localhost:11434.
  .\setup-windows.ps1

.EXAMPLE
  # Skip the subscription-login step (e.g. on a server, which stays on keys):
  .\setup-windows.ps1 -SkipLogins

.NOTES
  Subscription logins: this laptop's own tools use the vendor CLI logins first
  (Claude Max, ChatGPT, SuperGrok, Google AI Pro, Qwen Coding Plan, Ollama) and
  fall back to API keys only when no login exists. Every login is stored by the
  vendor CLI in its own credential store - nothing is written to this repo and
  no token is printed. Servers and CI stay on API keys (ROUTER_AUTH=key / CI).
#>
param(
    [string]$RoutingServer = "",   # e.g. http://<tailscale-ip-or-host>:11434
    [string]$SecondServer = "",    # optional second daemon in the chain
    [string]$OllamaApiKey = "",    # optional: Ollama Cloud backstop
    [string]$GlmTag = "",          # optional: override glm-5.2:cloud
    [switch]$SkipLogins            # skip the subscription-login step
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

function Test-Cli([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Confirm-Step([string]$Question) {
    $answer = Read-Host "$Question [y/N]"
    return $answer -match '^(y|yes)$'
}

if (-not $SkipLogins) {
    Write-Host "`nSubscription logins (vendor CLIs; tokens stay in each CLI's own store):" -ForegroundColor Cyan

    # 1. Claude Code - Claude Max. `claude -p` is how the router uses it.
    if (Test-Cli "claude") {
        # An API key in the environment outranks the login for `claude -p`, so
        # check the login itself with the key hidden from this one command.
        $savedKey = $env:ANTHROPIC_API_KEY
        Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
        $loggedIn = $false
        try { $loggedIn = [bool]((claude auth status --json | ConvertFrom-Json).loggedIn) } catch { }
        if ($savedKey) { $env:ANTHROPIC_API_KEY = $savedKey }
        if ($loggedIn) {
            Write-Host "  Claude: signed in" -ForegroundColor Green
        } else {
            Write-Host "  Claude: signing in with your Claude subscription (browser opens)..."
            claude auth login --claudeai
        }
        if ([Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")) {
            Write-Host "  note: ANTHROPIC_API_KEY is set for your user. Interactive Claude Code will ask whether to use it" -ForegroundColor Yellow
            Write-Host "        (API billing) instead of the subscription; the router hides it from 'claude -p'." -ForegroundColor Yellow
        }
        Write-Host "  (no browser on this box? 'claude setup-token' prints a one-year token: set it yourself as the"
        Write-Host "   user env var CLAUDE_CODE_OAUTH_TOKEN. This script never reads or stores it.)"
    } else {
        Write-Host "  Claude: 'claude' not installed - https://code.claude.com/docs/en/setup" -ForegroundColor Yellow
    }

    # 2. Codex CLI - ChatGPT plan.
    if (Test-Cli "codex") {
        $status = (codex login status 2>&1 | Out-String)
        if ($LASTEXITCODE -eq 0 -and $status -match "Logged in using ChatGPT") {
            Write-Host "  ChatGPT (Codex): signed in with ChatGPT" -ForegroundColor Green
        } else {
            Write-Host "  ChatGPT (Codex): signing in with your ChatGPT plan (browser opens)..."
            codex login
        }
    } else {
        Write-Host "  ChatGPT (Codex): 'codex' not installed - https://learn.chatgpt.com/docs/codex/cli" -ForegroundColor Yellow
    }

    # 3. Grok Build CLI - SuperGrok.
    if (Test-Cli "grok") {
        if (Test-Path (Join-Path $HOME ".grok\auth.json")) {
            Write-Host "  Grok: signed in" -ForegroundColor Green
        } else {
            Write-Host "  Grok: signing in with your SuperGrok subscription (browser opens)..."
            grok login
        }
    } else {
        Write-Host "  Grok: 'grok' not installed - https://x.ai/news/grok-build-cli" -ForegroundColor Yellow
    }

    # 4. Gemini CLI - Google AI Pro (higher Gemini CLI / Code Assist limits; no API access).
    if (Test-Cli "gemini") {
        if (Test-Path (Join-Path $HOME ".gemini\oauth_creds.json")) {
            Write-Host "  Gemini: Google sign-in cached" -ForegroundColor Green
        } else {
            Write-Host "  Gemini: there is no login subcommand. Start 'gemini', choose 'Sign in with Google' and use"
            Write-Host "          the Google account that holds Google AI Pro. Type /quit when it is done."
            if (Confirm-Step "  Start gemini now?") { gemini }
        }
    } else {
        Write-Host "  Gemini: 'gemini' not installed - https://geminicli.com/docs/get-started/authentication/" -ForegroundColor Yellow
    }

    # 5. Qwen Code - Alibaba Cloud Model Studio Coding Plan.
    if (Test-Cli "qwen") {
        Write-Host "  Qwen: there is no login subcommand. Start 'qwen', type /auth, choose Subscription Plan >"
        Write-Host "        Alibaba Cloud Model Studio Coding Plan, pick your region and paste the Coding Plan key (sk-sp-...)."
        Write-Host "        (Qwen OAuth's free tier ended on 2026-04-15.)"
        if (Confirm-Step "  Start qwen now?") { qwen }
    } else {
        Write-Host "  Qwen: 'qwen' not installed - https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/" -ForegroundColor Yellow
    }

    # 6. Ollama - Ollama Cloud plan; the signed-in local daemon runs :cloud tags keyless.
    if (Test-Cli "ollama") {
        if (Confirm-Step "  Run 'ollama signin' (skip if this daemon is already signed in)?") { ollama signin }
    } else {
        Write-Host "  Ollama: 'ollama' not installed - https://docs.ollama.com/api/authentication" -ForegroundColor Yellow
    }
}

Write-Host "`nRunning the setup check:" -ForegroundColor Cyan
python "$PSScriptRoot\router.py" --doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDoctor reported no ready engine - follow the fix lines above." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "`nBridge ready. Live smoke test:" -ForegroundColor Cyan
Write-Host '  python router.py --tier glm "reply with the word ready"'

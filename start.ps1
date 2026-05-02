$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendProcess = $null
$FrontendProcess = $null

function Write-GalaxyInfo {
    param([string]$Message)
    Write-Host "[galaxy-ai] $Message"
}

function Stop-Galaxy {
    param([string]$Message)
    Write-Error "[galaxy-ai] ERROR: $Message"
    exit 1
}

function Load-GalaxyEnv {
    $EnvFile = Join-Path $RootDir ".env"
    if (-not (Test-Path $EnvFile -PathType Leaf)) {
        Stop-Galaxy "Root .env is missing. Create it from .env.example, then run ./start.ps1 again."
    }

    Get-Content $EnvFile | ForEach-Object {
        $Line = $_.Trim()
        if (-not $Line -or $Line.StartsWith("#")) {
            return
        }
        $Parts = $Line.Split("=", 2)
        if ($Parts.Count -ne 2) {
            return
        }
        [Environment]::SetEnvironmentVariable($Parts[0].Trim(), $Parts[1].Trim(), "Process")
    }
}

function Require-Command {
    param(
        [string]$CommandName,
        [string]$InstallHint
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        Stop-Galaxy "$CommandName is required but was not found. Suggested install: $InstallHint"
    }
}

function Require-Path {
    param(
        [string]$Path,
        [string]$Message
    )

    if (-not (Test-Path $Path)) {
        Stop-Galaxy $Message
    }
}

function Resolve-Uv {
    $UvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if ($UvCommand) {
        return $UvCommand.Source
    }

    $UnixUserUv = Join-Path $HOME ".local/bin/uv"
    if (Test-Path $UnixUserUv -PathType Leaf) {
        return $UnixUserUv
    }

    $WindowsUserUv = Join-Path $env:USERPROFILE ".local\bin\uv.exe"
    if (Test-Path $WindowsUserUv -PathType Leaf) {
        return $WindowsUserUv
    }

    Stop-Galaxy "uv is required but was not found. Suggested install: irm https://astral.sh/uv/install.ps1 | iex"
}

function Stop-ProcessTree {
    param([int]$TargetProcessId)

    if (Get-Command "Get-CimInstance" -ErrorAction SilentlyContinue) {
        $ChildProcesses = Get-CimInstance Win32_Process -Filter "ParentProcessId = $TargetProcessId" -ErrorAction SilentlyContinue
        foreach ($ChildProcess in $ChildProcesses) {
            Stop-ProcessTree -TargetProcessId $ChildProcess.ProcessId
        }
    }

    Stop-Process -Id $TargetProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-GalaxyServices {
    if ($null -ne $FrontendProcess -and -not $FrontendProcess.HasExited) {
        Stop-ProcessTree -TargetProcessId $FrontendProcess.Id
    }

    if ($null -ne $BackendProcess -and -not $BackendProcess.HasExited) {
        Stop-ProcessTree -TargetProcessId $BackendProcess.Id
    }
}

Load-GalaxyEnv
$BackendHost = if ($env:GALAXY_AI_HOST) { $env:GALAXY_AI_HOST } else { "127.0.0.1" }
$BackendPort = if ($env:GALAXY_AI_PORT) { $env:GALAXY_AI_PORT } else { "8000" }
$FrontendHost = "127.0.0.1"
$FrontendPort = "5173"

Write-GalaxyInfo "Running startup checks."
Require-Command "python" "Install Python 3.11+ from https://www.python.org/downloads/"
Require-Command "node" "Install Node.js LTS from https://nodejs.org/"
Require-Command "npm" "Install npm with Node.js LTS from https://nodejs.org/"
$UvBin = Resolve-Uv

$FrontendDir = Join-Path $RootDir "frontend"
$BackendDir = Join-Path $RootDir "backend"
Require-Path $FrontendDir "frontend/ does not exist."
Require-Path (Join-Path $FrontendDir "node_modules") "frontend dependencies are missing. Run ./init.ps1 first."
Require-Path $BackendDir "backend/ does not exist."
Require-Path (Join-Path $BackendDir ".venv") "backend virtual environment is missing. Run ./init.ps1 first."

try {
    Write-GalaxyInfo "Starting backend on http://${BackendHost}:${BackendPort}"
    $BackendProcess = Start-Process -FilePath $UvBin -ArgumentList @("run", "--no-sync", "uvicorn", "app.main:app", "--host", $BackendHost, "--port", $BackendPort) -WorkingDirectory $BackendDir -NoNewWindow -PassThru

    Write-GalaxyInfo "Starting frontend on http://${FrontendHost}:${FrontendPort}"
    $FrontendProcess = Start-Process -FilePath "npm" -ArgumentList @("run", "dev", "--", "--host", $FrontendHost, "--port", $FrontendPort) -WorkingDirectory $FrontendDir -NoNewWindow -PassThru

    Write-GalaxyInfo "Galaxy AI is running. Press Ctrl+C to stop both services."

    while ($true) {
        if ($BackendProcess.HasExited) {
            Write-GalaxyInfo "Backend stopped with exit code $($BackendProcess.ExitCode)."
            exit $BackendProcess.ExitCode
        }

        if ($FrontendProcess.HasExited) {
            Write-GalaxyInfo "Frontend stopped with exit code $($FrontendProcess.ExitCode)."
            exit $FrontendProcess.ExitCode
        }

        Start-Sleep -Seconds 1
    }
} finally {
    Stop-GalaxyServices
}

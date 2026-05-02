$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-GalaxyInfo {
    param([string]$Message)
    Write-Host "[galaxy-ai] $Message"
}

function Stop-Galaxy {
    param([string]$Message)
    Write-Error "[galaxy-ai] ERROR: $Message"
    exit 1
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

Write-GalaxyInfo "Running initialization checks."
Require-Command "python" "Install Python 3.11+ from https://www.python.org/downloads/"
$UvBin = Resolve-Uv

$FrontendDir = Join-Path $RootDir "frontend"
if (Test-Path $FrontendDir -PathType Container) {
    Require-Command "node" "Install Node.js LTS from https://nodejs.org/"
    Require-Command "npm" "Install npm with Node.js LTS from https://nodejs.org/"

    $PackageLock = Join-Path $FrontendDir "package-lock.json"
    $PackageJson = Join-Path $FrontendDir "package.json"
    if (Test-Path $PackageLock -PathType Leaf) {
        Write-GalaxyInfo "Installing frontend dependencies with npm ci."
        Push-Location $FrontendDir
        npm ci
        Pop-Location
    } elseif (Test-Path $PackageJson -PathType Leaf) {
        Write-GalaxyInfo "Installing frontend dependencies with npm install."
        Push-Location $FrontendDir
        npm install
        Pop-Location
    }
} else {
    Write-GalaxyInfo "frontend/ does not exist yet; skipping Node.js and npm setup."
}

$BackendDir = Join-Path $RootDir "backend"
if (-not (Test-Path $BackendDir -PathType Container)) {
    Stop-Galaxy "backend/ does not exist."
}

Write-GalaxyInfo "Synchronizing backend environment with uv."
Push-Location $BackendDir
& $UvBin sync
Pop-Location

Write-GalaxyInfo "Initialization complete. Start Galaxy AI with ./start.ps1."

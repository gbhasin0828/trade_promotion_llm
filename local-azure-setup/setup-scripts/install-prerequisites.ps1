# WINDOWS AZURE LOCAL SIMULATION SETUP
# File Path: local-azure-setup/setup-scripts/install-prerequisites.ps1

# Run as Administrator
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator"))
{
    Write-Host "This script requires Administrator privileges. Please run as Administrator." -ForegroundColor Red
    Read-Host "Press any key to exit..."
    exit 1
}

Write-Host "=== AZURE LOCAL SIMULATION SETUP ===" -ForegroundColor Green
Write-Host "Installing prerequisites for Azure development..." -ForegroundColor Yellow

# 1. Install Chocolatey (Package Manager)
Write-Host "`n1. Installing Chocolatey..." -ForegroundColor Cyan
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "✅ Chocolatey installed" -ForegroundColor Green
} else {
    Write-Host "✅ Chocolatey already installed" -ForegroundColor Green
}

# 2. Install Node.js
Write-Host "`n2. Installing Node.js..." -ForegroundColor Cyan
try {
    choco install nodejs -y
    Write-Host "✅ Node.js installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Node.js" -ForegroundColor Red
    Write-Host "Manual install: https://nodejs.org/en/download/" -ForegroundColor Yellow
}

# 3. Install Azure CLI
Write-Host "`n3. Installing Azure CLI..." -ForegroundColor Cyan
try {
    choco install azure-cli -y
    Write-Host "✅ Azure CLI installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Azure CLI" -ForegroundColor Red
    Write-Host "Manual install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows" -ForegroundColor Yellow
}

# 4. Install Docker Desktop
Write-Host "`n4. Installing Docker Desktop..." -ForegroundColor Cyan
try {
    choco install docker-desktop -y
    Write-Host "✅ Docker Desktop installed" -ForegroundColor Green
    Write-Host "⚠️  You may need to restart and enable Kubernetes in Docker Desktop" -ForegroundColor Yellow
} catch {
    Write-Host "❌ Failed to install Docker Desktop" -ForegroundColor Red
    Write-Host "Manual install: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
}

# 5. Install Git
Write-Host "`n5. Installing Git..." -ForegroundColor Cyan
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    choco install git -y
    Write-Host "✅ Git installed" -ForegroundColor Green
} else {
    Write-Host "✅ Git already installed" -ForegroundColor Green
}

# 6. Install kubectl
Write-Host "`n6. Installing kubectl..." -ForegroundColor Cyan
try {
    choco install kubernetes-cli -y
    Write-Host "✅ kubectl installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install kubectl" -ForegroundColor Red
}

# 7. Install Helm
Write-Host "`n7. Installing Helm..." -ForegroundColor Cyan
try {
    choco install kubernetes-helm -y
    Write-Host "✅ Helm installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Helm" -ForegroundColor Red
}

# 8. Install Act (Local GitHub Actions)
Write-Host "`n8. Installing Act..." -ForegroundColor Cyan
try {
    choco install act-cli -y
    Write-Host "✅ Act installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Act" -ForegroundColor Red
}

Write-Host "`n=== POST-INSTALLATION STEPS ===" -ForegroundColor Green

# Refresh environment
Write-Host "Refreshing environment variables..." -ForegroundColor Cyan
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Verify installations
Write-Host "`nVerifying installations..." -ForegroundColor Cyan

$tools = @(
    @{Name="Node.js"; Command="node"; Args="--version"},
    @{Name="NPM"; Command="npm"; Args="--version"},
    @{Name="Azure CLI"; Command="az"; Args="--version"},
    @{Name="Docker"; Command="docker"; Args="--version"},
    @{Name="kubectl"; Command="kubectl"; Args="version --client"},
    @{Name="Helm"; Command="helm"; Args="version"},
    @{Name="Git"; Command="git"; Args="--version"}
)

foreach ($tool in $tools) {
    try {
        $result = & $tool.Command $tool.Args.Split() 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $($tool.Name) is working" -ForegroundColor Green
        } else {
            Write-Host "❌ $($tool.Name) failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ $($tool.Name) not found" -ForegroundColor Red
    }
}

Write-Host "`n=== NEXT STEPS ===" -ForegroundColor Green
Write-Host "1. Restart your PowerShell session" -ForegroundColor Yellow
Write-Host "2. Run: npm install -g azure-functions-core-tools@4 --unsafe-perm true" -ForegroundColor Yellow
Write-Host "3. Start Docker Desktop and enable Kubernetes" -ForegroundColor Yellow
Write-Host "4. Run the setup verification script" -ForegroundColor Yellow

Write-Host "`nSetup completed! Press any key to exit..." -ForegroundColor Green
Read-Host
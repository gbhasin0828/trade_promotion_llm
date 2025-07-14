# WINDOWS KUBERNETES SETUP (AKS SIMULATION)
# File Path: local-azure-setup/setup-scripts/setup-kubernetes-windows.ps1

Write-Host "=== WINDOWS KUBERNETES SETUP (AKS SIMULATION) ===" -ForegroundColor Green

# Check if Docker Desktop is installed
Write-Host "`n1. Checking Docker Desktop..." -ForegroundColor Cyan
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker Desktop found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Desktop not found!" -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "After installation, restart this script." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if Docker daemon is running
Write-Host "`n2. Checking Docker daemon..." -ForegroundColor Cyan
try {
    $dockerInfo = docker info 2>&1
    if ($dockerInfo -match "Server Version") {
        Write-Host "✅ Docker daemon is running" -ForegroundColor Green
    } else {
        Write-Host "❌ Docker daemon is not running" -ForegroundColor Red
        Write-Host "Please start Docker Desktop and try again" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "❌ Cannot connect to Docker daemon" -ForegroundColor Red
    Write-Host "Please ensure Docker Desktop is running" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Option 1: Enable Kubernetes in Docker Desktop
Write-Host "`n3. Kubernetes Setup Options:" -ForegroundColor Cyan
Write-Host "   Option 1: Docker Desktop Kubernetes (Recommended)" -ForegroundColor Yellow
Write-Host "   Option 2: Kind (Kubernetes in Docker)" -ForegroundColor Yellow

$choice = Read-Host "`nChoose option (1 or 2)"

if ($choice -eq "1") {
    Write-Host "`n=== DOCKER DESKTOP KUBERNETES SETUP ===" -ForegroundColor Green
    
    # Check if Kubernetes is already enabled
    try {
        $kubectlVersion = kubectl version --client 2>&1
        Write-Host "✅ kubectl is available" -ForegroundColor Green
        
        # Try to connect to cluster
        $clusterInfo = kubectl cluster-info 2>&1
        if ($clusterInfo -match "Kubernetes control plane") {
            Write-Host "✅ Kubernetes cluster is running" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Kubernetes cluster not accessible" -ForegroundColor Yellow
            Write-Host "Please enable Kubernetes in Docker Desktop:" -ForegroundColor Cyan
            Write-Host "   1. Open Docker Desktop" -ForegroundColor White
            Write-Host "   2. Go to Settings > Kubernetes" -ForegroundColor White
            Write-Host "   3. Check 'Enable Kubernetes'" -ForegroundColor White
            Write-Host "   4. Click 'Apply & Restart'" -ForegroundColor White
            Write-Host "   5. Wait for Kubernetes to start (green icon)" -ForegroundColor White
            
            Read-Host "`nPress Enter after enabling Kubernetes in Docker Desktop"
        }
    } catch {
        Write-Host "❌ kubectl not found" -ForegroundColor Red
        Write-Host "Installing kubectl..." -ForegroundColor Cyan
        
        # Install kubectl
        try {
            if (Get-Command choco -ErrorAction SilentlyContinue) {
                choco install kubernetes-cli -y
            } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
                winget install Kubernetes.kubectl
            } else {
                Write-Host "Please install kubectl manually:" -ForegroundColor Yellow
                Write-Host "https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "❌ Failed to install kubectl automatically" -ForegroundColor Red
        }
    }
    
} elseif ($choice -eq "2") {
    Write-Host "`n=== KIND (KUBERNETES IN DOCKER) SETUP ===" -ForegroundColor Green
    
    # Install Kind
    Write-Host "Installing Kind..." -ForegroundColor Cyan
    try {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            choco install kind -y
        } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install Kubernetes.kind
        } else {
            # Manual download
            $kindUrl = "https://kind.sigs.k8s.io/dl/v0.20.0/kind-windows-amd64"
            $kindPath = "$env:USERPROFILE\kind.exe"
            Write-Host "Downloading Kind..." -ForegroundColor Cyan
            Invoke-WebRequest -Uri $kindUrl -OutFile $kindPath
            $env:PATH += ";$env:USERPROFILE"
        }
        
        Write-Host "✅ Kind installed" -ForegroundColor Green
        
        # Create Kind cluster
        Write-Host "Creating Kind cluster..." -ForegroundColor Cyan
        kind create cluster --name trade-promotion-cluster
        
        # Set kubectl context
        kubectl cluster-info --context kind-trade-promotion-cluster
        
        Write-Host "✅ Kind cluster created" -ForegroundColor Green
        
    } catch {
        Write-Host "❌ Failed to setup Kind" -ForegroundColor Red
        Write-Host "Manual installation: https://kind.sigs.k8s.io/docs/user/quick-start/" -ForegroundColor Yellow
    }
}

# Verify Kubernetes setup
Write-Host "`n4. Verifying Kubernetes setup..." -ForegroundColor Cyan
try {
    # Check kubectl
    $kubectlVersion = kubectl version --client
    Write-Host "✅ kubectl client: $($kubectlVersion.Split()[2])" -ForegroundColor Green
    
    # Check cluster connection
    $nodes = kubectl get nodes 2>&1
    if ($nodes -match "Ready") {
        Write-Host "✅ Kubernetes cluster is accessible" -ForegroundColor Green
        Write-Host "Nodes:" -ForegroundColor Gray
        kubectl get nodes
    } else {
        Write-Host "❌ Cannot connect to Kubernetes cluster" -ForegroundColor Red
        Write-Host "Output: $nodes" -ForegroundColor Red
    }
    
    # Create namespace for our app
    Write-Host "`nCreating trade-promotion-ai namespace..." -ForegroundColor Cyan
    kubectl create namespace trade-promotion-ai --dry-run=client -o yaml | kubectl apply -f -
    Write-Host "✅ Namespace created" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Kubernetes verification failed" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Install Helm (Kubernetes package manager)
Write-Host "`n5. Installing Helm..." -ForegroundColor Cyan
try {
    $helmVersion = helm version 2>&1
    if ($helmVersion -match "version") {
        Write-Host "✅ Helm already installed" -ForegroundColor Green
    } else {
        throw "Helm not found"
    }
} catch {
    try {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            choco install kubernetes-helm -y
        } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install Helm.Helm
        } else {
            Write-Host "Please install Helm manually:" -ForegroundColor Yellow
            Write-Host "https://helm.sh/docs/intro/install/" -ForegroundColor Yellow
        }
        Write-Host "✅ Helm installed" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install Helm" -ForegroundColor Red
    }
}

# Setup local Docker registry (ACR simulation)
Write-Host "`n6. Setting up local Docker registry..." -ForegroundColor Cyan
try {
    # Check if registry is already running
    $registryRunning = docker ps --filter "name=local-registry" --format "table {{.Names}}"
    
    if ($registryRunning -match "local-registry") {
        Write-Host "✅ Local registry already running" -ForegroundColor Green
    } else {
        # Start local registry
        docker run -d --restart=always -p 5000:5000 --name local-registry registry:2
        Write-Host "✅ Local Docker registry started on port 5000" -ForegroundColor Green
    }
    
    # Test registry
    Start-Sleep -Seconds 3
    $registryTest = Invoke-WebRequest -Uri "http://localhost:5000/v2/" -UseBasicParsing 2>&1
    if ($registryTest.StatusCode -eq 200) {
        Write-Host "✅ Registry is accessible" -ForegroundColor Green
    }
    
} catch {
    Write-Host "❌ Failed to setup local registry" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
}

# Create summary
Write-Host "`n=== SETUP SUMMARY ===" -ForegroundColor Green

$components = @{
    "Docker Desktop" = $(try { docker --version; $true } catch { $false })
    "Docker Daemon" = $(try { docker info | Out-Null; $true } catch { $false })
    "kubectl" = $(try { kubectl version --client | Out-Null; $true } catch { $false })
    "Kubernetes Cluster" = $(try { kubectl get nodes | Out-Null; $true } catch { $false })
    "Helm" = $(try { helm version | Out-Null; $true } catch { $false })
    "Local Registry" = $(try { Invoke-WebRequest -Uri "http://localhost:5000/v2/" -UseBasicParsing | Out-Null; $true } catch { $false })
}

foreach ($component in $components.GetEnumerator()) {
    $status = if ($component.Value) { "✅" } else { "❌" }
    Write-Host "$status $($component.Key)" -ForegroundColor $(if($component.Value){"Green"}else{"Red"})
}

$workingComponents = ($components.Values | Where-Object { $_ -eq $true }).Count
$totalComponents = $components.Count

Write-Host "`nOverall Status: $workingComponents/$totalComponents components working" -ForegroundColor $(if($workingComponents -eq $totalComponents){"Green"}else{"Yellow"})

if ($workingComponents -eq $totalComponents) {
    Write-Host "`n🎉 Kubernetes setup complete! Ready for AKS simulation." -ForegroundColor Green
    
    # Create next steps file
    $nextSteps = @"
# KUBERNETES SETUP COMPLETED

## Available Commands:
kubectl get nodes                    # View cluster nodes
kubectl get namespaces              # View namespaces
kubectl get pods -A                 # View all pods
helm list                          # View Helm releases

## Next Steps:
1. Deploy your applications to Kubernetes
2. Use kubectl to manage deployments
3. Access local registry at localhost:5000
4. Use Helm for package management

## Test Commands:
kubectl run nginx --image=nginx --port=80
kubectl expose pod nginx --type=NodePort --port=80
kubectl get services

## Cleanup:
kubectl delete pod nginx
kubectl delete service nginx
"@
    
    $nextSteps | Out-File -FilePath "kubernetes-next-steps.txt" -Encoding UTF8
    Write-Host "Next steps saved to: kubernetes-next-steps.txt" -ForegroundColor Cyan
    
} else {
    Write-Host "`n⚠️  Some components need attention. Please fix the issues above." -ForegroundColor Yellow
}

Write-Host "`nPress any key to exit..." -ForegroundColor Green
Read-Host
# launch_finetune.ps1
# Monitors best.pt download, stops Vast.ai instance, launches LoRA fine-tuning
# Run: powershell -ExecutionPolicy Bypass -File D:\FlightMind\scripts\launch_finetune.ps1

$ErrorActionPreference = "Continue"
$bestPt = "D:\FlightMind\checkpoints\v2_cloud\best.pt"
$targetSize = 11476780274  # 11GB (exact size from cloud ls -la)
$vastai = "C:\Users\Administrator\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\vastai.exe"
$instanceId = "31013532"

Write-Host "=== FlightMind Download Monitor & Finetune Launcher ==="
Write-Host "Target: $bestPt ($([math]::Round($targetSize / 1GB, 1)) GB)"
Write-Host ""

# Step 1: Wait for download to complete
Write-Host "[1/3] Waiting for best.pt download to complete..."
while ($true) {
    if (Test-Path $bestPt) {
        $currentSize = (Get-Item $bestPt).Length
        $pct = [math]::Round(($currentSize / $targetSize) * 100, 1)
        $gb = [math]::Round($currentSize / 1GB, 2)
        Write-Host "  Progress: $gb GB / $([math]::Round($targetSize / 1GB, 2)) GB ($pct%)" -NoNewline
        Write-Host "`r" -NoNewline

        if ($currentSize -ge $targetSize) {
            Write-Host ""
            Write-Host "  Download complete! ($gb GB)"
            break
        }
    } else {
        Write-Host "  Waiting for file to appear..."
    }
    Start-Sleep -Seconds 30
}

# Step 2: Stop Vast.ai instance
Write-Host ""
Write-Host "[2/3] Stopping Vast.ai instance $instanceId..."
& $vastai stop instance $instanceId
Write-Host "  Instance stop command sent."
Start-Sleep -Seconds 5
$status = & $vastai show instances 2>&1
Write-Host "  Instance status: $status"

# Step 3: Launch fine-tuning
Write-Host ""
Write-Host "[3/3] Launching LoRA fine-tuning on RTX 4060..."
Write-Host "  Checkpoint: $bestPt"
Write-Host "  LoRA rank: 16, alpha: 32"
Write-Host "  Max steps: 3000"
Write-Host ""

Set-Location "D:\FlightMind"
python train/finetune.py `
    --checkpoint "checkpoints/v2_cloud/best.pt" `
    --lora-rank 16 `
    --lora-alpha 32 `
    --max-steps 3000 `
    --batch-size 2 `
    --grad-accum 8 `
    --lr 2e-4 `
    --min-lr 2e-5 `
    --warmup-steps 100 `
    --seq-len 512 `
    --log-every 10 `
    --checkpoint-every 500 `
    --device cuda

Write-Host ""
Write-Host "=== Fine-tuning complete! ==="

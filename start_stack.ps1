# Clean up previous instances
Get-Process -Name "*llama*", "*haven-server*", "*sd-server*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500

Write-Host "=================================================================="
Write-Host "🚀 LAUNCHING HAVEN SOVEREIGN ECOSYSTEM (C++ Bare-Metal)"
Write-Host "=================================================================="

# 1. Start Stable Diffusion C++ Server on port 8085 (6-step LCM)
$sdPath = "C:\Users\admin\stable-diffusion-cpp\sd-server.exe"
$sdModel = "C:\Users\admin\stable-diffusion-cpp\models\DreamShaper8_LCM_q4_0.gguf"
$taesdPath = "C:\Users\admin\stable-diffusion-cpp\models\taesd.safetensors"

if (Test-Path $sdPath) {
    Write-Host "🎨 Starting Stable Diffusion C++ Studio on port 8085..."
    Start-Process -FilePath $sdPath -ArgumentList "-m `"$sdModel`" --taesd `"$taesdPath`" --sampling-method lcm --steps 6 --cfg-scale 1.8 --listen-port 8085 --threads 8" -WorkingDirectory "C:\Users\admin\stable-diffusion-cpp" -WindowStyle Hidden
}

# 2. Start haven-server Sovereign C++ Engine & Web Studio on port 11436
$havenServer = "C:\Users\admin\source\haven-cpp\haven-server.exe"
if (-not (Test-Path $havenServer)) {
    $havenServer = "C:\Users\admin\source\haven-cpp\build\haven-server.exe"
}

Write-Host "🧠 Starting Sovereign haven-server (OpenAI API + Web Studio) on port 11436..."
Start-Process -FilePath $havenServer -ArgumentList "11436" -WorkingDirectory "C:\Users\admin\source\haven-cpp" -WindowStyle Hidden

Start-Sleep -Seconds 3

Write-Host "`n=== Active Haven Processes ==="
Get-Process -Name "*haven*", "*sd-server*" -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, @{Name="RAM_MB"; Expression={[math]::round($_.WorkingSet64 / 1MB, 2)}}

Write-Host "`n✓ Haven Sovereign Web Studio & OpenAI API: http://127.0.0.1:11436"
Write-Host "✓ Stable Diffusion C++ Studio: http://127.0.0.1:8085"
Write-Host "✓ Ready for Sanctuary & Android client connections!"
Write-Host "=================================================================="

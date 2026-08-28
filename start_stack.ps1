Get-Process -Name "*llama*", "*haven*" -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Starting llama-server backend on port 11436 (-fit off, -np 1, -c 8192)..."
Start-Process -FilePath "C:\Users\admin\source\llama.cpp\build\bin\Release\llama-server.exe" -ArgumentList "-m C:\Users\admin\gemma4-turbo-family\haven-chat-v5.0.gguf --port 11436 --host 0.0.0.0 -c 8192 -np 1 --threads 8 -fit off" -WindowStyle Hidden

Write-Host "Starting haven-server Web Studio on port 11438..."
Start-Process -FilePath "C:\Users\admin\source\haven-cpp\haven-server.exe" -ArgumentList "11438" -WorkingDirectory "C:\Users\admin\source\haven-cpp" -WindowStyle Hidden

Start-Sleep -Seconds 6

Write-Host "`n=== Active Process Memory Check ==="
Get-Process -Name "*llama*", "*haven*" | Select-Object Id, ProcessName, @{Name="RAM_MB"; Expression={[math]::round($_.WorkingSet64 / 1MB, 2)}}

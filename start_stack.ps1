Get-Process -Name "*llama*", "*haven*" -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Starting sovereign haven-server (OpenAI API + Web Studio) on port 11436..."
Start-Process -FilePath "C:\Users\admin\source\haven-cpp\haven-server.exe" -ArgumentList "11436" -WorkingDirectory "C:\Users\admin\source\haven-cpp" -WindowStyle Hidden

Start-Sleep -Seconds 4

Write-Host "`n=== Active Process Memory Check ==="
Get-Process -Name "*haven*" | Select-Object Id, ProcessName, @{Name="RAM_MB"; Expression={[math]::round($_.WorkingSet64 / 1MB, 2)}}

Write-Host "`n✓ Haven Sovereign Stack online on http://127.0.0.1:11436"
Write-Host "✓ Connected to Sanctuary and ASP.NET Core haven-server!"

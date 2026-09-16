# Kill listeners on DeepScan ports, then start all four servers.
$ErrorActionPreference = "Continue"
$ports = 3000, 5000, 5500, 7000
foreach ($port in $ports) {
  $listenPids = @()
  try {
    $listenPids = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique)
  } catch { }
  foreach ($procId in $listenPids) {
    if ($procId -and $procId -gt 0) {
      Write-Host "Stopping PID $procId (port $port)"
      Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
  }
}
Start-Sleep -Seconds 1
& (Join-Path $PSScriptRoot "start-all.ps1")

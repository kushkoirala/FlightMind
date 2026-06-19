$procs = Get-WmiObject Win32_Process -Filter "Name LIKE 'python%'" | Where-Object { $_.CommandLine -like '*finetune*' }
foreach ($p in $procs) {
    Write-Host "Killing PID $($p.ProcessId): $($p.CommandLine)"
    Stop-Process -Id $p.ProcessId -Force
}
if (-not $procs) { Write-Host "No finetune process found" }

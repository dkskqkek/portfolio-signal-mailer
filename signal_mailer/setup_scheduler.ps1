# Signal Mailer Windows Task Scheduler Setup Script
# Run this script with Administrator privileges

$TaskName = "SignalMailerDaily"
$PythonPath = "C:\Python313\python.exe"
$ScriptPath = "d:\gg\signal_mailer\daily_run.py"
$WorkingDirectory = "d:\gg\signal_mailer"

# 1. Existing Task Cleanup
Write-Host "Checking for existing task: $TaskName..."
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 2. Define Action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory $WorkingDirectory

# 3. Define Trigger (Every day at 9:00 AM)
$Trigger = New-ScheduledTaskTrigger -Daily -At 9:00am

# 4. Define Settings
# Allow waking the computer, and allow running on battery
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun

# 5. Register Task
Write-Host "Registering new task: $TaskName..."
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Daily Portfolio Signal Mailer (09:00 KST)"

Write-Host "`nSuccessfully registered $TaskName."
Write-Host "Task will run daily at 09:00 AM."
Write-Host "Check 'Task Scheduler' in Windows to verify settings."

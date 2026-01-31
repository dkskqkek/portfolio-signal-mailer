# 분봉 데이터 수집 자동화 스케줄러 설정
# 관리자 권한으로 실행 필요

Write-Host "=== Antigravity 분봉 데이터 수집 자동화 설정 ===" -ForegroundColor Cyan

# Python 경로 확인
$pythonPath = (Get-Command python).Source
Write-Host "`n[확인] Python 경로: $pythonPath" -ForegroundColor Green

# 작업 디렉터리
$workingDir = "d:\gg"
Write-Host "[확인] 작업 디렉터리: $workingDir" -ForegroundColor Green

# 1. 한국 주식 수집 작업 (15:40)
Write-Host "`n[생성] 한국 주식 분봉 수집 작업..." -ForegroundColor Yellow

$action_kr = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "d:\gg\scripts\collect_intraday_kr.py" `
    -WorkingDirectory $workingDir

$trigger_kr = New-ScheduledTaskTrigger -Daily -At "15:40"

$settings_kr = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# 기존 작업 삭제 (있다면)
$existingTask = Get-ScheduledTask -TaskName "Antigravity_KR_Intraday" -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName "Antigravity_KR_Intraday" -Confirm:$false
    Write-Host "  - 기존 작업 삭제됨" -ForegroundColor Gray
}

Register-ScheduledTask `
    -Action $action_kr `
    -Trigger $trigger_kr `
    -Settings $settings_kr `
    -TaskName "Antigravity_KR_Intraday" `
    -Description "한국 주식 5분봉 데이터 자동 수집 (15:40)" `
    -User $env:USERNAME | Out-Null

Write-Host "  ✅ 한국 주식 수집 작업 등록 완료 (매일 15:40)" -ForegroundColor Green

# 2. 미국 주식 수집 작업 (06:30)
Write-Host "`n[생성] 미국 주식 분봉 수집 작업..." -ForegroundColor Yellow

$action_us = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "d:\gg\scripts\collect_intraday_us.py" `
    -WorkingDirectory $workingDir

$trigger_us = New-ScheduledTaskTrigger -Daily -At "06:30"

$settings_us = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# 기존 작업 삭제 (있다면)
$existingTask = Get-ScheduledTask -TaskName "Antigravity_US_Intraday" -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName "Antigravity_US_Intraday" -Confirm:$false
    Write-Host "  - 기존 작업 삭제됨" -ForegroundColor Gray
}

Register-ScheduledTask `
    -Action $action_us `
    -Trigger $trigger_us `
    -Settings $settings_us `
    -TaskName "Antigravity_US_Intraday" `
    -Description "미국 주식 5분봉 데이터 자동 수집 (06:30)" `
    -User $env:USERNAME | Out-Null

Write-Host "  ✅ 미국 주식 수집 작업 등록 완료 (매일 06:30)" -ForegroundColor Green

# 3. 등록된 작업 확인
Write-Host "`n=== 등록된 작업 확인 ===" -ForegroundColor Cyan

$tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "Antigravity_*_Intraday" }
foreach ($task in $tasks) {
    $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
    Write-Host "`n[작업명] $($task.TaskName)" -ForegroundColor White
    Write-Host "  - 상태: $($task.State)" -ForegroundColor Gray
    Write-Host "  - 다음 실행: $($info.NextRunTime)" -ForegroundColor Gray
    Write-Host "  - 마지막 실행: $($info.LastRunTime)" -ForegroundColor Gray
}

# 4. 수동 실행 테스트 여부
Write-Host "`n=== 설정 완료 ===" -ForegroundColor Cyan
Write-Host "자동화가 설정되었습니다. 다음 실행 시간:" -ForegroundColor White
Write-Host "  - 한국 주식: 매일 15:40" -ForegroundColor Yellow
Write-Host "  - 미국 주식: 매일 06:30" -ForegroundColor Yellow

Write-Host "`n수동 테스트를 원하시면 다음 명령을 실행하세요:" -ForegroundColor Gray
Write-Host "  Start-ScheduledTask -TaskName 'Antigravity_KR_Intraday'" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName 'Antigravity_US_Intraday'" -ForegroundColor Cyan

Write-Host "`n작업 로그 확인:" -ForegroundColor Gray
Write-Host "  Get-ScheduledTaskInfo -TaskName 'Antigravity_KR_Intraday'" -ForegroundColor Cyan

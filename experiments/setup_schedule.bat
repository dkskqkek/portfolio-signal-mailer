@echo off
REM d:\gg\setup_schedule.bat
REM Windows Task Scheduler 등록 스크립트

chcp 65001 > nul
setlocal

echo ==========================================
echo Windows 작업 스케줄러 등록
echo ==========================================

set TASK_NAME="Portfolio Daily Signal"
set SCRIPT_PATH="d:\gg\daily_routine.bat"
set START_TIME=09:00

echo.
echo 작업 이름: %TASK_NAME%
echo 실행 파일: %SCRIPT_PATH%
echo 실행 시간: 매일 %START_TIME%
echo.

REM 기존 작업 삭제 (중복 방지)
schtasks /delete /tn %TASK_NAME% /f >nul 2>&1

REM 새 작업 생성
REM /sc daily: 매일
REM /tn: 작업 이름
REM /tr: 실행할 프로그램
REM /st: 시작 시간
REM /f: 강제 생성
REM /rl highest: 관리자 권한으로 실행 (필요시)

schtasks /create /sc daily /tn %TASK_NAME% /tr %SCRIPT_PATH% /st %START_TIME% /f

if errorlevel 1 (
    echo.
    echo [Error] 작업 생성 실패. 관리자 권한으로 실행했는지 확인하세요.
    echo.
) else (
    echo.
    echo [Success] 작업이 성공적으로 등록되었습니다!
    echo 매일 %START_TIME%에 자동으로 실행됩니다.
    echo.
    echo 확인 명령: schtasks /query /tn %TASK_NAME%
)

pause

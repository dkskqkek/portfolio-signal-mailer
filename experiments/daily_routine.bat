@echo off
REM d:\gg\daily_routine.bat
REM 매일 아침 실행되는 자동화 스크립트
REM 1. Git Commit & Push
REM 2. Daily Signal Email

chcp 65001 > nul
setlocal enabledelayedexpansion

echo ==========================================
echo [Start] Daily Routine: %date% %time%
echo ==========================================

cd /d "d:\gg"

echo.
echo [Step 1] Git Auto Commit
echo ------------------------------------------

REM Git 상태 확인
git status

REM 변경사항 추가
git add .

REM 커밋 (날짜 포함)
set CUR_DATE=%date:~0,10%
git commit -m "Auto-commit: Daily Report %CUR_DATE%"

REM 푸시
echo Pushing to GitHub...
git push

if errorlevel 1 (
    echo [Warning] Git Push failed or no changes to push.
) else (
    echo [Success] Git Push completed.
)

echo.
echo [Step 2] Sending Daily Signal Email
echo ------------------------------------------

REM Python 가상환경이 있다면 활성화 (여기서는 글로벌 파이썬 또는 특정 경로 파이썬 사용)
REM 필요시 C:\Python313\python.exe 등 절대경로 사용 권장
python d:\gg\signal_mailer\integrated_run.py

if errorlevel 1 (
    echo [Error] Failed to send email.
) else (
    echo [Success] Email sent successfully.
)

echo.
echo ==========================================
echo [End] Daily Routine Completed
echo ==========================================
pause

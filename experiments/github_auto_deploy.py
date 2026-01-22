# -*- coding: utf-8 -*-
"""
GitHub 저장소 자동 생성 및 푸시 (Python)

사용법:
    python github_auto_deploy.py
"""

import subprocess
import sys
import os
from getpass import getpass

def run_command(cmd, shell=True):
    """명령어 실행 및 결과 반환"""
    try:
        result = subprocess.run(
            cmd, 
            shell=shell, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "명령어 실행 시간 초과"
    except Exception as e:
        return 1, "", str(e)

def main():
    print("\n" + "="*60)
    print("GitHub 저장소 자동 생성 및 푸시")
    print("="*60 + "\n")
    
    # Step 1: GitHub 토큰 입력
    print("【Step 1】GitHub Personal Access Token")
    print("\n토큰 생성: https://github.com/settings/tokens/new")
    print("필요 권한: repo, workflow\n")
    
    github_token = getpass("GitHub Personal Access Token 입력: ")
    
    if not github_token:
        print("❌ 토큰이 필요합니다.")
        return 1
    
    # Step 2: GitHub 사용자명
    print("\n【Step 2】GitHub 사용자명")
    github_username = input("GitHub 사용자명 입력 (예: gamja-user): ").strip()
    
    if not github_username:
        print("❌ 사용자명이 필요합니다.")
        return 1
    
    # Step 3: 인증
    print("\n【Step 3】GitHub CLI 인증 중...")
    
    # echo token | gh auth login
    try:
        result = subprocess.run(
            f'echo {github_token} | gh auth login --with-token --git-protocol https',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ 인증 실패: {result.stderr}")
            return 1
        
        print("✓ 인증 성공")
    except Exception as e:
        print(f"❌ 인증 오류: {e}")
        return 1
    
    # Step 4: 저장소 생성 및 푸시
    print("\n【Step 4】GitHub 저장소 생성 중...")
    
    os.chdir('d:\\gg')
    
    # 저장소 생성
    returncode, stdout, stderr = run_command(
        'gh repo create portfolio-signal-mailer --public --source=. --remote=origin --push'
    )
    
    if returncode != 0:
        print(f"❌ 저장소 생성 실패")
        print(f"오류: {stderr}")
        
        # 저장소가 이미 있는 경우
        if "already exists" in stderr:
            print("\n💡 저장소가 이미 존재합니다.")
            print("기존 저장소를 삭제하고 다시 생성하거나,")
            print("다른 저장소명을 사용하세요.")
        return 1
    
    # Step 5: 완료
    print("\n" + "="*60)
    print("✅ 저장소 생성 및 푸시 완료!")
    print("="*60)
    
    print(f"\n📍 저장소 URL:")
    print(f"https://github.com/{github_username}/portfolio-signal-mailer")
    
    print("\n" + "="*60)
    print("【Step 5】GitHub Secrets 설정")
    print("="*60)
    
    print(f"\n다음 URL에서 Secrets을 설정하세요:")
    print(f"https://github.com/{github_username}/portfolio-signal-mailer/settings/secrets/actions")
    
    print("\n필요한 Secrets:")
    print("  1. SENDER_EMAIL")
    print("     값: gamjatangjo@gmail.com")
    print("\n  2. SENDER_PASSWORD")
    print("     값: [Gmail 앱 비밀번호]")
    print("\n  3. RECIPIENT_EMAIL")
    print("     값: gamjatangjo@gmail.com")
    print("\n  4. GEMINI_API_KEY")
    print("     값: AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA")
    
    print("\n" + "="*60)
    print("모든 설정이 완료되었습니다!")
    print("=" *60)
    
    print("\n다음: GitHub Secrets 설정 후 Actions 확인")
    print("예상 실행: 매일 UTC 0시 (KST 오전 9시)\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

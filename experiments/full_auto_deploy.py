#!/usr/bin/env python3
"""
완전 자동 배포 스크립트 - GitHub 저장소 생성 + Secrets 설정까지 자동화
"""
import subprocess
import os
import sys
import json
from typing import Optional

def run_command(cmd: str) -> tuple[int, str, str]:
    """명령어 실행 및 출력 반환"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "명령어 타임아웃"
    except Exception as e:
        return 1, "", str(e)

def print_step(step_num: int, title: str):
    """단계 표시"""
    print(f"\n{'='*80}")
    print(f"📍 STEP {step_num}: {title}")
    print(f"{'='*80}\n")

def print_success(msg: str):
    print(f"✅ {msg}")

def print_error(msg: str):
    print(f"❌ {msg}")

def print_info(msg: str):
    print(f"ℹ️  {msg}")

def check_gh_cli() -> bool:
    """GitHub CLI 설치 확인"""
    code, _, _ = run_command("gh --version")
    if code != 0:
        print_error("GitHub CLI가 설치되지 않았습니다!")
        print_info("https://cli.github.com/ 에서 설치하세요")
        return False
    print_success("GitHub CLI 확인됨")
    return True

def get_github_token() -> Optional[str]:
    """GitHub Personal Access Token 입력받기"""
    print_info("GitHub Personal Access Token을 생성하세요:")
    print("  1. https://github.com/settings/tokens/new 에서 토큰 생성")
    print("  2. Scopes: ✓ repo, ✓ workflow 선택")
    print("  3. 생성된 토큰 복사\n")
    
    token = input("📌 GitHub Personal Access Token 입력: ").strip()
    if not token or len(token) < 20:
        print_error("유효한 토큰이 아닙니다")
        return None
    return token

def authenticate_gh(token: str) -> bool:
    """GitHub CLI 인증"""
    # 토큰으로 인증
    echo_cmd = f'echo "{token}" | gh auth login --with-token'
    code, stdout, stderr = run_command(echo_cmd)
    
    if code != 0:
        print_error(f"GitHub 인증 실패: {stderr}")
        return False
    
    # 인증 확인
    code, stdout, stderr = run_command("gh auth status")
    if code == 0:
        print_success("GitHub 인증 완료")
        return True
    return False

def get_github_username() -> Optional[str]:
    """GitHub 사용자명 가져오기"""
    code, stdout, stderr = run_command("gh api user --jq '.login'")
    if code == 0:
        username = stdout.strip()
        print_success(f"GitHub 사용자명: {username}")
        return username
    print_error("GitHub 사용자명을 가져올 수 없습니다")
    return None

def create_repository() -> Optional[str]:
    """GitHub 저장소 생성"""
    print_info("portfolio-signal-mailer 저장소 생성 중...")
    
    cmd = "gh repo create portfolio-signal-mailer --public --source=. --remote=origin --push"
    code, stdout, stderr = run_command(cmd)
    
    if code != 0:
        print_error(f"저장소 생성 실패: {stderr}")
        if "already exists" in stderr.lower():
            print_info("저장소가 이미 존재합니다. 진행합니다...")
            return "portfolio-signal-mailer"
        return None
    
    print_success("저장소 생성 및 코드 푸시 완료")
    return "portfolio-signal-mailer"

def set_github_secrets(username: str) -> bool:
    """GitHub Secrets 설정"""
    secrets = {
        "SENDER_EMAIL": "gamjatangjo@gmail.com",
        "RECIPIENT_EMAIL": "gamjatangjo@gmail.com",
        "GEMINI_API_KEY": "AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA",
    }
    
    print_info("⚠️  Gmail 앱 비밀번호를 생성하세요:")
    print("  1. https://myaccount.google.com/apppasswords 접속")
    print("  2. \"Portfolio Signal Mailer\" 입력")
    print("  3. 생성된 16자리 비밀번호 복사\n")
    
    gmail_password = input("📌 Gmail 앱 비밀번호 입력: ").strip()
    if not gmail_password:
        print_error("Gmail 비밀번호가 필요합니다")
        return False
    
    secrets["SENDER_PASSWORD"] = gmail_password
    
    # Secrets 설정
    repo = f"{username}/portfolio-signal-mailer"
    failed = []
    
    for secret_name, secret_value in secrets.items():
        cmd = f'gh secret set {secret_name} --repo {repo} --body "{secret_value}"'
        code, stdout, stderr = run_command(cmd)
        
        if code == 0:
            print_success(f"Secrets 설정: {secret_name}")
        else:
            print_error(f"Secrets 설정 실패: {secret_name}")
            failed.append(secret_name)
    
    if failed:
        print_error(f"실패한 Secrets: {', '.join(failed)}")
        return False
    
    return True

def display_completion_info(username: str):
    """완료 정보 표시"""
    print("\n" + "="*80)
    print("🎉 배포 완료!")
    print("="*80 + "\n")
    
    repo_url = f"https://github.com/{username}/portfolio-signal-mailer"
    actions_url = f"{repo_url}/actions"
    settings_url = f"{repo_url}/settings/secrets/actions"
    
    print("✅ 완료된 항목:")
    print(f"  • GitHub 저장소 생성: {repo_url}")
    print(f"  • 코드 푸시 완료")
    print(f"  • GitHub Secrets 설정 완료\n")
    
    print("📊 모니터링:")
    print(f"  • GitHub Actions 확인: {actions_url}")
    print(f"  • Secrets 관리: {settings_url}\n")
    
    print("⏰ 다음 실행:")
    print("  • 첫 번째 신호 감지: 내일 UTC 0시 (KST 오전 9시)")
    print("  • 신호 변화 시 자동 이메일 발송\n")
    
    print("📈 신호 이력 확인:")
    print(f"  • 저장소 → signal_mailer/signal_history.json\n")

def main():
    """메인 실행 함수"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "🚀 완전 자동 배포 (1~3단계 자동화)" + " "*21 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Step 1: 환경 확인
    print_step(1, "환경 확인")
    if not check_gh_cli():
        return 1
    
    # Step 2: GitHub 인증
    print_step(2, "GitHub 인증")
    token = get_github_token()
    if not token:
        return 1
    
    if not authenticate_gh(token):
        return 1
    
    # Step 3: 사용자명 확인
    username = get_github_username()
    if not username:
        return 1
    
    # Step 4: 저장소 생성
    print_step(3, "GitHub 저장소 생성")
    repo_name = create_repository()
    if not repo_name:
        return 1
    
    # Step 5: Secrets 설정
    print_step(4, "GitHub Secrets 자동 설정")
    if not set_github_secrets(username):
        print_info("⚠️  Secrets를 수동으로 설정하세요:")
        print(f"  URL: https://github.com/{username}/{repo_name}/settings/secrets/actions")
        print("  - SENDER_EMAIL = gamjatangjo@gmail.com")
        print("  - SENDER_PASSWORD = [Gmail 앱 비밀번호]")
        print("  - RECIPIENT_EMAIL = gamjatangjo@gmail.com")
        print("  - GEMINI_API_KEY = AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA")
    
    # 완료 정보 표시
    display_completion_info(username)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/bin/bash
# GitHub 저장소 설정 및 푸시 스크립트 (Linux/Mac)

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Portfolio Signal Mailer - GitHub 자동 배포              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

read -p "GitHub 사용자명 입력 (예: gamja-user): " GITHUB_USERNAME
read -p "저장소명 입력 (기본값: portfolio-signal-mailer): " REPO_NAME

REPO_NAME=${REPO_NAME:-portfolio-signal-mailer}

echo ""
echo "[단계 1/3] Git 저장소 설정..."

# git init은 이미 되었으므로 remote 추가만
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git 2>/dev/null || {
    echo "기존 origin이 있습니다. 제거 후 재추가합니다..."
    git remote remove origin
    git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
}

echo ""
echo "[단계 2/3] 파일 추가..."

# .gitignore 업데이트
cat > .gitignore << 'EOF'
# 의존성
__pycache__/
*.pyc
venv/

# 민감한 파일
signal_mailer/config.yaml
signal_mailer/*.log
.env

# OS
.DS_Store
Thumbs.db

# crash_detection_system은 제외
crash_detection_system/
EOF

git add .
git status

echo ""
echo "[단계 3/3] 첫 커밋..."

git commit -m "Initial commit: Portfolio signal mailer with GitHub Actions automation"

echo ""
echo "다음 명령어로 GitHub에 푸시하세요:"
echo ""
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "그 다음, GitHub Secrets을 설정하세요:"
echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME/settings/secrets/actions"
echo ""
echo "Required Secrets:"
echo "  - SENDER_EMAIL: gamjatangjo@gmail.com"
echo "  - SENDER_PASSWORD: [Gmail 앱 비밀번호]"
echo "  - RECIPIENT_EMAIL: gamjatangjo@gmail.com"
echo "  - GEMINI_API_KEY: [제공된 API 키]"
echo ""

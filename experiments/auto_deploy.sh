#!/bin/bash
# GitHub 저장소 생성 및 푸시 자동화 스크립트

echo "=========================================="
echo "GitHub 저장소 자동 생성 및 푸시"
echo "=========================================="
echo ""

# Step 1: GitHub 인증 (토큰 사용)
echo "【Step 1】GitHub 인증"
echo "Personal Access Token을 생성하세요:"
echo "https://github.com/settings/tokens/new"
echo ""
echo "필요 권한:"
echo "  ✓ repo (전체 저장소 접근)"
echo "  ✓ workflow (Actions 권한)"
echo ""
read -sp "GitHub Personal Access Token 입력: " GITHUB_TOKEN
echo ""

# Step 2: GitHub 사용자명
echo ""
echo "【Step 2】GitHub 사용자명"
read -p "GitHub 사용자명 입력 (예: gamja-user): " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "사용자명이 필요합니다."
    exit 1
fi

# Step 3: 저장소 생성
echo ""
echo "【Step 3】GitHub 저장소 생성 중..."

# gh CLI에 토큰 저장
echo "$GITHUB_TOKEN" | gh auth login --with-token

# 저장소 생성
gh repo create portfolio-signal-mailer \
    --public \
    --source="$(pwd)" \
    --remote=origin \
    --push

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 저장소 생성 및 푸시 완료!"
    echo ""
    echo "저장소: https://github.com/$GITHUB_USERNAME/portfolio-signal-mailer"
else
    echo ""
    echo "❌ 저장소 생성 실패"
    exit 1
fi

# Step 4: GitHub Secrets 설정 안내
echo ""
echo "=========================================="
echo "【Step 4】GitHub Secrets 설정"
echo "=========================================="
echo ""
echo "다음 URL에서 Secrets을 설정하세요:"
echo "https://github.com/$GITHUB_USERNAME/portfolio-signal-mailer/settings/secrets/actions"
echo ""
echo "필요한 Secrets:"
echo "  1. SENDER_EMAIL: gamjatangjo@gmail.com"
echo "  2. SENDER_PASSWORD: [Gmail 앱 비밀번호]"
echo "  3. RECIPIENT_EMAIL: gamjatangjo@gmail.com"
echo "  4. GEMINI_API_KEY: AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA"
echo ""
echo "=========================================="
echo "✅ 모든 설정이 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음: GitHub Secrets 설정 후 Actions 확인"

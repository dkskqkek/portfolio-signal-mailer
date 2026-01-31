# 🚀 자동 배포 완료!

## 📋 준비 완료 항목

✅ GitHub 저장소 생성 스크립트
✅ 코드 푸시 스크립트  
✅ Secrets 설정 가이드
✅ 모든 파일 커밋 완료


## 🔑 1단계: GitHub Personal Access Token 생성

GitHub에서 토큰을 생성하세요:
**https://github.com/settings/tokens/new**

생성할 때:
1. Token name: "Portfolio Signal Mailer"
2. Expiration: "90 days" 또는 "No expiration"
3. Select scopes:
   ✓ repo (전체)
   ✓ workflow
4. "Generate token" 클릭
5. **토큰 복사** (다시 보이지 않음!)


## ⚡ 2단계: 자동 배포 스크립트 실행

### Windows (권장):
```batch
cd d:\gg
auto_deploy.bat
```

### Linux/Mac:
```bash
cd d/gg
bash auto_deploy.sh
```

스크립트가 물어보는 항목:
1. GitHub Personal Access Token (위에서 생성)
2. GitHub 사용자명 (예: gamja-user)

그러면:
✅ 저장소 자동 생성 (portfolio-signal-mailer)
✅ 코드 자동 푸시
✅ GitHub Actions 활성화


## 🔐 3단계: GitHub Secrets 설정

스크립트 완료 후 자동으로 URL이 표시됩니다:
**https://github.com/YOUR_USERNAME/portfolio-signal-mailer/settings/secrets/actions**

다음 4가지 추가:

| Secret | Value |
|--------|-------|
| SENDER_EMAIL | gamjatangjo@gmail.com |
| SENDER_PASSWORD | [Gmail 앱 비밀번호] |
| RECIPIENT_EMAIL | gamjatangjo@gmail.com |
| GEMINI_API_KEY | AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA |


## ✅ 완료!

모든 설정이 완료되면:
- 매일 UTC 0시(KST 오전 9시)에 자동 실행
- 신호 상태 변화 시 메일 수신
- 컴퓨터 켜지 않아도 작동


## 🆘 문제 해결

### "gh: command not found"
→ GitHub CLI 재설치: https://cli.github.com/

### "Authentication failed"
→ Personal Access Token이 올바른지 확인

### "Repository already exists"
→ 다른 저장소명으로 변경하거나 기존 저장소 삭제


## 📝 수동 방법 (스크립트 안되면)

```bash
cd d:\gg

# 1. GitHub에 로그인
gh auth login -w --git-protocol https

# 2. 저장소 생성
gh repo create portfolio-signal-mailer --public --source=. --remote=origin --push

# 3. 완료!
```

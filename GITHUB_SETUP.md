# GitHub 자동 배포 가이드

## 🚀 1단계: GitHub 저장소 생성

1. GitHub 로그인: https://github.com
2. 새 저장소 생성:
   - 저장소 이름: `portfolio-signal-mailer`
   - 설명: "자동 포트폴리오 신호 감지 및 메일 발송 시스템"
   - Public 선택 (Private도 가능)
   - "Create repository" 클릭

## 🔑 2단계: GitHub Secrets 설정 (매우 중요)

**절대로 API 키나 비밀번호를 코드에 하드코딩하지 마세요!**

저장소 페이지에서:
1. Settings → Secrets and variables → Actions
2. "New repository secret" 클릭
3. 다음 항목 추가:

### 필수 Secrets:

| Name | Value | 설명 |
|------|-------|------|
| SENDER_EMAIL | gamjatangjo@gmail.com | 발송 이메일 |
| SENDER_PASSWORD | [Gmail 앱 비밀번호] | Gmail 앱 비밀번호 (생성 방법 참고) |
| RECIPIENT_EMAIL | gamjatangjo@gmail.com | 수신 이메일 |
| GEMINI_API_KEY | AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA | Gemini API 키 |

### Gmail 앱 비밀번호 생성:
1. https://myaccount.google.com/apppasswords
2. 계정에 2단계 인증 활성화되어 있어야 함
3. "기타(사용자 정의 이름)" 선택
4. "Portfolio Signal Mailer" 입력
5. 생성된 16자리 비밀번호 복사
6. GitHub Secrets → SENDER_PASSWORD 에 붙여넣기

## 📝 3단계: Git 파일 추가 및 푸시

터미널에서:

```bash
cd d:/gg

# 파일 추가
git add .

# 변경사항 확인
git status

# 커밋
git commit -m "Initial commit: Portfolio signal mailer with GitHub Actions"

# GitHub 저장소 연결 (YOUR_USERNAME과 저장소명 수정)
git remote add origin https://github.com/YOUR_USERNAME/portfolio-signal-mailer.git

# 메인 브랜치로 푸시
git branch -M main
git push -u origin main
```

## ✅ 4단계: GitHub Actions 활성화 확인

저장소 페이지에서:
1. "Actions" 탭 클릭
2. "Portfolio Signal Mailer" 워크플로우 확인
3. 자동으로 매일 UTC 0시(KST 오전 9시)에 실행됨

### 수동 실행 방법:
- Actions → Portfolio Signal Mailer → Run workflow → Run workflow

## 📊 5단계: 실행 모니터링

### 매일 자동 실행:
- **시간**: 매일 UTC 0시 (KST 오전 9시)
- **빈도**: 일 1회

### 로그 확인:
1. Actions 탭에서 최신 실행 클릭
2. signal-mailer 작업 → 각 스텝 확인

### 메일 수신:
- 신호 상태 변화 시 gamjatangjo@gmail.com로 수신

## 🔧 6단계: 로컬 개발 (선택사항)

로컬에서 테스트하려면:

```bash
# 의존성 설치
pip install -r requirements.txt

# config.yaml 생성 (환경 변수 필요)
set SENDER_EMAIL=gamjatangjo@gmail.com
set SENDER_PASSWORD=your_app_password
set RECIPIENT_EMAIL=gamjatangjo@gmail.com
set GEMINI_API_KEY=AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA

python signal_mailer/create_config.py

# 신호 감지 테스트
python signal_mailer/run_once.py
```

## ⚠️ 7단계: 주의사항

### 보안:
- ❌ API 키를 코드나 설정 파일에 저장하지 마세요
- ✅ GitHub Secrets을 반드시 사용하세요
- ✅ .gitignore에 민감한 파일 추가됨

### 비용:
- GitHub Actions: 공개 저장소는 무제한 무료
- Gmail: 무료 계정 사용 가능
- Gemini API: 무료 tier 사용 가능

### 신뢰성:
- GitHub Actions는 99.9% 가용성 보장
- 메일 발송 실패 시 로그에 기록됨
- 신호 이력은 자동으로 커밋됨

## 🆘 문제 해결

### 1. "Authentication failed" 오류
→ SENDER_PASSWORD가 올바른 Gmail 앱 비밀번호인지 확인

### 2. "신호가 감지되지만 메일이 오지 않음"
→ 스팸 폴더 확인 또는 메일 필터 설정 확인

### 3. 워크플로우가 실행되지 않음
→ GitHub Secrets이 모두 설정되었는지 확인

### 4. "signal_history.json 업데이트 안됨"
→ GitHub token이 자동으로 생성됨 (조치 불필요)

## 💡 추가 기능

### 실행 시간 변경:
`.github/workflows/signal_mailer.yml` 수정:
```yaml
on:
  schedule:
    - cron: '0 10 * * *'  # UTC 10시 = KST 오후 7시로 변경
```

### 여러 이메일로 발송:
`signal_mailer/mailer.py` 수정:
```python
recipients = ['email1@gmail.com', 'email2@gmail.com']
for recipient in recipients:
    self.mailer.send_email(subject, body, recipient)
```

## 📚 리소스

- GitHub Actions 문서: https://docs.github.com/actions
- Gmail 앱 비밀번호: https://myaccount.google.com/apppasswords
- Gemini API 문서: https://ai.google.dev/

---

모든 설정이 완료되면 컴퓨터를 켠 상태로 두지 않아도 
GitHub Actions가 매일 자동으로 신호를 감지하고 메일을 발송합니다! 🎯

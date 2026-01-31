# 🚀 GitHub Actions 클라우드 자동 배포 완료!

## ✅ 완료된 항목

### 1. GitHub Actions 워크플로우 설정
- `.github/workflows/signal_mailer.yml` 생성됨
- 매일 UTC 0시(KST 오전 9시) 자동 실행
- 환경 변수 기반 안전한 설정

### 2. 클라우드 실행 스크립트
- `signal_mailer/run_once.py` - 한 번만 실행하는 스크립트
- `signal_mailer/create_config.py` - 환경 변수로 config 생성
- GitHub Actions에서 자동 실행

### 3. Git 저장소 초기화
- `.git` 폴더 생성
- 모든 파일 커밋 완료
- GitHub 푸시 준비 완료

### 4. 보안 설정
- `.gitignore` - 민감한 파일 제외
- GitHub Secrets 사용
- API 키 안전 저장

---

## 🔑 다음 단계 (5분)

### Step 1: GitHub 저장소 생성 (1분)
```
1. https://github.com/new 방문
2. 저장소명: portfolio-signal-mailer
3. Public 선택 (Private도 가능)
4. "Create repository" 클릭
```

### Step 2: GitHub에 코드 푸시 (2분)
```bash
cd d:/gg

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-signal-mailer.git
git push -u origin main
```

**YOUR_USERNAME을 자신의 GitHub 사용자명으로 바꾸세요!**

### Step 3: GitHub Secrets 설정 (2분)
```
저장소 페이지 → Settings → Secrets and variables → Actions
```

**다음 4가지 추가:**

| Secret Name | Value |
|------------|-------|
| SENDER_EMAIL | gamjatangjo@gmail.com |
| SENDER_PASSWORD | [Gmail 앱 비밀번호] |
| RECIPIENT_EMAIL | gamjatangjo@gmail.com |
| GEMINI_API_KEY | AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA |

#### Gmail 앱 비밀번호 생성 (필수!)
1. https://myaccount.google.com/apppasswords
2. 2단계 인증 활성화 필수
3. "기타(사용자 정의 이름)" 선택
4. "Portfolio Signal Mailer" 입력
5. 생성된 16자리 비밀번호 복사
6. SENDER_PASSWORD에 붙여넣기

---

## 🎯 자동 실행 확인

### 매일 실행됨
- **시간**: UTC 0시 = KST 오전 9시
- **빈도**: 1일 1회
- **비용**: 무료 ✅

### 실행 확인 방법
```
저장소 → Actions → Portfolio Signal Mailer
```

### 수동 실행 (테스트)
```
Actions → Portfolio Signal Mailer → Run workflow → Run workflow
```

---

## 📧 신호 수신

신호 상태 변화 시:
- **발송자**: GitHub Actions Bot
- **수신자**: gamjatangjo@gmail.com
- **내용**: 상세한 신호 리포트 + 포지션 조정 가이드

예시 이메일:
```
제목: [포트폴리오 신호] QQQ->XLP 전환 DANGER (QQQ->XLP 전환)

본문:
포트폴리오 신호 리포트
============================================================

[감지 시간]
2026-01-21 09:00:00

[신호 상태]
DANGER (QQQ->XLP 전환)

[상태 변화]
✓ 변화 있음 (메일 발송 필요)

[상세 정보]
- 20일 이동평균: -0.001234
- 20일 변동성: 0.025678
...
```

---

## 🔒 보안 팁

### ✅ 안전하게 저장됨
- GitHub Secrets 사용
- 환경 변수로 로드
- 코드에 노출 안됨
- 로그에 마스킹됨

### ❌ 절대 하지 말 것
- API 키를 코드에 하드코딩
- 비밀번호를 .gitignore에서 제외
- 민감한 정보를 public으로 푸시

### ✅ 하면 좋을 것
- 정기적으로 API 키 로테이션
- GitHub Secrets 권한 제한
- 두 명 이상이 관리할 경우 Team 설정

---

## 📊 신호 이력 자동 저장

매 실행 후 자동으로:
```
signal_mailer/signal_history.json
```
에 저장되고 GitHub에 커밋됩니다.

모든 신호의 기록을 확인할 수 있습니다:
```
저장소 → signal_mailer/signal_history.json
```

---

## ⚙️ 커스터마이징

### 실행 시간 변경
`.github/workflows/signal_mailer.yml` 수정:
```yaml
cron: '0 10 * * *'  # UTC 10시 = KST 오후 6시
```

### 신호 기준 변경
`signal_mailer/signal_detector.py` 수정:
```python
ma_threshold = np.nanpercentile(ma20_returns, 30)  # 25 → 30
vol_threshold = np.nanpercentile(std20_returns, 70)  # 75 → 70
```

### 추가 이메일 발송
`signal_mailer/run_once.py` 수정:
```python
recipients = ['email1@gmail.com', 'email2@gmail.com']
for recipient in recipients:
    self.mailer.send_email(subject, body, recipient)
```

---

## 🆘 문제 해결

### Q: GitHub에 푸시할 수 없습니다
**A:** 비밀번호 대신 Personal Access Token 사용:
```bash
git remote set-url origin https://YOUR_TOKEN@github.com/YOUR_USERNAME/portfolio-signal-mailer.git
```

### Q: 메일이 오지 않습니다
**A:** 확인 사항:
1. GitHub Actions 실행 로그 확인
2. Secrets가 모두 설정되었는지 확인
3. Gmail 스팸 폴더 확인
4. Gmail 앱 비밀번호가 정확한지 확인

### Q: Actions 실행 안됨
**A:** 확인 사항:
1. `.github/workflows/signal_mailer.yml` 파일 존재 확인
2. 저장소 → Settings → Actions 에서 활성화 확인
3. Secrets이 모두 설정되었는지 확인

### Q: 실행은 되는데 신호 발송 안됨
**A:** 신호가 없는 정상 상태일 수 있습니다:
- 상태 변화가 있어야만 메일 발송
- signal_history.json 확인

---

## 💡 모니터링

### 주간 점검
```
저장소 → Actions 에서 최근 실행 결과 확인
```

### 신호 이력 확인
```
signal_mailer/signal_history.json 에서 JSON 데이터 확인
```

### 로그 확인
```
Actions → 최신 실행 → signal-mailer 클릭 → 각 스텝 확인
```

---

## 🎉 완료!

이제 더 이상 컴퓨터를 켜 둘 필요가 없습니다!

GitHub Actions가 매일 자동으로:
1. ✅ SPY 데이터 수집
2. ✅ 20일 이동평균 계산
3. ✅ 변동성 분석
4. ✅ QQQ→XLP 신호 판정
5. ✅ 상태 변화 시 메일 발송
6. ✅ 신호 이력 저장

24시간 당신의 포트폴리오를 지켜줍니다! 💼

---

## 📞 추가 도움

더 자세한 내용:
- `GITHUB_SETUP.md` - 상세 설정 가이드
- `QUICK_START.md` - 빠른 시작 가이드
- `signal_mailer/README.md` - 신호 메일러 상세 설명

GitHub Actions 문서:
- https://docs.github.com/actions

---

**모든 설정이 완료되었습니다! GitHub 저장소를 만들고 푸시하기만 하면 됩니다!** 🚀

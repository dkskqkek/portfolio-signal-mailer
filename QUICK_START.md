포트폴리오 신호 메일러 - GitHub Actions 자동 배포
=================================================

📌 개요
-------
이 시스템은 GitHub Actions를 사용하여 클라우드에서 자동으로 실행됩니다.
컴퓨터를 켜지 않아도 매일 오전 9시(KST)에 신호를 감지하고 메일을 발송합니다.


🚀 빠른 시작 (3분)
-------------------

1️⃣ GitHub 저장소 생성
   - https://github.com/new
   - 저장소명: portfolio-signal-mailer
   - Public 선택
   - Create repository

2️⃣ 로컬에서 푸시
   
   Windows:
   $ setup_github.bat
   
   Linux/Mac:
   $ bash setup_github.sh
   
   그 다음:
   $ git branch -M main
   $ git push -u origin main

3️⃣ GitHub Secrets 설정
   
   https://github.com/YOUR_USERNAME/portfolio-signal-mailer/settings/secrets/actions
   
   다음 4가지 추가:
   
   ┌─────────────────────────────────────────────────────────┐
   │ SENDER_EMAIL                                            │
   │ gamjatangjo@gmail.com                                   │
   └─────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────┐
   │ SENDER_PASSWORD                                         │
   │ [Gmail 앱 비밀번호 - 아래 참고]                         │
   └─────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────┐
   │ RECIPIENT_EMAIL                                         │
   │ gamjatangjo@gmail.com                                   │
   └─────────────────────────────────────────────────────────┘
   
   ┌─────────────────────────────────────────────────────────┐
   │ GEMINI_API_KEY                                          │
   │ AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA               │
   └─────────────────────────────────────────────────────────┘

✅ 완료! 매일 오전 9시에 자동 실행됩니다.


📧 Gmail 앱 비밀번호 생성 (필수)
--------------------------------

1. https://myaccount.google.com/apppasswords
2. "기타(사용자 정의 이름)" 선택
3. "Portfolio Signal Mailer" 입력 후 생성
4. 생성된 16자리 비밀번호 복사
5. GitHub Secrets → SENDER_PASSWORD에 붙여넣기


🔍 실행 모니터링
----------------

GitHub Actions 확인:
1. 저장소 → Actions 탭
2. "Portfolio Signal Mailer" 워크플로우
3. 매일 UTC 0시(KST 오전 9시) 자동 실행

수동 실행:
1. Actions → Portfolio Signal Mailer
2. Run workflow → Run workflow 클릭


📊 신호 이력
------------

GitHub 저장소에 자동 저장:
- signal_mailer/signal_history.json
- 매 실행 후 커밋됨
- 모든 신호 기록 포함


⏰ 실행 일정
-----------

기본값:
- 시간: 매일 UTC 0시 = KST 오전 9시
- 빈도: 1일 1회

변경하려면:
.github/workflows/signal_mailer.yml 의 cron 값 수정:

현재:    '0 0 * * *'      (UTC 0시 = KST 9시)
오후 6시: '9 * * *'        (UTC 9시 = KST 오후 6시)
자정:    '15 0 * * *'     (UTC 15시 = KST 자정)


🔄 업데이트
-----------

로컬에서 수정한 후:

$ git add .
$ git commit -m "Update signal detection logic"
$ git push


🆘 문제 해결
-----------

Q: 메일이 오지 않습니다
A: 1. 스팸 폴더 확인
   2. GitHub Actions 로그에서 오류 확인
   3. Secrets이 올바른지 확인

Q: API 키 노출 걱정
A: Secrets에 저장되므로 안전합니다
   - 코드에는 표시되지 않음
   - 로그에도 마스킹됨
   - GitHub는 엔드-투-엔드 암호화 사용

Q: 비용이 드나요?
A: 아니오. 무료입니다!
   - GitHub Actions: 공개 저장소 무제한 무료
   - Gmail: 무료 계정 가능
   - Gemini API: 무료 tier 가능


💡 추가 팁
----------

1. Slack 알림 추가
   → GitHub Actions → Slack integration

2. 여러 이메일로 발송
   → RECIPIENT_EMAIL에 쉼표로 구분

3. 시장 시간만 실행
   → cron 수정으로 특정 시간 제한

4. 신호 로그 이메일 첨부
   → mailer.py 수정


📚 파일 구조
-----------

portfolio-signal-mailer/
├── .github/
│   └── workflows/
│       └── signal_mailer.yml      # GitHub Actions 설정
├── signal_mailer/
│   ├── main.py                    # 로컬 실행용 (선택사항)
│   ├── run_once.py                # GitHub Actions 용
│   ├── create_config.py           # 설정 생성
│   ├── signal_detector.py         # 신호 감지
│   ├── mailer.py                  # 메일 발송
│   ├── config.yaml                # ⚠️ .gitignore에 포함
│   └── signal_history.json        # 신호 이력
├── requirements.txt               # 파이썬 의존성
├── .gitignore                     # Git 제외 파일
└── README.md                      # 프로젝트 설명


🎯 다음 단계
-----------

1. ✅ GitHub 저장소 생성
2. ✅ 로컬 푸시 (setup_github.bat/sh)
3. ✅ GitHub Secrets 설정
4. ✅ 자동 실행 확인

완료되면 매일 오전 9시마다 자동으로 실행됩니다! 🚀


📞 추가 지원
-----------

문제가 있으시면:
1. GitHub Issues 생성
2. Actions 로그 확인
3. signal_mailer/mailer.log 검토


더 이상 컴퓨터를 켜 둘 필요가 없습니다!
GitHub Actions가 24시간 당신의 포트폴리오를 지켜줍니다. 💼

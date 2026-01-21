포트폴리오 신호 메일러 설정 가이드
=====================================

## 1단계: Gmail 앱 비밀번호 생성 (Gmail 사용자만)

1. Google 계정 로그인: https://myaccount.google.com
2. 왼쪽 메뉴 "보안" 클릭
3. 하단 "2단계 인증" 활성화 (이미 활성화됨면 스킵)
4. "앱 비밀번호" 클릭
5. "기타(사용자 정의 이름)"에서 "Windows 컴퓨터" 또는 "기타 기기" 선택
6. 이름 입력 예: "포트폴리오 신호 메일러"
7. 생성된 16자리 비밀번호 복사

## 2단계: config.yaml 설정

파일 위치: d:\gg\signal_mailer\config.yaml

다음 항목 수정:

email:
  smtp_server: "smtp.gmail.com"              # Gmail 기본값
  smtp_port: 587                              # Gmail 기본값
  sender_email: "your_email@gmail.com"        # <- 여기 수정
  sender_password: "your_app_password"        # <- 여기 수정 (위의 16자리)
  recipient_email: "your_email@gmail.com"     # <- 여기 수정

scheduler:
  run_time: "09:00"                          # 매일 아침 9시 (원하는 시간으로 수정 가능)

## 3단계: 테스트 실행

터미널에서 다음 명령 실행:

cd d:\gg\signal_mailer

# 신호 감지 테스트
python main.py --test-signal

# 이메일 발송 테스트  
python main.py --test-email

## 4단계: 자동 실행 시작

터미널에서:
python main.py

(스케줄러 시작됨. Ctrl+C로 종료)

## 동작 방식

1. 매일 아침 설정된 시간에 자동 실행
2. SPY의 20일 이동평균과 변동성 분석
3. 위험신호 판단:
   - 20일 MA < 25 percentile 또는
   - 20일 변동성 > 75 percentile
4. 신호 상태 변화가 있을 때만 메일 발송
5. 모든 신호 이력은 signal_history.json에 저장
6. 실행 로그는 mailer.log에 저장

## 신호 해석

[신호] NORMAL (QQQ 유지)
- 정상 상태
- 포지션 조정 불필요
- SCHD 34% + QQQ 34% + XLP 0% + KOSPI 17% + GOLD 15%

[신호] DANGER (QQQ->XLP 전환)
- 위험신호 발생
- QQQ를 XLP로 전환 권장
- SCHD 34% + QQQ 0% + XLP 34% + KOSPI 17% + GOLD 15%

## 다른 이메일 서비스 설정

### Naver 메일
email:
  smtp_server: "smtp.naver.com"
  smtp_port: 587
  sender_email: "your_naver_id@naver.com"
  sender_password: "your_naver_password"

### Outlook/Hotmail
email:
  smtp_server: "smtp-mail.outlook.com"
  smtp_port: 587
  sender_email: "your_email@outlook.com"
  sender_password: "your_password"

## 문제 해결

1. "이메일 설정이 불완전" 오류
   -> config.yaml에서 sender_email, sender_password, recipient_email이 입력되었는지 확인

2. "로그인 실패" 오류 (Gmail)
   -> Gmail 계정에서 앱 비밀번호가 올바르게 생성되었는지 확인
   -> "2단계 인증"이 활성화되어 있는지 확인

3. "연결 거부" 오류
   -> SMTP 서버 주소와 포트가 올바른지 확인
   -> 방화벽에서 587번 포트 차단 여부 확인

4. 메일이 스팸 폴더에 들어가는 경우
   -> 신뢰할 수 있는 발신자로 등록

## 유용한 명령어

# 현재 신호 상태 확인 (한번 실행)
python main.py --test-signal

# 테스트 메일 발송
python main.py --test-email

# 신호 이력 확인
type signal_history.json

# 로그 실시간 확인
tail -f mailer.log  (Linux/Mac)
또는 텍스트 에디터로 mailer.log 열기

## 자동 시작 설정 (Windows)

Windows 작업 스케줄러를 사용하여 컴퓨터 시작 시 자동으로 실행하도록 설정할 수 있습니다:

1. 작업 스케줄러 실행
2. 기본 작업 만들기
3. 이름: "포트폴리오 신호 메일러"
4. 트리거: "컴퓨터 시작할 때" 또는 특정 시간
5. 작업: C:\Python313\python.exe d:\gg\signal_mailer\main.py

질문이 있으시면 로그(mailer.log)를 확인하세요.

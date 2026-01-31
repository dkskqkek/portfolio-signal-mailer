# -*- coding: utf-8 -*-
import datetime


def log_conversation(
    topic: str, description: str, log_file: str = "d:/gg/FULL_CONVERSATION_LOG.txt"
):
    """
    작업 내역을 FULL_CONVERSATION_LOG.txt에 기록합니다.

    Args:
        topic: 작업 주제
        description: 상세 작업 내용
        log_file: 로그 파일 경로
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}] TOPIC: {topic}\n"
    log_entry += f"DESCRIPTION: {description}\n"
    log_entry += "-" * 50 + "\n"

    try:
        # 파일이 없으면 생성, 있으면 이어서 쓰기
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"[LOG] {topic} 기록 완료.")
    except Exception as e:
        print(f"[ERROR] 로그 기록 실패: {e}")


if __name__ == "__main__":
    # 간단한 테스트
    log_conversation("System initialization", "Logger utility created.")

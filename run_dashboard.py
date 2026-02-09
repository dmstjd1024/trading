#!/usr/bin/env python3
"""
대시보드 실행 스크립트

사용법:
    python run_dashboard.py

또는 직접 streamlit 실행:
    streamlit run dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent
    app_path = project_root / "dashboard" / "app.py"

    if not app_path.exists():
        print(f"오류: {app_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    print("한국주식 트레이딩 대시보드를 시작합니다...")
    print(f"앱 경로: {app_path}")
    print()

    # streamlit 실행
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                "--server.headless", "true",
            ],
            cwd=str(project_root),
        )
    except KeyboardInterrupt:
        print("\n대시보드를 종료합니다.")
    except FileNotFoundError:
        print("오류: streamlit이 설치되지 않았습니다.")
        print("설치: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()

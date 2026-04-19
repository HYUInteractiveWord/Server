from pathlib import Path
from app.services.pronunciation.service import (
    analyze_pronunciation,
    save_all_pitch_graphs,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TTS_DIR = DATASET_DIR / "tts"
USER_DIR = DATASET_DIR / "user"
OUTPUT_DIR = BASE_DIR / "outputs"

# 자동 생성
TTS_DIR.mkdir(parents=True, exist_ok=True)
USER_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 테스트 파일명 설정
# ---------------------------
TTS_FILE = "사과.wav"
USER_FILE = "안녕하세요.wav"

tts_path = str(TTS_DIR / TTS_FILE)
user_path = str(USER_DIR / USER_FILE)


def print_result_block(scores: dict):
    print("\n===== 분석 결과 =====")
    print(f"Pronunciation Score : {scores['pronunciation_score']:.2f}")
    print(f"Pitch Score         : {scores['pitch_score']:.2f}")
    print(f"Duration Score      : {scores['duration_score']:.2f}")
    print(f"Intensity Score     : {scores['intensity_score']:.2f}")
    print(f"Final Score         : {scores['final_score']:.2f}")


def main():
    print("=== 전체 분석 ===")
    result = analyze_pronunciation(tts_path, user_path)
    scores = result["scores"]

    # 1. 점수 콘솔 출력
    print_result_block(scores)

    # 2. test용 폴더명 규칙은 여기서만 적용
    tts_stem = Path(TTS_FILE).stem
    user_stem = Path(USER_FILE).stem
    test_output_dir = OUTPUT_DIR / f"{tts_stem}__{user_stem}"

    # 3. 그래프 저장
    graph_result = save_all_pitch_graphs(
        tts_path=tts_path,
        user_path=user_path,
        output_dir=str(test_output_dir),
    )

    # 4. 저장 위치 출력
    print("\n===== 그래프 저장 위치 =====")
    print(f"Folder : {graph_result['graph_dir']}")
    for key, path in graph_result["graph_paths"].items():
        print(f"{key:<25}: {path}")

    # 5. 마지막 총점 강조 출력
    print("\n===== 최종 요약 =====")
    print(f"TOTAL SCORE : {scores['final_score']:.2f}")


if __name__ == "__main__":
    main()


# python -m app.services.pronunciation.test.test_pronunciation

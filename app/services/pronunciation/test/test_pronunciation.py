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
TTS_FILE = "물.wav"
USER_FILE = "안녕하세요.wav"

tts_path = str(TTS_DIR / TTS_FILE)
user_path = str(USER_DIR / USER_FILE)


def print_result_block(scores: dict):
    print("\n===== 🎙️ 분석 결과 (V2 아키텍처 4대 지표) =====")
    # KeyError 방지를 위해 get() 사용 및 항목별 한글 설명 추가
    print(f"Pronunciation (발음/MFCC)   : {scores.get('pronunciation_score', 0):.2f} 점")
    print(f"Phoneme       (음소 정확도) : {scores.get('phoneme_score', 0):.2f} 점")
    print(f"Pitch         (억양 흐름)   : {scores.get('pitch_score', 0):.2f} 점")
    print(f"Duration      (박자/타이밍) : {scores.get('duration_score', 0):.2f} 점")
    print("-" * 45)
    # 강세는 점수가 아닌 Pass/Fail 여부로 출력
    print(f"Intensity Pass (강세 통과)  : {scores.get('intensity_pass', False)}")
    print("===============================================")


def main():
    print("=== 전체 분석 시작 ===")
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
    print("\n===== 📊 그래프 저장 위치 =====")
    print(f"Folder : {graph_result['graph_dir']}")
    for key, path in graph_result["graph_paths"].items():
        print(f"{key:<25}: {path}")

    # 5. 마지막 총점 강조 출력
    print("\n===== 🏆 최종 요약 =====")
    print(f"TOTAL SCORE : {scores.get('final_score', 0):.2f} 점")


if __name__ == "__main__":
    main()

# 실행 명령어:
# python -m app.services.pronunciation.test.test_pronunciation
import os
from pathlib import Path

from .analysis import (
    load_audio_flexible,
    validate_audio,
    preprocess_for_pitch,
    extract_pitch,
    check_voiced_ratio,
)
from .scoring import (
    pronunciation_score_mfcc_dtw,
    pitch_similarity_scores,
    duration_similarity_score,
    intensity_similarity_score,
    calculate_final_score,
)
from .visualization import (
    save_single_pitch_graph,
    save_overlay_pitch_graph,
    save_alignment_summary_graph,
    save_score_summary_graph,
)


def analyze_pronunciation(tts_path: str, user_path: str, sr: int = 16000) -> dict:
    """
    전체 분석 결과 반환
    """
    y_ref, sr_ref = load_audio_flexible(tts_path, sr=sr)
    y_usr, sr_usr = load_audio_flexible(user_path, sr=sr)

    tts_validation = validate_audio(y_ref, sr_ref, "TTS")
    user_validation = validate_audio(y_usr, sr_usr, "USER")

    ref_pitch_proc = preprocess_for_pitch(y_ref, sr_ref)
    usr_pitch_proc = preprocess_for_pitch(y_usr, sr_usr)

    y_ref_pitch = ref_pitch_proc["y"]
    y_usr_pitch = usr_pitch_proc["y"]

    t_ref, f0_ref = extract_pitch(y_ref_pitch, sr_ref)
    t_usr, f0_usr = extract_pitch(y_usr_pitch, sr_usr)

    tts_voiced_ratio = check_voiced_ratio(f0_ref, "TTS")
    user_voiced_ratio = check_voiced_ratio(f0_usr, "USER")

    pron_result = pronunciation_score_mfcc_dtw(
        y_ref, sr_ref,
        y_usr, sr_usr,
    )

    pitch_result = pitch_similarity_scores(
        f0_ref,
        f0_usr,
    )

    duration_result = duration_similarity_score(
        y_ref_pitch, sr_ref,
        y_usr_pitch, sr_usr,
    )

    intensity_result = intensity_similarity_score(
        y_ref_pitch,
        y_usr_pitch,
    )

    final_score = calculate_final_score(
        pronunciation_score=pron_result["score"],
        pitch_score=pitch_result["pitch_score"],
        duration_score=duration_result["duration_score"],
        intensity_score=intensity_result["intensity_score"],
    )

    return {
        "input": {
            "tts_path": tts_path,
            "user_path": user_path,
            "sr": sr,
        },
        "validation": {
            "tts": tts_validation,
            "user": user_validation,
        },
        "voiced_ratio": {
            "tts": float(tts_voiced_ratio),
            "user": float(user_voiced_ratio),
        },
        "scores": {
            "pronunciation_score": float(pron_result["score"]),
            "pitch_score": float(pitch_result["pitch_score"]),
            "duration_score": float(duration_result["duration_score"]),
            "intensity_score": float(intensity_result["intensity_score"]),
            "final_score": float(final_score),
        },
        "pronunciation_detail": {
            "raw_distance": pron_result["raw_distance"],
            "normalized_distance": pron_result["normalized_distance"],
        },
        "pitch_features": {
            "tts": pitch_result["ref_features"],
            "user": pitch_result["usr_features"],
        },
        "duration_detail": duration_result,
        "intensity_detail": intensity_result,
        "plot_data": {
            "t_ref": t_ref,
            "f0_ref": f0_ref,
            "t_usr": t_usr,
            "f0_usr": f0_usr,
        },
    }


def get_scores(tts_path: str, user_path: str, sr: int = 16000) -> dict:
    result = analyze_pronunciation(tts_path, user_path, sr=sr)
    return result["scores"]


def get_pitch_features_only(tts_path: str, user_path: str, sr: int = 16000) -> dict:
    result = analyze_pronunciation(tts_path, user_path, sr=sr)
    return result["pitch_features"]


def save_all_pitch_graphs(
    tts_path: str,
    user_path: str,
    output_dir: str,
    sr: int = 16000,
) -> dict:
    """
    공용 서비스 함수:
    - 전달받은 output_dir에 바로 저장
    - test용 폴더명 규칙은 여기서 처리하지 않음
    """
    result = analyze_pronunciation(tts_path, user_path, sr=sr)

    t_ref = result["plot_data"]["t_ref"]
    f0_ref = result["plot_data"]["f0_ref"]
    t_usr = result["plot_data"]["t_usr"]
    f0_usr = result["plot_data"]["f0_usr"]
    scores = result["scores"]

    final_output_dir = Path(output_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    paths = {
        "tts_pitch": save_single_pitch_graph(
            t_ref,
            f0_ref,
            title="TTS Pitch",
            label="TTS",
            output_path=str(final_output_dir / "01_tts_pitch.png"),
        ),
        "user_pitch": save_single_pitch_graph(
            t_usr,
            f0_usr,
            title="User Pitch",
            label="USER",
            output_path=str(final_output_dir / "02_user_pitch.png"),
        ),
        "pitch_original": save_overlay_pitch_graph(
            t_ref,
            f0_ref,
            t_usr,
            f0_usr,
            output_path=str(final_output_dir / "03_pitch_original.png"),
        ),
        "pitch_alignment_summary": save_alignment_summary_graph(
            t_ref,
            f0_ref,
            t_usr,
            f0_usr,
            output_path=str(final_output_dir / "04_pitch_alignment_summary.png"),
        ),
        "score_summary": save_score_summary_graph(
            scores=scores,
            output_path=str(final_output_dir / "05_score_summary.png"),
        ),
    }

    return {
        "scores": scores,
        "pitch_features": result["pitch_features"],
        "graph_dir": str(final_output_dir),
        "graph_paths": paths,
    }
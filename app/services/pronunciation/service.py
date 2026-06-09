import os
import math
import numpy as np
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
    phoneme_similarity_score,
    calculate_final_score,
)
from .visualization import (
    save_single_pitch_graph,
    save_overlay_pitch_graph,
    save_alignment_summary_graph,
    save_score_summary_graph,
)


def _safe_float_list(arr) -> list:
    """
    Numpy 배열을 List로 변환하면서 NaN(결측치)이나 Inf 값을 0.0으로 치환합니다.
    이 과정을 거치지 않으면 안드로이드 JSON 파서가 에러를 뱉거나 빈 배열로 무시하여 
    유저 피치 그래프가 사라지는 현상이 발생합니다.
    """
    if not hasattr(arr, 'tolist'):
        arr = np.array(arr)
    
    safe_list = []
    for x in arr.tolist():
        if math.isnan(x) or math.isinf(x):
            safe_list.append(0.0)
        else:
            safe_list.append(float(x))
    return safe_list


def analyze_pronunciation(tts_path: str, user_path: str, sr: int = 16000) -> dict:
    """
    전체 분석 결과 반환 (안드로이드 DTO 스펙 완벽 대응)
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

    phoneme_result = phoneme_similarity_score(
        y_ref_pitch, sr_ref,
        y_usr_pitch, sr_usr
    )

    if not intensity_result["is_pass"]:
        pron_result["score"] = 0.0
        pitch_result["pitch_score"] = 0.0
        duration_result["duration_score"] = 0.0
        phoneme_result["phoneme_score"] = 0.0
        final_score = 0.0
    else:
        # Pass일 경우에만 정상적으로 최종 점수 계산
        final_score = calculate_final_score(
            pronunciation_score=pron_result["score"],
            pitch_score=pitch_result["pitch_score"],
            duration_score=duration_result["duration_score"],
            phoneme_score=phoneme_result["phoneme_score"],
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
        
        "details": {
            "pronunciation": float(pron_result["score"]),
            "phoneme": float(phoneme_result["phoneme_score"]),
            "pitch": float(pitch_result["pitch_score"]),
            "timing": float(duration_result["duration_score"]),
            "is_intensity_good": intensity_result["is_pass"]
        },
        
        "raw_graph_data": {
            "tts_time": _safe_float_list(t_ref),
            "tts_pitch": _safe_float_list(f0_ref),
            "user_time": _safe_float_list(t_usr),
            "user_pitch": _safe_float_list(f0_usr),
        },
        
        # 기존 백엔드 내부 로직을 위해 유지
        "scores": {
            "pronunciation_score": float(pron_result["score"]),
            "pitch_score": float(pitch_result["pitch_score"]),
            "duration_score": float(duration_result["duration_score"]),
            "phoneme_score": float(phoneme_result["phoneme_score"]),
            "intensity_pass": intensity_result["is_pass"],          
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
        "phoneme_detail": phoneme_result,
        
        # 서버에서 이미지 그릴 때 사용하는 데이터
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
        "details": result["details"],          
        "raw_graph_data": result["raw_graph_data"], 
        "pitch_features": result["pitch_features"],
        "graph_dir": str(final_output_dir),
        "graph_paths": paths,
    }
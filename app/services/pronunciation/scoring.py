import numpy as np
import librosa
from librosa.sequence import dtw

from .analysis import (
    extract_mfcc,
    extract_intensity,
    pitch_features,
    preprocess_for_pronunciation,
    resample_array,
    safe_mean,
)


def normalize_score_from_error(error_ratio: float) -> float:
    return max(0.0, 100.0 * (1.0 - error_ratio))


def calculate_similarity_score(distance: float, scale: float = 100.0) -> float:
    return max(0.0, 100.0 - scale * distance)


def pronunciation_score_mfcc_dtw(y_ref, sr_ref: int, y_usr, sr_usr: int) -> dict:
    """
    pronunciation score:
    - 무음 제거
    - onset 정렬
    - 볼륨 정규화
    - MFCC 추출
    - DTW 거리 계산
    """
    ref_proc = preprocess_for_pronunciation(y_ref, sr_ref)
    usr_proc = preprocess_for_pronunciation(y_usr, sr_usr)

    y_ref_p = ref_proc["y"]
    y_usr_p = usr_proc["y"]

    mfcc_ref = extract_mfcc(y_ref_p, sr_ref)
    mfcc_usr = extract_mfcc(y_usr_p, sr_usr)

    D, wp = dtw(
        X=mfcc_ref,
        Y=mfcc_usr,
        metric="euclidean",
    )

    raw_distance = D[-1, -1]
    path_len = len(wp)
    normalized_distance = raw_distance / max(path_len, 1)

    score = max(0.0, 100.0 * np.exp(-0.015 * normalized_distance))

    return {
        "raw_distance": float(raw_distance),
        "normalized_distance": float(normalized_distance),
        "score": float(score),
        "dtw_matrix": D,
        "warping_path": wp,
    }


def pitch_similarity_scores(f0_ref, f0_usr) -> dict:
    ref_feat = pitch_features(f0_ref)
    usr_feat = pitch_features(f0_usr)

    ref_mean = ref_feat["mean_f0"]
    usr_mean = usr_feat["mean_f0"]
    mean_error = abs(ref_mean - usr_mean) / max(ref_mean, 1e-6)
    pitch_level_score = normalize_score_from_error(mean_error)

    ref_range = ref_feat["range_f0"]
    usr_range = usr_feat["range_f0"]
    if ref_range > 0:
        range_error = abs(ref_range - usr_range) / ref_range
    else:
        range_error = 1.0
    pitch_range_score = normalize_score_from_error(range_error)

    ref_resampled = resample_array(f0_ref, target_len=200)
    usr_resampled = resample_array(f0_usr, target_len=200)

    def z_norm(x):
        std = np.std(x)
        if std < 1e-6:
            return np.zeros_like(x)
        return (x - np.mean(x)) / std

    ref_z = z_norm(ref_resampled)
    usr_z = z_norm(usr_resampled)

    contour_distance = np.mean(np.abs(ref_z - usr_z))
    pitch_contour_score = calculate_similarity_score(
        contour_distance,
        scale=35.0,
    )

    pitch_score = (
        0.3 * pitch_level_score +
        0.3 * pitch_range_score +
        0.4 * pitch_contour_score
    )

    return {
        "pitch_level_score": float(pitch_level_score),
        "pitch_range_score": float(pitch_range_score),
        "pitch_contour_score": float(pitch_contour_score),
        "pitch_score": float(pitch_score),
        "ref_features": ref_feat,
        "usr_features": usr_feat,
    }


def duration_similarity_score(y_ref, sr_ref: int, y_usr, sr_usr: int) -> dict:
    ref_duration = librosa.get_duration(y=y_ref, sr=sr_ref)
    usr_duration = librosa.get_duration(y=y_usr, sr=sr_usr)

    error = abs(ref_duration - usr_duration) / max(ref_duration, 1e-6)
    score = normalize_score_from_error(error)

    return {
        "ref_duration": float(ref_duration),
        "usr_duration": float(usr_duration),
        "duration_score": float(score),
    }


def intensity_similarity_score(y_ref, y_usr) -> dict:
    _, rms_ref = extract_intensity(y_ref)
    _, rms_usr = extract_intensity(y_usr)

    ref_mean = safe_mean(rms_ref)
    usr_mean = safe_mean(rms_usr)

    error = abs(ref_mean - usr_mean) / max(ref_mean, 1e-6)
    score = normalize_score_from_error(error)

    return {
        "ref_mean_rms": float(ref_mean),
        "usr_mean_rms": float(usr_mean),
        "intensity_score": float(score),
    }


def calculate_final_score(
    pronunciation_score: float,
    pitch_score: float,
    duration_score: float,
    intensity_score: float,
) -> float:
    final_score = (
        0.55 * pronunciation_score +
        0.20 * pitch_score +
        0.15 * duration_score +
        0.10 * intensity_score
    )
    return float(final_score)
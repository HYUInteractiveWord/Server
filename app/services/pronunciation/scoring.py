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
    extract_formant_sequence, 
    extract_plosive_sequence,
)


def normalize_score_from_error(error_ratio: float) -> float:
    return max(0.0, 100.0 * (1.0 - error_ratio))


def calculate_similarity_score(distance: float, scale: float = 100.0) -> float:
    return max(0.0, 100.0 - scale * distance)


def pronunciation_score_mfcc_dtw(y_ref, sr_ref: int, y_usr, sr_usr: int) -> dict:
    ref_proc = preprocess_for_pronunciation(y_ref, sr_ref)
    usr_proc = preprocess_for_pronunciation(y_usr, sr_usr)

    mfcc_ref = extract_mfcc(ref_proc["y"], sr_ref)
    mfcc_usr = extract_mfcc(usr_proc["y"], sr_usr)

    D, wp = dtw(X=mfcc_ref, Y=mfcc_usr, metric="euclidean")

    local_distances = []
    for ref_idx, usr_idx in wp:
        dist = np.linalg.norm(mfcc_ref[:, ref_idx] - mfcc_usr[:, usr_idx])
        local_distances.append(float(dist))
    
    f0 = librosa.yin(usr_proc["y"], fmin=50, fmax=500)
    valid_f0 = f0[f0 > 0]
    user_avg_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 150.0

    is_child = user_avg_pitch > 280.0


    def calculate_score(error_val: float, max_score: float) -> float:
        if error_val == 0.0:
            return max_score
            
        if is_child:
            t1, t2 = 8.5, 13.5    
            k1, k2, k3 = 0.001, 0.02, 0.1    
        else:
            t1, t2 = 5.5, 9.5    
            k1, k2, k3 = 0.005, 0.03, 0.15
        
        if error_val <= t1:
            return max_score * np.exp(-k1 * error_val)
        elif error_val <= t2:
            b1 = max_score * np.exp(-k1 * t1)
            return b1 * np.exp(-k2 * (error_val - t1))
        else:
            b1 = max_score * np.exp(-k1 * t1)
            b2 = b1 * np.exp(-k2 * (t2 - t1))
            return b2 * np.exp(-k3 * (error_val - t2)) 
        
    # ---------------------------------------------------------
    # 2. [파트 1] 전체 오차 평균 (Global Score) -> 50점 만점
    # ---------------------------------------------------------
    global_error = np.mean(local_distances) if local_distances else 0.0
    global_score = calculate_score(global_error, max_score=50.0)


    # ---------------------------------------------------------
    # 3. [파트 2] 구역별 점수 부여 (Local Chunk Scores) -> 각 10점 x 5구역 = 50점 만점
    # ---------------------------------------------------------
    num_chunks = max(1, min(5, len(local_distances)))
    chunk_size = max(1, len(local_distances) // num_chunks)
    
    chunk_errors = []
    chunk_scores = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        # 마지막 구간은 남은 데이터를 모두 포함하도록 처리
        end_idx = len(local_distances) if i == num_chunks - 1 else (i + 1) * chunk_size
        
        chunk_data = local_distances[start_idx:end_idx]
        
        if chunk_data:
            c_error = float(np.mean(chunk_data))
            max_score_per_chunk = 50.0 / num_chunks
            c_score = calculate_score(c_error, max_score=max_score_per_chunk)
        else:
            c_error = global_error
            c_score = 0.0
            
        chunk_errors.append(c_error)
        chunk_scores.append(c_score)

    total_chunk_score = sum(chunk_scores)

    # ---------------------------------------------------------
    # 4. 최종 점수 합산 (최대 100점)
    # ---------------------------------------------------------
    final_score = global_score + total_chunk_score

    return {
        "raw_distance": float(D[-1, -1]),
        "normalized_distance": float(global_error), 
        "global_error": float(global_error),
        "global_score": float(global_score),          # 50점 만점 중 획득 점수
        "chunk_errors": chunk_errors,                 # 각 5구역의 오차 리스트
        "chunk_scores": chunk_scores,                 # 각 5구역의 점수 리스트
        "total_chunk_score": float(total_chunk_score),# 50점 만점 중 획득 점수
        "score": float(final_score),                  # 최종 100점 만점 점수
        "dtw_matrix": D,
        "warping_path": wp,
    }


def pitch_similarity_scores(f0_ref, f0_usr) -> dict:
    ref_feat = pitch_features(f0_ref)
    usr_feat = pitch_features(f0_usr)
    
    ref_voiced = f0_ref[f0_ref > 0]
    usr_voiced = f0_usr[f0_usr > 0]
    
    ref_mean = np.mean(ref_voiced) if len(ref_voiced) > 0 else 1e-6
    usr_mean = np.mean(usr_voiced) if len(usr_voiced) > 0 else 1e-6
    ref_std = np.std(ref_voiced) if len(ref_voiced) > 0 else 1e-6
    usr_std = np.std(usr_voiced) if len(usr_voiced) > 0 else 1e-6

    ref_cv = ref_std / max(ref_mean, 1e-6)
    usr_cv = usr_std / max(usr_mean, 1e-6)

    if ref_cv < 1e-3:
        if usr_cv < 1e-3:
            pitch_range_score = 100.0
        else:
            pitch_range_score = max(0.0, 100.0 - (usr_cv * 100.0))
    else:
        cv_error = abs(ref_cv - usr_cv) / ref_cv
        lenient_error = cv_error * 0.5
        pitch_range_score = normalize_score_from_error(lenient_error)

    def fill_nan_and_znorm(x):
        x = np.asarray(x, dtype=float)
        if np.all(np.isnan(x)):
            return np.zeros_like(x)
        
        # 무성음(NaN) 구간을 선형 보간
        valid = ~np.isnan(x)
        idx = np.arange(len(x))
        x_filled = np.interp(idx, idx[valid], x[valid])
        
        # Z-Score 정규화
        std = np.std(x_filled)
        if std < 1e-6:
            return np.zeros_like(x_filled)
        return (x_filled - np.mean(x_filled)) / std

    ref_z = fill_nan_and_znorm(f0_ref)
    usr_z = fill_nan_and_znorm(f0_usr)

    D_pitch, wp_pitch = dtw(X=ref_z, Y=usr_z, metric="euclidean")
    contour_distance = D_pitch[-1, -1] / max(len(wp_pitch), 1)
    
    pitch_contour_score = calculate_similarity_score(
        contour_distance,
        scale=15.0, 
    )

    pitch_score = (0.7 * pitch_contour_score) + (0.3 * pitch_range_score)

    return {
        "pitch_level_score": 0.0,
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

    tolerance = 0.20 
    if error <= tolerance:
        score = 100.0 - (error * 50.0)
    else:
        boundary_score = 100.0 - (tolerance * 50.0) 
        adjusted_error = error - tolerance
        score = max(0.0, boundary_score * np.exp(-5.0 * adjusted_error))

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

    # 강세는 점수가 아닌 P/F 통과 여부만 판단
    is_pass = bool(usr_mean > 0.01 or usr_mean > (ref_mean * 0.3))

    return {
        "ref_mean_rms": float(ref_mean),
        "usr_mean_rms": float(usr_mean),
        "is_pass": is_pass,  
    }

def phoneme_similarity_score(y_ref, sr_ref: int, y_usr, sr_usr: int) -> dict:
    """
    모음(Formant) 흐름과 자음(Plosive) 타격 리듬을 
    각각 50%씩 합산하여 '음소(Phoneme) 정확도'를 산출합니다.
    """
    def z_norm_voiced(x):
        x = np.asarray(x, dtype=float)
        voiced_idx = x > 0.0  # 0.0보다 큰 '진짜 소리' 구간만 찾음
        
        if np.sum(voiced_idx) < 2:
            return np.zeros_like(x)
            
        mean_val = np.mean(x[voiced_idx])
        std_val = np.std(x[voiced_idx])
        
        result = np.zeros_like(x)
        if std_val > 1e-6:
            result[voiced_idx] = (x[voiced_idx] - mean_val) / std_val
        return result

    f1_ref, f2_ref = extract_formant_sequence(y_ref, sr_ref)
    f1_usr, f2_usr = extract_formant_sequence(y_usr, sr_usr)
    
    feat_ref = np.vstack([z_norm_voiced(f1_ref), z_norm_voiced(f2_ref)])
    feat_usr = np.vstack([z_norm_voiced(f1_usr), z_norm_voiced(f2_usr)])

    D_f, wp_f = dtw(X=feat_ref, Y=feat_usr, metric="euclidean")
    dist_f = D_f[-1, -1] / max(len(wp_f), 1)
    
    soft_limit = 2.0  # 완화된 감점이 적용되는 마지노선
    if dist_f <= soft_limit:
        formant_score = 100.0 - (dist_f * 5.0) 
    else:
        formant_score = max(0.0, 90.0 * np.exp(-0.05 * (dist_f - soft_limit)))

    def z_norm_standard(x):
        s = np.std(x)
        return (x - np.mean(x)) / s if s > 1e-6 else np.zeros_like(x)

    env_ref = extract_plosive_sequence(y_ref, sr_ref)
    env_usr = extract_plosive_sequence(y_usr, sr_usr)

    D_p, wp_p = dtw(X=z_norm_standard(env_ref), Y=z_norm_standard(env_usr), metric="euclidean")
    dist_p = D_p[-1, -1] / max(len(wp_p), 1)
    
    plosive_score = max(0.0, 100.0 * np.exp(-0.3 * dist_p))

    phoneme_score = (formant_score * 0.5) + (plosive_score * 0.5)

    return {
        "formant_sub_score": float(formant_score),
        "plosive_sub_score": float(plosive_score),
        "phoneme_score": float(phoneme_score)
    }


def calculate_final_score(
    pronunciation_score: float, # MFCC
    pitch_score: float,         # 피치
    duration_score: float,      # 타이밍
    phoneme_score: float,       # 음소
) -> float:
    # 4개 항목을 25%씩 동일 비율로 산정
    final_score = (
        0.25 * pronunciation_score +
        0.25 * phoneme_score +
        0.25 * pitch_score +
        0.25 * duration_score
    )
    return float(final_score)

import os
import numpy as np
import librosa


def load_audio_flexible(path: str, sr: int = 16000):
    """
    오디오 파일을 읽고 mono / target sr로 통일해서 반환
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    print(f"[로드 시도] {path}")
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)

    if y is None or len(y) == 0:
        raise RuntimeError(f"오디오 로드 실패: {path}")

    print(f"[로드 성공] samples={len(y)}, sr={loaded_sr}")
    return y, loaded_sr


def validate_audio(y, sr: int, name: str = "audio") -> dict:
    """
    오디오 길이 / 평균 RMS 검사
    """
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 0.0

    print(f"[{name}] duration={duration:.3f}s, mean_rms={mean_rms:.6f}")

    warnings = []
    if duration < 0.15:
        warnings.append(f"{name}: 길이가 너무 짧습니다.")
    if mean_rms < 0.005:
        warnings.append(f"{name}: 음량이 너무 작습니다.")

    for w in warnings:
        print(f"[경고] {w}")

    return {
        "duration": float(duration),
        "mean_rms": float(mean_rms),
        "warnings": warnings,
    }


def trim_silence(y, top_db: int = 25):
    """
    앞뒤 무음 제거
    """
    y_trimmed, idx = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed, idx


def normalize_audio(y):
    """
    peak normalize
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def align_by_onset(y, sr: int):
    """
    첫 onset 기준으로 시작점 정렬
    """
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=True,
    )

    if len(onset_frames) == 0:
        return y

    onset_sample = librosa.frames_to_samples(onset_frames[0])
    return y[onset_sample:]


def preprocess_for_pitch(y, sr: int, top_db: int = 25):
    """
    pitch 분석용 전처리:
    - 무음 제거
    """
    y_trimmed, trim_idx = trim_silence(y, top_db=top_db)
    return {
        "y": y_trimmed,
        "trim_idx": trim_idx,
        "sr": sr,
    }


def preprocess_for_pronunciation(y, sr: int, top_db: int = 25):
    """
    pronunciation 분석용 전처리:
    - 무음 제거
    - onset 정렬
    - 볼륨 정규화
    """
    y_trimmed, trim_idx = trim_silence(y, top_db=top_db)
    y_aligned = align_by_onset(y_trimmed, sr)
    y_norm = normalize_audio(y_aligned)

    return {
        "y": y_norm,
        "trim_idx": trim_idx,
        "sr": sr,
    }


def safe_mean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def resample_array(arr, target_len: int = 200):
    """
    길이가 다른 배열을 target_len으로 보간
    """
    arr = np.asarray(arr, dtype=float)

    if len(arr) == 0:
        return np.zeros(target_len)

    x = np.arange(len(arr))
    valid = ~np.isnan(arr)

    if np.sum(valid) < 2:
        fill_value = np.nanmean(arr) if np.sum(valid) > 0 else 0.0
        return np.full(target_len, fill_value, dtype=float)

    arr_interp = np.interp(x, x[valid], arr[valid])
    x_new = np.linspace(0, len(arr_interp) - 1, target_len)
    arr_resampled = np.interp(x_new, x, arr_interp)

    return arr_resampled


def extract_mfcc(y, sr: int, n_mfcc: int = 13):
    """
    MFCC 추출 + 전역 정규화
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfcc.shape[1]
    
    if n_frames < 3:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)
    else:
        delta_width = min(9, n_frames)
        if delta_width % 2 == 0:
            delta_width -= 1
            
        delta = librosa.feature.delta(mfcc, width=delta_width)
        delta2 = librosa.feature.delta(mfcc, order=2, width=delta_width)
    
    mfcc_combined = np.vstack([mfcc, delta, delta2])
    
    mean = np.mean(mfcc_combined, axis=1, keepdims=True)
    std = np.std(mfcc_combined, axis=1, keepdims=True)
    mfcc_norm = (mfcc_combined - mean) / (std + 1e-6)
    
    return mfcc_norm


def extract_pitch(y, sr: int, fmin: str = "C2", fmax: str = "G5"):
    """
    pyin 기반 F0 추출
    무성구간은 NaN
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
    )
    times = librosa.times_like(f0, sr=sr)
    return times, f0


def extract_intensity(y):
    """
    RMS 기반 intensity
    """
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    return times, rms


def check_voiced_ratio(f0, name: str = "audio") -> float:
    total = len(f0)
    voiced = np.sum(~np.isnan(f0))
    ratio = voiced / total if total > 0 else 0.0

    print(f"[{name} voiced ratio] {voiced}/{total} = {ratio:.2%}")
    return ratio


def pitch_features(f0) -> dict:
    valid = np.asarray(f0, dtype=float)
    valid = valid[~np.isnan(valid)]

    if len(valid) == 0:
        return {
            "mean_f0": 0.0,
            "max_f0": 0.0,
            "min_f0": 0.0,
            "range_f0": 0.0,
        }

    mean_f0 = float(np.mean(valid))
    max_f0 = float(np.max(valid))
    min_f0 = float(np.min(valid))
    range_f0 = max_f0 - min_f0

    return {
        "mean_f0": mean_f0,
        "max_f0": max_f0,
        "min_f0": min_f0,
        "range_f0": float(range_f0),
    }


def align_pitch_by_first_voiced(times, f0):
    """
    첫 유성구간 기준 정렬
    """
    times = np.asarray(times, dtype=float)
    f0 = np.asarray(f0, dtype=float)

    voiced_idx = np.where(~np.isnan(f0))[0]
    if len(voiced_idx) == 0:
        return times, f0, None

    first_idx = voiced_idx[0]
    aligned_times = times - times[first_idx]
    return aligned_times, f0, first_idx


def align_contours_by_cross_correlation(f0_ref, f0_usr, target_len: int = 200):
    """
    contour 정렬
    """
    ref = resample_array(f0_ref, target_len=target_len)
    usr = resample_array(f0_usr, target_len=target_len)

    def fill_and_norm(x):
        x = np.asarray(x, dtype=float)

        if np.all(np.isnan(x)):
            return np.zeros_like(x)

        valid = ~np.isnan(x)
        idx = np.arange(len(x))
        x = np.interp(idx, idx[valid], x[valid])

        std = np.std(x)
        if std < 1e-6:
            return np.zeros_like(x)

        return (x - np.mean(x)) / std

    ref_n = fill_and_norm(ref)
    usr_n = fill_and_norm(usr)

    corr = np.correlate(ref_n, usr_n, mode="full")
    shift = np.argmax(corr) - (len(ref_n) - 1)

    if shift > 0:
        usr_shifted = np.pad(usr_n, (shift, 0), mode="constant")[:len(usr_n)]
    elif shift < 0:
        usr_shifted = np.pad(usr_n, (0, -shift), mode="constant")[-shift: len(usr_n) - shift]
    else:
        usr_shifted = usr_n.copy()

    return ref_n, usr_n, usr_shifted, int(shift)


def extract_formant_sequence(y, sr: int, hop_length: int = 512):
    """
    [V2 업데이트] 오디오를 프레임 단위로 쪼개어 시간에 따른 F1, F2 모음 변화의 '흐름'을 추출합니다.
    """
    y_preemp = librosa.effects.preemphasis(y)
    frames = librosa.util.frame(y_preemp, frame_length=1024, hop_length=hop_length)
    
    f1_list, f2_list = [], []
    order = 16
    
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        frame = frame * np.hanning(len(frame))
        
        # 무음 구간은 0으로 처리
        if np.sum(np.abs(frame)) < 1e-4:
            f1_list.append(0.0)
            f2_list.append(0.0)
            continue
            
        a = librosa.lpc(frame, order=order)
        roots = np.roots(a)
        roots = [r for r in roots if np.imag(r) > 0]
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        freqs = sorted([f for f in freqs if f > 90])
        
        f1_list.append(freqs[0] if len(freqs) > 0 else 0.0)
        f2_list.append(freqs[1] if len(freqs) > 1 else 0.0)
        
    return np.array(f1_list), np.array(f2_list)


def extract_plosive_sequence(y, sr: int):
    """
    [V2 업데이트] 파열음이 '어느 타이밍'에 터졌는지 시계열 배열로 반환합니다.
    """
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 3000Hz 이상 고주파 대역 필터링
    high_freq_idx = np.where(freqs > 3000)[0]
    if len(high_freq_idx) == 0:
        return np.zeros(S.shape[1])
        
    S_high = S[high_freq_idx, :]
    
    # 타격감(Onset Strength) 추출
    onset_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S_high, ref=np.max), 
        sr=sr
    )
    
    return onset_env
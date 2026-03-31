"""
오디오 처리 파이프라인: Demucs(배경음 제거) → Whisper(STT)

추후 LLM 후보정 → KoNLPy 형태소 분석이 이 파일에 추가될 예정.
"""
import os
import uuid
import subprocess
import tempfile
import whisper

# 서버 기동 후 첫 요청 시 로드, 이후 재사용
_model_cache: dict[str, whisper.Whisper] = {}


def _get_whisper_model(model_size: str) -> whisper.Whisper:
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]


def _run_demucs(input_path: str, out_dir: str) -> str | None:
    cmd = [
        "python", "-m", "demucs",
        "--two-stems=vocals",
        "--out", out_dir,
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(out_dir, "htdemucs", base_name, "vocals.wav")
    return vocals_path if os.path.exists(vocals_path) else None


def _run_whisper(audio_path: str, model_size: str) -> str:
    model = _get_whisper_model(model_size)
    result = model.transcribe(audio_path, language="ko")
    return result["text"]


def run_pipeline(audio_bytes: bytes, ffmpeg_bin: str, whisper_model_size: str) -> str:
    """
    Demucs → Whisper 파이프라인 실행.
    Demucs 실패 시 원본 오디오로 Whisper를 실행(fallback).

    Returns:
        Whisper 인식 텍스트 (LLM 후보정 전 raw 텍스트)
    """
    if ffmpeg_bin:
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}.wav")
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        vocals_path = _run_demucs(input_path, tmpdir)
        if vocals_path is None:
            vocals_path = input_path  # Demucs 실패 시 원본으로 fallback

        return _run_whisper(vocals_path, whisper_model_size)

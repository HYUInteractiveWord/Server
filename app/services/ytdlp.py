import asyncio
import functools
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def _sec_to_ts(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s_int = divmod(rem, 60)
    frac = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s_int:02d}.{frac:03d}"


def _extract_sync(url: str, start_sec: float, end_sec: float) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        out_tmpl = os.path.join(tmp, "audio.%(ext)s")
        section  = f"*{_sec_to_ts(start_sec)}-{_sec_to_ts(end_sec)}"

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--download-sections", section,
            "-x",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            "-o", out_tmpl,
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"yt-dlp 오류: {err}")

        files = list(Path(tmp).glob("audio.*"))
        if not files:
            raise RuntimeError("오디오 추출 실패: 출력 파일이 생성되지 않았습니다.")

        return files[0].read_bytes()


async def extract_youtube_audio(url: str, end_sec: float, duration_sec: float = 10.0) -> bytes:
    """YouTube URL의 (end_sec - duration_sec) ~ end_sec 구간 오디오를 추출합니다."""
    start_sec = max(0.0, end_sec - duration_sec)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(_extract_sync, url, start_sec, end_sec)
    )

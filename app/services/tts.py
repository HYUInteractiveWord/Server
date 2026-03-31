import hashlib
from pathlib import Path

TTS_DIR = Path("static/tts")


def generate_tts(korean_word: str) -> str | None:
    """
    gTTS 또는 Edge-TTS로 단어 발음 오디오 생성.
    파일이 이미 존재하면 재생성하지 않음.
    """
    TTS_DIR.mkdir(parents=True, exist_ok=True)

    filename = hashlib.md5(korean_word.encode()).hexdigest() + ".mp3"
    filepath = TTS_DIR / filename

    if filepath.exists():
        return str(filepath)

    try:
        from gtts import gTTS
        tts = gTTS(text=korean_word, lang="ko")
        tts.save(str(filepath))
        return str(filepath)
    except Exception:
        return None

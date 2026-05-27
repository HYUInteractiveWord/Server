from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import edge_tts
import urllib.parse

router = APIRouter(prefix="/tts", tags=["tts"])

@router.get("/stream")
async def stream_tts(
    text: str = Query(..., description="읽어줄 텍스트"),
    lang: str = Query("ru", description="언어 코드 (ru, en, ko 등)")
):
    """
    디스크에 파일을 저장하지 않고 즉시 오디오 데이터를 스트리밍 반환합니다.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="텍스트가 비어 있습니다.")

    # URL 인코딩된 텍스트 복구
    decoded_text = urllib.parse.unquote(text)

    # 언어별 지원 목소리 매핑
    voice_map = {
        "ko": "ko-KR-SunHiNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "en": "en-US-AriaNeural"
    }
    voice = voice_map.get(lang, "ru-RU-SvetlanaNeural") # 기본값: 러시아어

    async def audio_generator():
        try:
            communicate = edge_tts.Communicate(decoded_text, voice)
            # stream() 메서드로 오디오 청크를 실시간으로 받아옴
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            print(f"[TTS 스트리밍 에러] {e}")

    # StreamingResponse를 사용해 생성되는 즉시 클라이언트에게 전송 (메모리, 용량 절약)
    return StreamingResponse(audio_generator(), media_type="audio/mpeg")
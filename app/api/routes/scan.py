from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.models.scan_record import ScanRecord
from app.schemas.scan import AudioScanRequest, AudioScanResponse
from app.services.pipeline import run_pipeline
from app.core.config import settings

router = APIRouter(prefix="/scan", tags=["scan"])


@router.post("/process", response_model=AudioScanResponse)
def process_scan_result(
    body: AudioScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    임성빈(배경음 제거) + 서정빈(STT) 파이프라인의 결과를 수신하여 처리.
    추출된 단어가 단어장에 있으면 scan_count 증가, 없으면 새 후보로 반환.
    """
    matched = []
    new_candidates = []

    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    for word in body.extracted_words:
        if word in word_map:
            card = word_map[word]
            card.scan_count += 1
            db.add(ScanRecord(
                user_id=current_user.id,
                word_card_id=card.id,
                scan_source=body.scan_source,
            ))
            matched.append({"word_card_id": card.id, "korean_word": word, "scan_count": card.scan_count})
        else:
            new_candidates.append(word)

    db.commit()
    return AudioScanResponse(matched_words=matched, new_word_candidates=new_candidates)


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    scan_source: str = Form("mic"),  # "mic" | "media"
    current_user: User = Depends(get_current_user),
):
    """
    오디오 파일을 받아 Demucs → Whisper 파이프라인 실행 후 인식 텍스트 반환.
    이후 LLM 후보정 → KoNLPy 형태소 분석은 추후 추가 예정.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. (wav/mp3/m4a/ogg)")

    audio_bytes = await file.read()
    whisper_text = run_pipeline(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )
    return {"whisper_text": whisper_text, "scan_source": scan_source}

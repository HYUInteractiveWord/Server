from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.models.scan_record import ScanRecord
from app.schemas.scan import AudioScanRequest, AudioScanResponse

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
    current_user: User = Depends(get_current_user),
):
    """
    오디오 파일을 받아 임성빈/서정빈 AI 파이프라인으로 전달하는 엔드포인트 (스텁).
    실제 AI 모듈 연동 시 구현 완성 예정.
    """
    # TODO: 파일을 저장하고 Demucs + Whisper 파이프라인 호출
    return {"status": "received", "filename": file.filename, "message": "AI pipeline integration pending"}

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.models.pronunciation_record import PronunciationRecord
from app.schemas.pronunciation import PronunciationSubmit, PronunciationResponse
from app.services.gamification import calculate_xp_gain, update_word_level

router = APIRouter(prefix="/pronunciation", tags=["pronunciation"])

XP_PER_PRACTICE = 10


@router.post("/submit", response_model=PronunciationResponse)
def submit_pronunciation(
    body: PronunciationSubmit,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    김성준 모듈에서 DTW 기반 발음 분석 결과를 수신하여 저장.
    최고 점수 갱신 시 단어 레벨 업 처리.
    """
    card = db.query(WordCard).filter(
        WordCard.id == body.word_card_id,
        WordCard.user_id == current_user.id,
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")

    record = PronunciationRecord(
        user_id=current_user.id,
        word_card_id=card.id,
        score=body.score,
        user_pitch_data=body.user_pitch_data,
        reference_pitch_data=body.reference_pitch_data,
        dtw_distance=body.dtw_distance,
    )
    db.add(record)

    is_new_best = body.score > card.best_score
    if is_new_best:
        card.best_score = body.score
        new_level = update_word_level(card.level, body.score)
        card.level = new_level

    xp_gained = calculate_xp_gain(body.score, is_new_best)
    current_user.xp += xp_gained

    db.commit()
    db.refresh(record)
    db.refresh(card)

    return PronunciationResponse(
        record_id=record.id,
        score=body.score,
        is_new_best=is_new_best,
        xp_gained=xp_gained,
        word_card_level=card.level,
    )


@router.get("/{word_card_id}/history")
def get_pronunciation_history(
    word_card_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = db.query(PronunciationRecord).filter(
        PronunciationRecord.word_card_id == word_card_id,
        PronunciationRecord.user_id == current_user.id,
    ).order_by(PronunciationRecord.recorded_at.desc()).all()
    return records

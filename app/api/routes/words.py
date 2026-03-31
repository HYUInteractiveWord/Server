from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.schemas.word_card import WordCardCreate, WordCardResponse
from app.services.dictionary import fetch_word_info
from app.services.tts import generate_tts

router = APIRouter(prefix="/words", tags=["words"])


@router.get("/", response_model=list[WordCardResponse])
def get_my_words(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(WordCard).filter(WordCard.user_id == current_user.id).all()


@router.post("/", response_model=WordCardResponse, status_code=status.HTTP_201_CREATED)
def create_word_card(
    body: WordCardCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 단어 슬롯 제한 체크
    word_count = db.query(WordCard).filter(WordCard.user_id == current_user.id).count()
    if word_count >= current_user.max_word_slots:
        raise HTTPException(status_code=400, detail="Word slot limit reached. Complete missions to unlock more.")

    # 중복 체크
    existing = db.query(WordCard).filter(
        WordCard.user_id == current_user.id,
        WordCard.korean_word == body.korean_word,
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Word already in your collection")

    # 국립국어원 사전 API 조회
    word_info = fetch_word_info(body.korean_word)

    # TTS 생성
    tts_path = generate_tts(body.korean_word)

    card = WordCard(
        user_id=current_user.id,
        korean_word=body.korean_word,
        source=body.source,
        pos=word_info.get("pos"),
        definition=word_info.get("definition"),
        example_sentences=word_info.get("examples", []),
        tts_audio_path=tts_path,
    )
    db.add(card)
    db.commit()
    db.refresh(card)
    return card


@router.get("/{word_id}", response_model=WordCardResponse)
def get_word_card(
    word_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(WordCard).filter(
        WordCard.id == word_id, WordCard.user_id == current_user.id
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")
    return card


@router.delete("/{word_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_word_card(
    word_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = db.query(WordCard).filter(
        WordCard.id == word_id, WordCard.user_id == current_user.id
    ).first()
    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")
    db.delete(card)
    db.commit()

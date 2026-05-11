from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.schemas.word_card import WordCardCreate, WordCardResponse
from app.services.dictionary import fetch_word_info
from app.services.tts import generate_tts
from app.services.pipeline import KoreanLearningPipeline
from app.core.config import settings


router = APIRouter(prefix="/words", tags=["words"])

nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY,
    dict_api_key=settings.DICT_API_KEY,
    model_name=settings.LLM_MODEL_NAME,
)


def _norm_path(path: str | None) -> str | None:
    if not path:
        return None
    return path.replace("\\", "/")


def _merge_examples_with_tts(examples: list | None, example_tts_paths: list | None) -> list:
    merged = []
    examples = examples or []
    example_tts_paths = example_tts_paths or []

    for i, ex in enumerate(examples):
        if isinstance(ex, dict):
            item = dict(ex)
        else:
            item = {
                "type": "fallback",
                "korean": str(ex),
                "english": "",
            }

        if i < len(example_tts_paths):
            item["tts_audio_path"] = _norm_path(example_tts_paths[i])

        merged.append(item)

    return merged


@router.get("/", response_model=list[WordCardResponse])
def get_my_words(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(WordCard).filter(WordCard.user_id == current_user.id).all()


@router.post("/", response_model=WordCardResponse, status_code=status.HTTP_201_CREATED)
async def create_word_card(
    body: WordCardCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    word = body.korean_word.strip()

    if not word:
        raise HTTPException(status_code=400, detail="Word is empty")

    # 단어 슬롯 제한 체크
    word_count = db.query(WordCard).filter(WordCard.user_id == current_user.id).count()
    if word_count >= current_user.max_word_slots:
        raise HTTPException(
            status_code=400,
            detail="Word slot limit reached. Complete missions to unlock more.",
        )

    # 중복 체크
    existing = db.query(WordCard).filter(
        WordCard.user_id == current_user.id,
        WordCard.korean_word == word,
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Word already in your collection")

    # 프론트에서 선택한 품사/뜻이 있으면 우선 사용하고, 없으면 사전 fallback 사용
    fallback_info = fetch_word_info(word)
    selected_info = {
        "pos": body.pos or fallback_info.get("pos") or "명사",
        "definition": body.definition or fallback_info.get("definition") or "",
    }

    generated = None

    try:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words={word: selected_info},
            output_dir=user_output_dir,
        )
        if generated_cards:
            generated = generated_cards[0]
    except Exception as e:
        print(f"[Word Create] phase2_generate failed for '{word}': {e}", flush=True)

    if generated:
        audio = generated.get("audio", {}) or {}
        example_sentences = _merge_examples_with_tts(
            generated.get("examples", []),
            audio.get("examples_tts", []),
        )

        card = WordCard(
            user_id=current_user.id,
            korean_word=word,
            source=body.source,
            pos=generated.get("pos_type") or selected_info["pos"],
            definition=generated.get("definition_korean") or selected_info["definition"],
            example_sentences=example_sentences,
            tts_audio_path=_norm_path(audio.get("word_tts")),
        )
    else:
        # LLM/Gemma 서버 문제 발생 시 최소 fallback 저장
        tts_path = generate_tts(word)
        card = WordCard(
            user_id=current_user.id,
            korean_word=word,
            source=body.source,
            pos=selected_info["pos"],
            definition=selected_info["definition"],
            example_sentences=fallback_info.get("examples", []),
            tts_audio_path=_norm_path(tts_path),
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
        WordCard.id == word_id,
        WordCard.user_id == current_user.id,
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
        WordCard.id == word_id,
        WordCard.user_id == current_user.id,
    ).first()

    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")

    db.delete(card)
    db.commit()

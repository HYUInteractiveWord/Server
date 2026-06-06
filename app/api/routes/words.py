from datetime import datetime, timezone, timedelta
import os
import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db import get_db
from app.models.mission import Mission
from app.models.user import User
from app.models.word_card import WordCard
from app.schemas.word_card import WordCardCreate, WordCardResponse
from app.services.dictionary import fetch_word_info
from app.services.gamification import get_rank_for_xp, RANK_WORD_SLOTS
from app.services.pipeline import KoreanLearningPipeline
from app.services.tts import generate_tts


router = APIRouter(prefix="/words", tags=["words"])

nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY,
    dict_api_key=settings.DICT_API_KEY,
    model_name=settings.LLM_MODEL_NAME,
)

QUIZ_MISSION_TYPE = "daily_word_quiz"
COLLECT_NOUN_MISSION_TYPE = "daily_collect_noun"

DAILY_MISSION_FALLBACKS = {
    QUIZ_MISSION_TYPE: {"target": 1, "xp_reward": 150},
    COLLECT_NOUN_MISSION_TYPE: {"target": 1, "xp_reward": 100},
}


class WordQuizItemResult(BaseModel):
    word_id: int
    is_correct: bool


class WordQuizResultRequest(BaseModel):
    quiz_type: str = "meaning"
    results: list[WordQuizItemResult]


class BulkGenerateRequest(BaseModel):
    words: list[str]
    output_dir: str = "assets/demo_data"
    target_language: str = "ru"


def _effect_level_from_point(point: int) -> int:
    point = max(0, min(100, point))

    if point >= 100:
        return 4
    if point >= 76:
        return 3
    if point >= 51:
        return 2
    if point >= 26:
        return 1

    return 0


def _is_noun_pos(pos: str | None) -> bool:
    if not pos:
        return False

    normalized = pos.strip().lower()
    return "명사" in normalized or normalized == "noun"


def _refresh_user_rank_and_slots(user: User) -> None:
    new_rank = get_rank_for_xp(user.xp)

    if new_rank != user.rank:
        user.rank = new_rank
        user.max_word_slots = RANK_WORD_SLOTS[new_rank]


def _increase_daily_mission_progress(
    db: Session,
    user_id: int,
    mission_type: str,
    amount: int = 1,
) -> Mission | None:
    today = datetime.now(timezone(timedelta(hours=9))).date()
    fallback = DAILY_MISSION_FALLBACKS.get(mission_type)

    mission = (
        db.query(Mission)
        .filter(
            Mission.user_id == user_id,
            Mission.mission_type == mission_type,
        )
        .first()
    )

    if mission is None:
        if fallback is None:
            return None

        mission = Mission(
            user_id=user_id,
            mission_type=mission_type,
            target=fallback["target"],
            xp_reward=fallback["xp_reward"],
            progress=0,
            is_completed=False,
            completed_at=None,
            last_reset_date=today,
        )
        db.add(mission)
    else:
        if mission.last_reset_date != today:
            mission.progress = 0
            mission.is_completed = False
            mission.completed_at = None
            mission.last_reset_date = today

        if fallback is not None:
            mission.target = fallback["target"]
            mission.xp_reward = fallback["xp_reward"]

    if not mission.is_completed:
        mission.progress = min(
            mission.target,
            (mission.progress or 0) + max(0, amount),
        )

    return mission


def _complete_mission_if_ready(mission: Mission | None, user: User) -> int:
    if mission is None:
        return 0

    if mission.is_completed:
        return 0

    if (mission.progress or 0) < mission.target:
        return 0

    mission.is_completed = True

    if hasattr(mission, "completed_at"):
        mission.completed_at = datetime.utcnow()

    reward = mission.xp_reward or 0

    if reward > 0:
        user.xp += reward
        _refresh_user_rank_and_slots(user)

    return reward


def _norm_path(path: str | None) -> str | None:
    if not path:
        return None

    return path.replace("\\", "/")


def _normalize_target_language(language: str | None) -> str:
    lang = (language or "ko").strip().lower().split("-")[0]

    if lang in {"ru", "en", "ko"}:
        return lang

    return "ko"


def _pick_translated_definition(generated: dict) -> str | None:
    candidates = [
        generated.get("definition_translated"),
        generated.get("translated_definition"),
        generated.get("translated_def_text"),
        generated.get("definition_target"),
        generated.get("definition_ru"),
        generated.get("def_translated"),
        generated.get("def_translation"),
        generated.get("meaning_translated"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _pick_def_trans_audio_path(generated: dict, audio: dict) -> str | None:
    candidates = [
        generated.get("def_trans_audio_path"),
        generated.get("definition_trans_audio_path"),
        generated.get("translated_definition_audio_path"),
        generated.get("def_trans_tts"),
        audio.get("def_trans_audio_path"),
        audio.get("definition_trans_audio_path"),
        audio.get("translated_definition_audio_path"),
        audio.get("def_trans_tts"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return _norm_path(value.strip())

    return None


def _merge_examples_with_tts(examples, example_tts_paths):
    merged = []
    tts_paths = example_tts_paths or []

    for i, ex in enumerate(examples or []):
        if isinstance(ex, dict):
            item = dict(ex)
            item.setdefault("type", "llm_generated")
        else:
            item = {
                "type": "fallback",
                "korean": str(ex),
                "english": "",
            }

        if i < len(tts_paths) and not item.get("tts_audio_path"):
            item["tts_audio_path"] = _norm_path(tts_paths[i])

        if item.get("audio_path"):
            item["audio_path"] = _norm_path(item.get("audio_path"))

        if item.get("tts_audio_path"):
            item["tts_audio_path"] = _norm_path(item.get("tts_audio_path"))

        if item.get("trans_audio_path"):
            item["trans_audio_path"] = _norm_path(item.get("trans_audio_path"))

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

    word_count = (
        db.query(WordCard)
        .filter(WordCard.user_id == current_user.id)
        .count()
    )

    if word_count >= current_user.max_word_slots:
        raise HTTPException(
            status_code=400,
            detail="Word slot limit reached. Complete missions to unlock more.",
        )

    existing = (
        db.query(WordCard)
        .filter(
            WordCard.user_id == current_user.id,
            WordCard.korean_word == word,
        )
        .first()
    )

    if existing:
        existing.word_point = (existing.word_point or 0) + 5
        existing.effect_level = _effect_level_from_point(existing.word_point or 0)
        current_user.xp += 10
        _refresh_user_rank_and_slots(current_user)

        db.commit()
        db.refresh(existing)

        return existing

    fallback_info = fetch_word_info(word)

    selected_info = {
        "pos": body.pos or fallback_info.get("pos") or "명사",
        "definition": body.definition or fallback_info.get("definition") or "",
    }

    target_language = _normalize_target_language(
        getattr(current_user, "preferred_language", None)
    )

    generated = None

    try:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words={word: selected_info},
            output_dir=user_output_dir,
            target_language=target_language,
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

        word_tts_path = (
            audio.get("word_tts")
            or generated.get("tts_audio_path")
            or generated.get("audio_path")
        )

        card = WordCard(
            user_id=current_user.id,
            korean_word=word,
            source=body.source,
            pos=generated.get("pos_type") or generated.get("pos") or selected_info["pos"],
            definition=generated.get("definition_korean") or selected_info["definition"],
            definition_translated=_pick_translated_definition(generated),
            example_sentences=example_sentences,
            tts_audio_path=_norm_path(word_tts_path),
            def_trans_audio_path=_pick_def_trans_audio_path(generated, audio),
        )
    else:
        tts_path = generate_tts(word)

        card = WordCard(
            user_id=current_user.id,
            korean_word=word,
            source=body.source,
            pos=selected_info["pos"],
            definition=selected_info["definition"],
            definition_translated=None,
            example_sentences=fallback_info.get("examples", []),
            tts_audio_path=_norm_path(tts_path),
            def_trans_audio_path=None,
        )

    db.add(card)

    if _is_noun_pos(card.pos):
        mission = _increase_daily_mission_progress(
            db=db,
            user_id=current_user.id,
            mission_type=COLLECT_NOUN_MISSION_TYPE,
            amount=1,
        )
        _complete_mission_if_ready(mission, current_user)

    db.commit()
    db.refresh(card)

    return card


@router.post("/quiz-result")
def submit_word_quiz_result(
    body: WordQuizResultRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    단어 테스트 결과를 반영한다.

    - 정답 1개당 user.xp +10
    - 정답 단어의 word_point +10
    - 테스트 1회 제출 시 daily_word_quiz 미션 progress +1
    - 미션 완료 조건을 만족하면 미션 보상 XP 지급
    """
    if not body.results:
        raise HTTPException(status_code=400, detail="Quiz result is empty")

    seen_word_ids: set[int] = set()
    checked_words = []
    correct_count = 0
    valid_count = 0

    for result in body.results:
        if result.word_id in seen_word_ids:
            continue

        seen_word_ids.add(result.word_id)

        card = (
            db.query(WordCard)
            .filter(
                WordCard.id == result.word_id,
                WordCard.user_id == current_user.id,
            )
            .first()
        )

        if card is None:
            continue

        valid_count += 1

        if result.is_correct:
            correct_count += 1
            card.word_point = (card.word_point or 0) + 10
            card.effect_level = _effect_level_from_point(card.word_point)

        checked_words.append(
            {
                "id": card.id,
                "korean_word": card.korean_word,
                "is_correct": result.is_correct,
                "word_point": card.word_point,
                "effect_level": card.effect_level,
            }
        )

    if valid_count == 0:
        raise HTTPException(status_code=400, detail="No valid word cards in result")

    perfect_bonus = 10 if valid_count >= 3 and correct_count == valid_count else 0
    quiz_score = correct_count * 10 + perfect_bonus
    quiz_xp_gained = correct_count * 10
    current_user.xp += quiz_xp_gained
    _refresh_user_rank_and_slots(current_user)

    mission = _increase_daily_mission_progress(
        db=db,
        user_id=current_user.id,
        mission_type=QUIZ_MISSION_TYPE,
        amount=1,
    )

    mission_xp_gained = _complete_mission_if_ready(mission, current_user)

    db.commit()

    return {
        "quiz_type": body.quiz_type,
        "total": valid_count,
        "correct": correct_count,
        "score": quiz_score,
        "perfect_bonus": perfect_bonus,
        "quiz_xp_gained": quiz_xp_gained,
        "mission_xp_gained": mission_xp_gained,
        "total_xp_gained": quiz_xp_gained + mission_xp_gained,
        "user": {
            "xp": current_user.xp,
            "rank": current_user.rank,
            "max_word_slots": current_user.max_word_slots,
        },
        "checked_words": checked_words,
        "mission": {
            "mission_type": mission.mission_type,
            "progress": mission.progress,
            "target": mission.target,
            "is_completed": mission.is_completed,
            "xp_reward": mission.xp_reward,
        }
        if mission
        else None,
    }


@router.get("/{word_id}", response_model=WordCardResponse)
def get_word_card(
    word_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(WordCard)
        .filter(
            WordCard.id == word_id,
            WordCard.user_id == current_user.id,
        )
        .first()
    )

    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")

    return card


@router.delete("/{word_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_word_card(
    word_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    card = (
        db.query(WordCard)
        .filter(
            WordCard.id == word_id,
            WordCard.user_id == current_user.id,
        )
        .first()
    )

    if not card:
        raise HTTPException(status_code=404, detail="Word card not found")

    db.delete(card)
    db.commit()


@router.post("/admin/generate_bulk")
async def generate_bulk_data(body: BulkGenerateRequest):
    """
    [HTML 테스트 전용] 로그인 없이 다수의 단어를 한 번에 생성하여 지정된 폴더에 저장합니다.
    """
    os.makedirs(body.output_dir, exist_ok=True)

    results = []
    for word_str in body.words:
        word = word_str.strip()
        if not word:
            continue

        fallback_info = fetch_word_info(word)
        selected_info = {
            "pos": fallback_info.get("pos") or "명사",
            "definition": fallback_info.get("definition") or "",
        }

        try:
            cards = await nlp_pipeline.phase2_generate(
                selected_words={word: selected_info},
                output_dir=body.output_dir,
                target_language=body.target_language,
            )
            if cards:
                results.extend(cards)
        except Exception as e:
            print(f"[Bulk Gen Error] '{word}': {e}", flush=True)

        await asyncio.sleep(0.5)

    return {
        "success_count": len(results),
        "output_dir": body.output_dir,
        "cards": results,
    }
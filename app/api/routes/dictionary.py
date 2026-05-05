from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.services.pipeline import KoreanLearningPipeline
from app.core.config import settings
from app.services.dictionary import fetch_word_info

router = APIRouter(prefix="/dictionary", tags=["dictionary"])

nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY, 
    dict_api_key=settings.DICT_API_KEY, 
    model_name=settings.LLM_MODEL_NAME
)

class DictProcessRequest(BaseModel):
    extracted_words: Dict[str, Any]


@router.get("/search")
async def search_dictionary(
    word: str = Query(..., min_length=1, description="검색할 단어"),
    source_lang: str = Query("한국어", description="입력 언어 지정"),
):
    """
    통합 사전 검색 API.

    - 프론트 단어 검색용: word, pos, definition 반환
    - 기존 파이프라인용: search_query, candidates 유지
    """
    word = word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")

    # 1. 한국어 단어 정보 조회
    word_info = fetch_word_info(word)

    # 2. 기존 후보 검색 기능도 유지
    candidates = []
    try:
        candidates = await nlp_pipeline.search_dictionary_candidates(
            query=word,
            source_lang=source_lang,
        )
    except Exception as e:
        print(f"[Dict Search] candidate search failed: {e}")

    return {
        # 프론트용 필드
        "word": word,
        "pos": word_info.get("pos"),
        "definition": word_info.get("definition"),

        # 기존 기능 유지용 필드
        "search_query": word,
        "candidates": candidates,
    }


@router.post("/process")
async def process_dictionary_words(
    req: DictProcessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    [Phase 2] 사전 검색을 통해 선택된 단어를 단어장 카드로 생성
    🚨 학습 스캔(scan_count) 포인트는 올리지 않음.
    """
    # 유저의 기존 단어장 조회
    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    already_exists = []
    new_candidates_for_generation = {}

    for korean_word, info in req.extracted_words.items():
        if korean_word in word_map:
            # 💡 [핵심 차이점] 이미 단어장에 있는 경우:
            # db.add(ScanRecord(...)) 를 생략하고 scan_count도 건드리지 않습니다.
            already_exists.append({
                "word_card_id": word_map[korean_word].id, 
                "korean_word": korean_word, 
                "status": "already_in_wordbook"
            })
        else:
            # 단어장에 없는 새로운 단어만 생성 큐로 이동
            new_candidates_for_generation[korean_word] = info

    generated_cards = []
    
    # 새로운 단어가 있다면 LLM 예문 및 TTS 파일 생성
    if new_candidates_for_generation:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir
        )
        
        # TODO: 생성된 generated_cards 데이터를 WordCard DB에 Insert (필요시 구현)

    return {
        "already_exists": already_exists, 
        "new_word_cards": generated_cards
    }
from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
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
    target_language: str = "ru"

class PreviewRequest(BaseModel):
    word: str
    definition: str
    pos: str
    target_language: str = "ru" 
@router.get("/search")
async def search_dictionary(
    word: str = Query(..., min_length=1, description="검색할 단어"),
    source_lang: str = Query("한국어", description="입력 언어 지정"),
):
    """
    통합 사전 검색 API.
    """
    word = word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")

    word_info = fetch_word_info(word)

    candidates = []
    try:
        candidates = await nlp_pipeline.search_dictionary_candidates(
            query=word,
            source_lang=source_lang,
        )
    except Exception as e:
        print(f"[Dict Search] candidate search failed: {e}")

    return {
        "word": word,
        "pos": word_info.get("pos"),
        "definition": word_info.get("definition"),
        "search_query": word,
        "candidates": candidates,
    }


@router.post("/preview")
async def get_word_preview(req: PreviewRequest):
    """
    사전 검색 후 단어장 추가 전, 뜻과 발음(TTS)만 임시로 생성하여 반환 (다국어 호환)
    """
    output_dir = "static/tts/temp"
    # ★ 연결 포인트: target_language 파라미터 유실 누락 결합 완료
    result = await nlp_pipeline.generate_word_preview(
        word=req.word, 
        definition=req.definition, 
        pos=req.pos, 
        output_dir=output_dir,
        target_language=req.target_language 
    )
    return result


@router.post("/verify")
async def verify_pronunciation(
    file: UploadFile = File(...),
    target_word: str = Form(...)
):
    """
    사용자 녹음 파일 수신 후 타겟 단어와 발음 일치 여부 검증
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    audio_bytes = await file.read()
    
    result = await nlp_pipeline.verify_spoken_word(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
        target_word=target_word
    )
    
    return result


@router.post("/process")
async def process_dictionary_words(
    req: DictProcessRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    [Phase 2] 사전 검색을 통해 선택된 단어를 단어장 카드로 생성
    """
    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    already_exists = []
    new_candidates_for_generation = {}

    for korean_word, info in req.extracted_words.items():
        if korean_word in word_map:
            already_exists.append({
                "word_card_id": word_map[korean_word].id, 
                "korean_word": korean_word, 
                "status": "already_in_wordbook"
            })
        else:
            new_candidates_for_generation[korean_word] = info

    generated_cards = []
    
    if new_candidates_for_generation:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir,
            target_language=req.target_language
        )
        
        for card_data in generated_cards:
            new_word_card = WordCard(
                user_id=current_user.id,
                korean_word=card_data["word"],
                english_meaning=card_data["definition_translated"], # 공통 번역 필드로 대응
                korean_definition=card_data["definition_korean"],
                part_of_speech=card_data["pos_type"],
                semantic_category=card_data["semantic_category"],
                pronunciation=card_data["pronunciation"],
                audio_path_word=card_data["audio"]["word_tts"],
                scan_count=1
            )
            db.add(new_word_card)
        db.commit()

    return {
        "already_exists": already_exists, 
        "new_word_cards": generated_cards
    }
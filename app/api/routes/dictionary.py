from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional

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

def _normalize_target_language(language: str | None) -> str:
    lang = (language or "ko").strip().lower().split("-")[0]
    if lang in {"ko", "ru", "en"}:
        return lang
    return "ko"


def _source_lang_for_target(language: str) -> str:
    if language == "ru":
        return "러시아어"
    if language == "en":
        return "영어"
    return "한국어"


# ==========================================
# Pydantic Schemas
# ==========================================
class DictProcessRequest(BaseModel):
    extracted_words: Dict[str, Any]
    target_language: str = "ru"

class PreviewRequest(BaseModel):
    word: str
    definition: str
    pos: str
    target_language: str = "ru"

# ==========================================
# Routes
# ==========================================
@router.get("/search")
async def search_dictionary(
    word: str = Query(..., min_length=1, description="검색할 단어"),
    source_lang: Optional[str] = Query(None, description="입력 언어 지정"),
    current_user: User = Depends(get_current_user),
):
    """
    통합 사전 검색 API.

    - 프론트 단어 검색용: word, pos, definition 반환
    - 기존 파이프라인용: search_query, candidates 유지
    """
    word = word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="검색어를 입력해주세요.")

    target_language = _normalize_target_language(current_user.preferred_language)
    effective_source_lang = source_lang or _source_lang_for_target(target_language)

    # 1. 한국어 단어 정보 조회
    word_info = fetch_word_info(word)

    # 2. 기존 후보 검색 기능도 유지
    candidates = []
    try:
        candidates = await nlp_pipeline.search_dictionary_candidates(
            query=word,
            source_lang=effective_source_lang,
        )
    except Exception as e:
        print(f"[Dict Search] candidate search failed: {e}")


    candidate_map = {}

    for candidate in candidates or []:
        if isinstance(candidate, dict):
            candidate_word = (
                candidate.get("word")
                or candidate.get("korean_word")
                or candidate.get("korean")
                or candidate.get("lemma")
                or ""
            )
            candidate_word = str(candidate_word).strip()

            item = {
                "pos": candidate.get("pos") or candidate.get("pos_type"),
                "definition": candidate.get("definition") or "",
            }
        else:
            candidate_word = str(candidate).strip()
            fetched = fetch_word_info(candidate_word) if candidate_word else {}
            item = {
                "pos": fetched.get("pos"),
                "definition": fetched.get("definition") or "",
            }

        if not candidate_word:
            continue

        if target_language != "ko" and item.get("definition"):
            try:
                preview = await nlp_pipeline.generate_word_preview(
                    word=candidate_word,
                    definition=item.get("definition") or "",
                    pos=item.get("pos") or "",
                    output_dir="static/tts/temp",
                    target_language=target_language,
                )
                item["definition_translated"] = preview.get("definition_translated")
                item["pronunciation"] = preview.get("pronunciation")
                item["def_trans_audio_path"] = preview.get("def_trans_audio_path")
            except Exception as e:
                print(f"[Dict Search] translate preview failed for '{candidate_word}': {e}", flush=True)

        candidate_map[candidate_word] = item

    candidates = candidate_map

    return {
        # 프론트용 필드
        "word": word,
        "pos": word_info.get("pos"),
        "definition": word_info.get("definition"),

        # 기존 기능 유지용 필드
        "search_query": word,
        "candidates": candidates,
    }


@router.post("/preview")
async def get_word_preview(req: PreviewRequest):
    output_dir = "static/tts/temp"
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
    학습 스캔(scan_count) 포인트는 올리지 않음.
    """
    # 유저의 기존 단어장 조회
    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    already_exists = []
    new_candidates_for_generation = {}

    for korean_word, info in req.extracted_words.items():
        if korean_word in word_map:
            # 이미 단어장에 있는 경우 포인트(scan_count) 증가 없이 상태만 반환
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
            output_dir=user_output_dir,
            target_language=req.target_language
        )
        
        # TODO: 생성된 generated_cards 데이터를 WordCard DB에 Insert (필요시 구현)

    return {
        "already_exists": already_exists, 
        "new_word_cards": generated_cards
    }

@router.delete("/cleanup")
async def cleanup_tts_temp():
    """static/tts/temp 폴더 내의 모든 파일을 즉시 삭제합니다."""
    temp_dir = Path("static/tts/temp")
    
    if not temp_dir.exists():
        return {"status": "success", "message": "삭제할 폴더가 이미 없습니다."}
    
    try:
        for file in temp_dir.iterdir():
            if file.is_file():
                os.remove(file)
            elif file.is_dir():
                shutil.rmtree(file)
                
        return {"status": "success", "message": "임시 오디오 파일 정리가 완료되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 중 오류 발생: {str(e)}")
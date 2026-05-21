from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any

from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.models.scan_record import ScanRecord
from app.schemas.scans import AudioScanRequest, AudioScanResponse
from app.core.config import settings

from app.services.pipeline import extract_text_from_audio, KoreanLearningPipeline
from app.services.ytdlp import extract_youtube_audio

router = APIRouter(prefix="/scan", tags=["scan"])

# 서버 구동 시 파이프라인 객체 1회 초기화 (성능 최적화)
nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY, 
    dict_api_key=settings.DICT_API_KEY, 
    model_name=settings.LLM_MODEL_NAME
)


# [프론트엔드 HTML 테스트용 API] 
class GenerateRequest(BaseModel):
    selected_words: Dict[str, Any]
    target_language: str = "ru"

@router.post("/generate")
async def generate_cards_for_test(req: GenerateRequest):
    try:
        output_dir = "static/tts/test_user"
        final_cards = await nlp_pipeline.phase2_generate(
            selected_words=req.selected_words, 
            output_dir=output_dir,
            target_language=req.target_language 
        )
        return {"status": "success", "cards": final_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=AudioScanResponse)
async def process_scan_result(
    body: AudioScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """최종 선택된 단어들을 DB에 저장하고 처리하는 로직"""
    matched = []
    new_candidates_for_generation = {}

    # 1. 기존 유저 단어장 조회
    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    # 2. 추출된 단어 분류 (기존 DB 단어 vs 신규 단어)
    for word, info in body.extracted_words.items():
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
            new_candidates_for_generation[word] = info

    # 스캔 횟수 증가분 우선 반영
    db.commit()

    generated_cards = []
    
    # 3. 신규 단어 파이프라인 처리 및 DB Insert
    if new_candidates_for_generation:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir,
            target_language=body.target_language
        )
        
        # 새롭게 생성된 단어 카드를 WordCard DB에 등록
        for card_data in generated_cards:
            new_word_card = WordCard(
                user_id=current_user.id,
                korean_word=card_data["word"],
                english_meaning=card_data["definition_english"],
                korean_definition=card_data["definition_korean"],
                part_of_speech=card_data["pos_type"],
                semantic_category=card_data["semantic_category"],
                pronunciation=card_data["pronunciation"],
                audio_path_word=card_data["audio"]["word_tts"],
                scan_count=1  # 처음 생성 시 기본 1회 스캔으로 인정
            )
            db.add(new_word_card)
        
        # 신규 단어 DB 반영
        db.commit()

    return AudioScanResponse(
        matched_words=matched, 
        new_word_cards=generated_cards 
    )


class YouTubeScanRequest(BaseModel):
    url: str
    end_sec: float
    duration_sec: float = 10.0


@router.post("/youtube")
async def scan_youtube(req: YouTubeScanRequest):
    """
    [Live 모드] YouTube URL + 재생 타임스탬프 → yt-dlp로 구간 추출 → STT + 어휘 분석
    end_sec: 현재 재생 위치(초), duration_sec: 캡처 길이(기본 10초)
    """
    try:
        audio_bytes = await extract_youtube_audio(req.url, req.end_sec, req.duration_sec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 추출 실패: {e}")

    raw_text = extract_text_from_audio(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )
    result = await nlp_pipeline.phase1_analyze(raw_text)

    return {
        "scan_source": "youtube_live",
        "raw_text": raw_text,
        "corrected_text": result["corrected_text"],
        "llm_raw_output": result["llm_raw_output"],
        "extracted_words": result["extracted_words"],
        "candidates": result["candidates"],
    }


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    scan_source: str = Form("mic"),
):
    """
    오디오 파일을 받아 Demucs → Whisper → LLM 분석 후 모든 중간 결과를 HTML에 반환.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    audio_bytes = await file.read()
    
    raw_whisper_text = extract_text_from_audio(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )
    
    analysis_result = await nlp_pipeline.phase1_analyze(raw_whisper_text)
    
    return {
        "scan_source": scan_source,
        "raw_text": raw_whisper_text,
        "corrected_text": analysis_result["corrected_text"],
        "llm_raw_output": analysis_result["llm_raw_output"],
        "extracted_words": analysis_result["extracted_words"],
        "candidates": analysis_result["candidates"]
    }
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

@router.post("/generate")
async def generate_cards_for_test(req: GenerateRequest):
    """Phase 2: 선택된 단어 기반 학습 카드 및 TTS 생성 (HTML 테스트 전용)"""
    try:
        # 프론트에서 접근 가능한 static 폴더에 저장
        output_dir = "static/tts/test_user"
        final_cards = await nlp_pipeline.phase2_generate(req.selected_words, output_dir)
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

    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

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

    db.commit()

    generated_cards = []
    if new_candidates_for_generation:
        user_output_dir = f"static/tts/user_{current_user.id}"
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir
        )

    return AudioScanResponse(
        matched_words=matched, 
        new_word_cards=generated_cards 
    )


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    scan_source: str = Form("mic"),
    # 🚨 주의: HTML 프로토타입 테스트 시에는 Depends(get_current_user)를 주석 처리해야 401 에러가 나지 않습니다.
):
    """
    오디오 파일을 받아 Demucs → Whisper → LLM 분석 후 모든 중간 결과를 HTML에 반환.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    audio_bytes = await file.read()
    
    # 1. 오디오 엔진 (Demucs + Whisper) 실행
    raw_whisper_text = extract_text_from_audio(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )
    
    # 2. NLP 파이프라인 분석 실행 (Phase 1: 보정, 추출, 사전 검증)
    analysis_result = await nlp_pipeline.phase1_analyze(raw_whisper_text)
    
    return {
        "scan_source": scan_source,
        "raw_text": raw_whisper_text,
        "corrected_text": analysis_result["corrected_text"],
        "llm_raw_output": analysis_result["llm_raw_output"],    # LLM 답변 원문 추가
        "extracted_words": analysis_result["extracted_words"], # 필터링 전 단어 리스트 추가
        "candidates": analysis_result["candidates"]            # 사전 검증 완료 단어
    }
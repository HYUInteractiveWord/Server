from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.word_card import WordCard
from app.models.scan_record import ScanRecord
from app.schemas.scan import AudioScanRequest, AudioScanResponse
from app.core.config import settings

from app.services.pipeline import extract_text_from_audio, KoreanLearningPipeline

router = APIRouter(prefix="/scan", tags=["scan"])

# 서버 구동 시 파이프라인 객체 1회 초기화 (메모리 절약)
nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY, 
    dict_api_key=settings.DICT_API_KEY, 
    model_name=settings.LLM_MODEL_NAME # "gemma-4"
)

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    scan_source: str = Form("mic"),
    current_user: User = Depends(get_current_user),
):
    """
    [Phase 1] 오디오 수신 -> Demucs/Whisper STT -> LLM 교정 및 후보 추출 반환
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. (wav/mp3/m4a/ogg)")

    audio_bytes = await file.read()
    
    # 1. Demucs + Whisper 파이프라인 실행
    raw_whisper_text = extract_text_from_audio(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )
    
    # 2. LLM/NLP 파이프라인 Phase 1 실행
    analysis_result = await nlp_pipeline.phase1_analyze(raw_whisper_text)
    
    # 앱 프론트엔드로 Raw 텍스트, 교정된 텍스트, 그리고 사전에 등록된 유효한 후보 단어들 반환
    return {
        "scan_source": scan_source,
        "raw_text": analysis_result["raw_text"],
        "corrected_text": analysis_result["corrected_text"],
        "llm_raw_output": analysis_result["llm_raw_output"],    
        "extracted_words": analysis_result["extracted_words"],  
        "candidates": analysis_result["candidates"] 
    }


@router.post("/process", response_model=AudioScanResponse)
async def process_scan_result(
    body: AudioScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    [Phase 2] 앱에서 선택된 단어를 수신하여 중복 체크 후 LLM 단어장(TTS 포함) 생성
    """
    matched = []
    new_candidates_for_generation = {}

    # 유저의 기존 단어장 조회
    user_words = db.query(WordCard).filter(WordCard.user_id == current_user.id).all()
    word_map = {card.korean_word: card for card in user_words}

    # body.extracted_words는 클라이언트가 선택한 딕셔너리 {"단어": {"pos":"명사", "definition":"..."}}
    for word, info in body.extracted_words.items():
        if word in word_map:
            # 이미 단어장에 있는 단어는 카운트만 증가
            card = word_map[word]
            card.scan_count += 1
            db.add(ScanRecord(
                user_id=current_user.id,
                word_card_id=card.id,
                scan_source=body.scan_source,
            ))
            matched.append({"word_card_id": card.id, "korean_word": word, "scan_count": card.scan_count})
        else:
            # 단어장에 없는 새로운 단어만 LLM 생성 리스트로 분류
            new_candidates_for_generation[word] = info

    db.commit()

    generated_cards = []
    
    # 새로운 단어가 있다면 LLM/TTS 처리 시작
    if new_candidates_for_generation:
        # 각 유저별 격리된 폴더 경로 생성
        user_output_dir = f"./output_vocab/user_{current_user.id}"
        
        # 파이프라인 Phase 2 실행 (내부적으로 각 단어 폴더에 TTS 및 JSON 저장됨)
        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir
        )
        
        # TODO: generated_cards 데이터를 WordCard DB에 Insert 하는 로직 구현 필요

    return AudioScanResponse(
    matched_words=matched, 
    new_word_candidates=generated_cards
)

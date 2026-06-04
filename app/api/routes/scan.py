import subprocess
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.db import get_db
from app.models.scan_record import ScanRecord
from app.models.user import User
from app.models.word_card import WordCard
from app.schemas.scans import AudioScanRequest, AudioScanResponse
from app.services.pipeline import KoreanLearningPipeline, extract_text_from_audio


router = APIRouter(prefix="/scan", tags=["scan"])

# 서버 구동 시 파이프라인 객체 1회 초기화
nlp_pipeline = KoreanLearningPipeline(
    term_api_key=settings.TERM_API_KEY,
    dict_api_key=settings.DICT_API_KEY,
    model_name=settings.LLM_MODEL_NAME,
)


class GenerateRequest(BaseModel):
    selected_words: Dict[str, Any]
    target_language: str = "ru"


class YouTubeScanRequest(BaseModel):
    transcript_text: str


def _norm_path(path: str | None) -> str | None:
    if not path:
        return None

    return path.replace("\\", "/")


def _normalize_target_language(language: str | None) -> str:
    lang = (language or "ko").strip().lower().split("-")[0]

    if lang in {"ko", "en", "ru"}:
        return lang

    return "ko"


def _pick_translated_definition(card_data: dict) -> str | None:
    candidates = [
        card_data.get("definition_translated"),
        card_data.get("translated_definition"),
        card_data.get("translated_def_text"),
        card_data.get("definition_target"),
        card_data.get("definition_ru"),
        card_data.get("def_translated"),
        card_data.get("def_translation"),
        card_data.get("meaning_translated"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _pick_def_trans_audio_path(card_data: dict) -> str | None:
    audio = card_data.get("audio", {}) or {}

    candidates = [
        card_data.get("def_trans_audio_path"),
        card_data.get("definition_trans_audio_path"),
        card_data.get("translated_definition_audio_path"),
        card_data.get("def_trans_tts"),
        audio.get("def_trans_audio_path"),
        audio.get("definition_trans_audio_path"),
        audio.get("translated_definition_audio_path"),
        audio.get("def_trans_tts"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return _norm_path(value.strip())

    return None


def _pick_word_tts_path(card_data: dict) -> str | None:
    audio = card_data.get("audio", {}) or {}

    candidates = [
        audio.get("word_tts"),
        card_data.get("tts_audio_path"),
        card_data.get("audio_path"),
    ]

    for value in candidates:
        if isinstance(value, str) and value.strip():
            return _norm_path(value.strip())

    return None


def _normalize_examples(card_data: dict) -> list:
    examples = card_data.get("examples", []) or []
    audio = card_data.get("audio", {}) or {}
    example_tts_paths = audio.get("examples_tts", []) or []

    normalized_examples = []

    for index, example in enumerate(examples):
        if isinstance(example, dict):
            item = dict(example)
        else:
            item = {
                "type": "fallback",
                "korean": str(example),
                "english": "",
            }

        if index < len(example_tts_paths) and not item.get("tts_audio_path"):
            item["tts_audio_path"] = _norm_path(example_tts_paths[index])

        if item.get("audio_path"):
            item["audio_path"] = _norm_path(item.get("audio_path"))

        if item.get("tts_audio_path"):
            item["tts_audio_path"] = _norm_path(item.get("tts_audio_path"))

        if item.get("trans_audio_path"):
            item["trans_audio_path"] = _norm_path(item.get("trans_audio_path"))

        normalized_examples.append(item)

    return normalized_examples


def _empty_scan_response(scan_source: str, raw_text: str = "") -> dict:
    return {
        "scan_source": scan_source,
        "raw_text": raw_text,
        "corrected_text": "",
        "llm_raw_output": "",
        "extracted_words": [],
        "candidates": {},
    }


def _audio_has_enough_signal(audio_bytes: bytes) -> bool:
    """
    Whisper 호출 전에 오디오에 실제 음성으로 볼 만한 신호가 있는지 검사한다.

    방식:
    - ffmpeg로 입력 오디오를 16kHz mono PCM으로 변환
    - 20ms 단위 frame의 RMS를 계산
    - peak RMS, 평균 RMS, 유효 frame 비율이 너무 낮으면 무음/저음량으로 판단

    이 함수는 Whisper가 만든 텍스트를 후처리로 막는 것이 아니라,
    무음/저음량 오디오가 STT로 넘어가는 것 자체를 막기 위한 1차 방어다.
    """
    if not audio_bytes:
        return False

    ffmpeg_bin = settings.FFMPEG_BIN or "ffmpeg"

    try:
        process = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        pcm = process.stdout

        if process.returncode != 0 or not pcm:
            print(
                "[Scan Upload] audio signal check failed: "
                f"{process.stderr.decode(errors='ignore')}",
                flush=True,
            )
            # 검사 실패만으로 정상 음성을 막으면 안 되므로 통과
            return True

        sample_width = 2
        sample_rate = 16000
        frame_ms = 20
        frame_bytes = int(sample_rate * frame_ms / 1000) * sample_width

        if len(pcm) < frame_bytes:
            return False

        # 혹시 모를 홀수 바이트 방지
        if len(pcm) % sample_width != 0:
            pcm = pcm[: len(pcm) - (len(pcm) % sample_width)]

        rms_values = []

        for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
            frame = pcm[start:start + frame_bytes]

            if len(frame) < frame_bytes:
                continue

            samples = memoryview(frame).cast("h")

            if not samples:
                continue

            square_sum = sum(sample * sample for sample in samples)
            rms = (square_sum / len(samples)) ** 0.5
            rms_values.append(rms)

        if not rms_values:
            return False

        avg_rms = sum(rms_values) / len(rms_values)
        peak_rms = max(rms_values)

        # 20ms frame 중 일정 음량 이상인 frame 비율
        voiced_threshold = 350
        voiced_frames = [value for value in rms_values if value >= voiced_threshold]
        voiced_ratio = len(voiced_frames) / len(rms_values)

        print(
            f"[Scan Upload] audio_signal avg_rms={avg_rms:.1f}, "
            f"peak_rms={peak_rms:.1f}, voiced_ratio={voiced_ratio:.3f}",
            flush=True,
        )

        # 데모 안정성 기준:
        # peak가 너무 낮으면 실제 음성으로 보기 어렵다.
        if peak_rms < 700:
            return False

        # 전체 평균이 낮고 유효 frame도 거의 없으면 무음/저음량.
        if avg_rms < 120 and voiced_ratio < 0.03:
            return False

        # 유효 frame 비율이 지나치게 낮으면 무음/잡음으로 판단.
        if voiced_ratio < 0.01:
            return False

        return True

    except Exception as e:
        print(f"[Scan Upload] audio signal check error: {e}", flush=True)
        # 검사 실패만으로 정상 음성을 막으면 안 되므로 통과
        return True


def _is_unusable_whisper_text(text: str, scan_source: str) -> bool:
    """
    오디오 신호 검사를 통과한 뒤에도 남는 최소한의 텍스트 안전장치.

    핵심 방어는 _audio_has_enough_signal()에서 수행한다.
    여기서는 빈 문자열, 숫자/기호뿐인 결과 정도만 막는다.
    """
    normalized = " ".join((text or "").strip().split())

    if not normalized:
        return True

    compact = "".join(ch for ch in normalized if not ch.isspace())

    if not compact:
        return True

    if len(compact) < 2:
        return True

    if not any(ch.isalpha() for ch in compact):
        return True

    return False


@router.post("/generate")
async def generate_cards_for_test(req: GenerateRequest):
    """
    프론트엔드 HTML 테스트용 API.
    선택된 단어 후보를 단어카드 형태로 생성한다.
    """
    try:
        output_dir = "static/tts/test_user"
        final_cards = await nlp_pipeline.phase2_generate(
            selected_words=req.selected_words,
            output_dir=output_dir,
            target_language=_normalize_target_language(req.target_language),
        )

        return {
            "status": "success",
            "cards": final_cards,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=AudioScanResponse)
async def process_scan_result(
    body: AudioScanRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    최종 선택된 단어들을 DB에 저장하고 처리한다.

    - 기존 단어장에 있는 단어는 scan_count만 증가
    - 신규 단어는 phase2_generate 결과를 WordCard에 저장
    - 러시아어/영어 계정용 번역 뜻과 번역 TTS 경로도 함께 저장
    """
    matched = []
    new_candidates_for_generation = {}

    user_words = (
        db.query(WordCard)
        .filter(WordCard.user_id == current_user.id)
        .all()
    )
    word_map = {card.korean_word: card for card in user_words}

    for word, info in body.extracted_words.items():
        if word in word_map:
            card = word_map[word]
            card.scan_count = (card.scan_count or 0) + 1

            db.add(
                ScanRecord(
                    user_id=current_user.id,
                    word_card_id=card.id,
                    scan_source=body.scan_source,
                )
            )

            matched.append(
                {
                    "word_card_id": card.id,
                    "korean_word": word,
                    "scan_count": card.scan_count,
                }
            )
        else:
            new_candidates_for_generation[word] = info

    db.commit()

    generated_cards = []

    if new_candidates_for_generation:
        user_output_dir = f"static/tts/user_{current_user.id}"

        target_language = _normalize_target_language(
            getattr(body, "target_language", None)
            or getattr(current_user, "preferred_language", None)
        )

        generated_cards = await nlp_pipeline.phase2_generate(
            selected_words=new_candidates_for_generation,
            output_dir=user_output_dir,
            target_language=target_language,
        )

        for card_data in generated_cards:
            word = card_data.get("word", "")

            if not word:
                continue

            new_word_card = WordCard(
                user_id=current_user.id,
                korean_word=word,
                pos=card_data.get("pos_type") or card_data.get("pos") or "",
                definition=card_data.get("definition_korean") or "",
                definition_english=card_data.get("definition_english"),
                definition_translated=_pick_translated_definition(card_data),
                example_sentences=_normalize_examples(card_data),
                tts_audio_path=_pick_word_tts_path(card_data),
                def_trans_audio_path=_pick_def_trans_audio_path(card_data),
                pronunciation=card_data.get("pronunciation", ""),
                source=body.scan_source if hasattr(body, "scan_source") else "scan",
                scan_count=1,
            )

            db.add(new_word_card)
            db.flush()

            db.add(
                ScanRecord(
                    user_id=current_user.id,
                    word_card_id=new_word_card.id,
                    scan_source=body.scan_source,
                )
            )

        db.commit()

    return AudioScanResponse(
        matched_words=matched,
        new_word_cards=generated_cards,
    )


@router.post("/youtube")
async def scan_youtube(req: YouTubeScanRequest):
    """
    Android에서 직접 추출한 YouTube 자막 텍스트를 LLM 어휘 분석으로 넘긴다.
    """
    transcript_text = (req.transcript_text or "").strip()

    if not transcript_text:
        return _empty_scan_response(
            scan_source="youtube_transcript",
            raw_text="",
        )

    result = await nlp_pipeline.phase1_analyze(transcript_text)

    return {
        "scan_source": "youtube_transcript",
        "raw_text": transcript_text,
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
    오디오 파일을 받아 Whisper → LLM 분석 후 결과를 반환한다.

    mic 입력은 먼저 오디오 신호 검사를 수행한다.
    무음/너무 약한 입력이면 Whisper를 호출하지 않고 빈 결과를 반환한다.
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    audio_bytes = await file.read()

    if scan_source == "mic":
        if not _audio_has_enough_signal(audio_bytes):
            print("[Scan Upload] skipped whisper: silent or too weak audio", flush=True)
            return _empty_scan_response(
                scan_source=scan_source,
                raw_text="",
            )

    raw_whisper_text = extract_text_from_audio(
        audio_bytes=audio_bytes,
        ffmpeg_bin=settings.FFMPEG_BIN,
        whisper_model_size=settings.WHISPER_MODEL,
    )

    normalized_text = (raw_whisper_text or "").strip()

    print(
        f"[Scan Upload] source={scan_source}, raw_whisper_text={normalized_text!r}",
        flush=True,
    )

    if _is_unusable_whisper_text(normalized_text, scan_source):
        return _empty_scan_response(
            scan_source=scan_source,
            raw_text=normalized_text,
        )

    analysis_result = await nlp_pipeline.phase1_analyze(normalized_text)

    return {
        "scan_source": scan_source,
        "raw_text": normalized_text,
        "corrected_text": analysis_result["corrected_text"],
        "llm_raw_output": analysis_result["llm_raw_output"],
        "extracted_words": analysis_result["extracted_words"],
        "candidates": analysis_result["candidates"],
    }
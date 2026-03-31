from pydantic import BaseModel
from typing import Optional


class AudioScanRequest(BaseModel):
    """임성빈(배경음 제거) + 서정빈(STT) 처리 후 결과를 이 스키마로 수신"""
    extracted_words: list[str]       # 추출된 단어 후보 리스트
    full_text: Optional[str] = None  # 전체 STT 텍스트 (참고용)
    scan_source: str = "mic"         # "mic" | "media"


class AudioScanResponse(BaseModel):
    matched_words: list[dict]        # 단어장에 이미 있는 단어 (scan_count 증가)
    new_word_candidates: list[str]   # 단어장에 없는 새 단어 후보

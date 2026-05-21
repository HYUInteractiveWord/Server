from pydantic import BaseModel
from typing import Optional


class AudioScanRequest(BaseModel):
    extracted_words: dict 
    target_language: str = "en"
    scan_source: str = "mic"

class AudioScanResponse(BaseModel):
    matched_words: list[dict]
    new_word_cards: list[dict]

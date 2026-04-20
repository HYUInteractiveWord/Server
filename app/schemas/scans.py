from pydantic import BaseModel
from typing import Optional


class AudioScanRequest(BaseModel):
    extracted_words: dict 
    scan_source: str = "mic"

class AudioScanResponse(BaseModel):
    matched_words: list[dict]
    new_word_cards: list[dict]

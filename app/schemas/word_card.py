from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class WordCardCreate(BaseModel):
    korean_word: str
    source: str = "dictionary"  # "scan" | "dictionary"


class WordCardResponse(BaseModel):
    id: int
    korean_word: str
    pos: Optional[str]
    definition: Optional[str]
    example_sentences: Optional[list]
    tts_audio_path: Optional[str]
    level: int
    best_score: float
    scan_count: int
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}


class WordCardUpdate(BaseModel):
    level: Optional[int] = None
    best_score: Optional[float] = None

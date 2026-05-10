from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class WordCardCreate(BaseModel):
    korean_word: str
    source: str = "dictionary"  # "scan" | "dictionary"
    pos: Optional[str] = None
    definition: Optional[str] = None


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
    word_point: int = 0
    speaking_count: int = 0
    effect_level: int = 0
    source: str
    created_at: datetime

    model_config = {"from_attributes": True}


class WordCardUpdate(BaseModel):
    level: Optional[int] = None
    best_score: Optional[float] = None

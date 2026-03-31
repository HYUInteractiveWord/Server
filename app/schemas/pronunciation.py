from pydantic import BaseModel
from typing import Optional


class PronunciationSubmit(BaseModel):
    """김성준(피치 분석) 모듈에서 결과를 이 스키마로 수신"""
    word_card_id: int
    score: float                          # 0~100
    user_pitch_data: list[float]          # 사용자 피치 배열
    reference_pitch_data: list[float]     # TTS 기준 피치 배열
    dtw_distance: Optional[float] = None


class PronunciationResponse(BaseModel):
    record_id: int
    score: float
    is_new_best: bool
    xp_gained: int
    word_card_level: int

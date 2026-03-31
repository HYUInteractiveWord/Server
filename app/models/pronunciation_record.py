from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class PronunciationRecord(Base):
    __tablename__ = "pronunciation_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    word_card_id = Column(Integer, ForeignKey("word_cards.id"), nullable=False)

    score = Column(Float, nullable=False)         # 0~100
    user_pitch_data = Column(JSON)                # 김성준 모듈에서 넘어오는 피치 배열
    reference_pitch_data = Column(JSON)           # TTS 기준 피치 배열
    dtw_distance = Column(Float)                  # raw DTW 거리 값

    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    word_card = relationship("WordCard", back_populates="pronunciation_records")

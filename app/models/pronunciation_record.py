from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class PronunciationRecord(Base):
    __tablename__ = "pronunciation_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    word_card_id = Column(Integer, ForeignKey("word_cards.id"), nullable=False)

    score = Column(Float, nullable=False)
    
    pronunciation_score = Column(Float)
    formant_score = Column(Float)
    pitch_score = Column(Float)
    timing_score = Column(Float)
    is_intensity_good = Column(Boolean)
    
    xp_gained = Column(Integer, default=0)

    user_pitch_data = Column(JSON)
    reference_pitch_data = Column(JSON)
    dtw_distance = Column(Float)

    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    word_card = relationship("WordCard", back_populates="pronunciation_records")

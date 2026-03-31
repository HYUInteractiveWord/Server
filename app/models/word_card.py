from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class WordCard(Base):
    __tablename__ = "word_cards"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    korean_word = Column(String, nullable=False)
    pos = Column(String)                  # 품사: 명사, 동사, 형용사 등
    definition = Column(String)           # 국립국어원 사전 뜻
    example_sentences = Column(JSON)      # LLM 생성 예문 리스트
    tts_audio_path = Column(String)       # TTS 파일 경로

    # 학습 현황
    level = Column(Integer, default=1)    # 1~5
    best_score = Column(Float, default=0.0)  # 최고 발음 점수 0~100
    scan_count = Column(Integer, default=0)  # 스캔된 횟수

    # 수집 방법: "scan" | "dictionary"
    source = Column(String, default="dictionary")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    user = relationship("User", back_populates="word_cards")
    scan_records = relationship("ScanRecord", back_populates="word_card", cascade="all, delete-orphan")
    pronunciation_records = relationship("PronunciationRecord", back_populates="word_card", cascade="all, delete-orphan")

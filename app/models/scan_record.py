from sqlalchemy import Column, Integer, ForeignKey, DateTime, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    word_card_id = Column(Integer, ForeignKey("word_cards.id"), nullable=False)

    # "mic" | "media"
    scan_source = Column(String, default="mic")
    scanned_at = Column(DateTime(timezone=True), server_default=func.now())

    word_card = relationship("WordCard", back_populates="scan_records")

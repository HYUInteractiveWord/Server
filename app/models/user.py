from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # 게이미피케이션
    xp = Column(Integer, default=0)
    rank = Column(String, default="Bronze")
    max_word_slots = Column(Integer, default=20)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    word_cards = relationship("WordCard", back_populates="user", cascade="all, delete-orphan")
    missions = relationship("Mission", back_populates="user", cascade="all, delete-orphan")

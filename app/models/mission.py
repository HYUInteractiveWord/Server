from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db import Base


class Mission(Base):
    __tablename__ = "missions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 미션 종류: "daily_pronunciation" | "daily_scan" | "collect_pos" | "collect_topic"
    mission_type = Column(String, nullable=False)
    # 세부 파라미터: 예) pos="동사", topic="과학"
    parameter = Column(String)

    progress = Column(Integer, default=0)
    target = Column(Integer, nullable=False)
    is_completed = Column(Boolean, default=False)

    xp_reward = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="missions")

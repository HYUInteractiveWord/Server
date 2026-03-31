from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.mission import Mission
from app.schemas.mission import MissionResponse

router = APIRouter(prefix="/missions", tags=["missions"])


@router.get("/", response_model=list[MissionResponse])
def get_my_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Mission).filter(Mission.user_id == current_user.id).all()


@router.get("/daily")
def get_daily_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """오늘의 미션 목록 반환 (일일 발음 연습, 일일 스캔 등)"""
    from datetime import date
    today_start = date.today()
    missions = db.query(Mission).filter(
        Mission.user_id == current_user.id,
        Mission.mission_type.in_(["daily_pronunciation", "daily_scan"]),
        Mission.is_completed == False,
    ).all()
    return missions

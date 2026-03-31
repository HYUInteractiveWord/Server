from datetime import date, datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.mission import Mission
from app.schemas.mission import MissionResponse
from app.services.gamification import get_rank_for_xp, RANK_WORD_SLOTS

router = APIRouter(prefix="/missions", tags=["missions"])

DAILY_MISSION_TEMPLATES = [
    {"mission_type": "daily_pronunciation", "target": 3, "xp_reward": 150},
    {"mission_type": "daily_scan", "target": 5, "xp_reward": 150},
]


def _get_or_create_daily_missions(user_id: int, db: Session) -> list[Mission]:
    today = date.today()
    existing = db.query(Mission).filter(
        Mission.user_id == user_id,
        Mission.mission_type.in_(["daily_pronunciation", "daily_scan"]),
        func.date(Mission.created_at) == today,
    ).all()

    if existing:
        return existing

    missions = [Mission(user_id=user_id, **t) for t in DAILY_MISSION_TEMPLATES]
    for m in missions:
        db.add(m)
    db.commit()
    for m in missions:
        db.refresh(m)
    return missions


@router.get("/", response_model=list[MissionResponse])
def get_my_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Mission).filter(Mission.user_id == current_user.id).all()


@router.get("/daily", response_model=list[MissionResponse])
def get_daily_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """오늘의 일일 미션 반환. 없으면 자동 생성."""
    return _get_or_create_daily_missions(current_user.id, db)


@router.post("/{mission_id}/complete", response_model=MissionResponse)
def complete_mission(
    mission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """미션 완료 처리 및 XP 지급."""
    mission = db.query(Mission).filter(
        Mission.id == mission_id,
        Mission.user_id == current_user.id,
    ).first()
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    if mission.is_completed:
        raise HTTPException(status_code=400, detail="Mission already completed")
    if mission.progress < mission.target:
        raise HTTPException(
            status_code=400,
            detail=f"Mission not yet finished ({mission.progress}/{mission.target})",
        )

    mission.is_completed = True
    mission.completed_at = datetime.now(timezone.utc)

    current_user.xp += mission.xp_reward
    new_rank = get_rank_for_xp(current_user.xp)
    if new_rank != current_user.rank:
        current_user.rank = new_rank
        current_user.max_word_slots = RANK_WORD_SLOTS[new_rank]

    db.commit()
    db.refresh(mission)
    return mission

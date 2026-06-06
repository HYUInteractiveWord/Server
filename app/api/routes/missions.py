from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
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
    {"mission_type": "daily_word_quiz", "target": 1, "xp_reward": 150},
    {"mission_type": "daily_collect_noun", "target": 1, "xp_reward": 100},
]

def _get_or_create_daily_missions(user_id: int, db: Session) -> list[Mission]:
    today = datetime.now(timezone(timedelta(hours=9))).date()
    mission_types = [t["mission_type"] for t in DAILY_MISSION_TEMPLATES]

    existing = db.query(Mission).filter(
        Mission.user_id == user_id,
        Mission.mission_type.in_(mission_types),
    ).all()

    existing_by_type = {mission.mission_type: mission for mission in existing}
    missions: list[Mission] = []

    for template in DAILY_MISSION_TEMPLATES:
        mission_type = template["mission_type"]
        mission = existing_by_type.get(mission_type)

        if mission is None:
            mission = Mission(
                user_id=user_id,
                mission_type=mission_type,
                target=template["target"],
                xp_reward=template["xp_reward"],
                progress=0,
                is_completed=False,
                completed_at=None,
                last_reset_date=today,
            )
            db.add(mission)
        else:
            if mission.last_reset_date != today:
                mission.progress = 0
                mission.is_completed = False
                mission.completed_at = None
                mission.last_reset_date = today

            mission.target = template["target"]
            mission.xp_reward = template["xp_reward"]

        missions.append(mission)

    db.commit()

    for mission in missions:
        db.refresh(mission)

    return sorted(
        missions,
        key=lambda m: mission_types.index(m.mission_type)
        if m.mission_type in mission_types
        else len(mission_types),
    )

@router.get("/", response_model=list[MissionResponse])
def get_my_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _get_or_create_daily_missions(current_user.id, db)

@router.get("/daily", response_model=list[MissionResponse])
def get_daily_missions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """오늘의 일일 미션 반환. 없으면 자동 생성."""
    return _get_or_create_daily_missions(current_user.id, db)

@router.post("/{mission_id}/complete", response_model=MissionResponse)
def complete_mission(mission_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    mission = db.query(Mission).filter(Mission.id == mission_id, Mission.user_id == current_user.id).first()
    
    if not mission:
        raise HTTPException(status_code=404, detail="Mission not found")
    if mission.progress < mission.target:
        raise HTTPException(status_code=400, detail="미션이 아직 완료되지 않았습니다.")

    # 1. 보상 지급 및 랭크 반영
    current_user.xp += mission.xp_reward
    new_rank = get_rank_for_xp(current_user.xp)
    if new_rank != current_user.rank:
        current_user.rank = new_rank
        current_user.max_word_slots = RANK_WORD_SLOTS.get(new_rank, 20)

    # 2. 삭제 전 반환용 데이터 복사 (DetachedError 방지)
    mission_response_data = {
        "id": mission.id,
        "user_id": mission.user_id,
        "mission_type": mission.mission_type,
        "parameter": mission.parameter,
        "progress": mission.progress,
        "target": mission.target,
        "is_completed": True,
        "xp_reward": mission.xp_reward,
        "created_at": mission.created_at,
        "completed_at": datetime.now(timezone.utc),
        "last_reset_date": mission.last_reset_date
    }

    db.delete(mission) 
    db.commit()
    
    return mission_response_data
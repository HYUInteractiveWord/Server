from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class MissionResponse(BaseModel):
    id: int
    mission_type: str
    parameter: Optional[str]
    progress: int
    target: int
    is_completed: bool
    xp_reward: int
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}

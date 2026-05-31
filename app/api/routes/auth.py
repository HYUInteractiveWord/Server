from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.mission import Mission
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

INITIAL_DAILY_MISSIONS = [
    {"mission_type": "daily_pronunciation", "target": 3, "xp_reward": 150},
    {"mission_type": "daily_scan", "target": 5, "xp_reward": 150},
    {"mission_type": "daily_word_quiz", "target": 1, "xp_reward": 150},
    {"mission_type": "daily_collect_noun", "target": 1, "xp_reward": 100},
]


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
        preferred_language=body.preferred_language,
    )
    db.add(user)
    db.flush()

    for m in INITIAL_DAILY_MISSIONS:
        db.add(Mission(user_id=user.id, **m))

    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": str(user.id)})
    return {
        "access_token": token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
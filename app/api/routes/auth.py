from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.mission import Mission
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse
from app.core.security import hash_password, verify_password, create_access_token
from app.models.word_card import WordCard
import uuid
import os
import json
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

@router.post("/demo", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_demo_user_and_login(
    body: UserCreate, db: Session = Depends(get_db)
):
    """
    [앱 연동용] 데모 유저를 즉시 생성하고 10000 XP를 부여한 뒤, 
    해당 언어의 데모 단어장을 이식하고 바로 로그인할 수 있는 토큰을 반환합니다.
    """
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
    
    # 2. 언어별 경로 설정 로직
    target_lang = body.preferred_language.strip().lower().split("-")[0]
    if target_lang not in ["ru", "en", "ko"]:
        target_lang = "ko"
        
    demo_dir = f"assets/demo_data/{target_lang}"
    all_vocab_path = os.path.join(demo_dir, "all_vocab_cards.json")
    
    # 안전장치: 폴더가 없으면 기본 경로 사용
    if not os.path.exists(all_vocab_path):
        all_vocab_path = "assets/demo_data/all_vocab_cards.json"
        
    # 3. 데모 단어장 DB에 이식하기
    if os.path.exists(all_vocab_path):
        with open(all_vocab_path, "r", encoding="utf-8") as f:
            cards_data = json.load(f)
            
        for data in cards_data:
            audio = data.get("audio", {})
            word_tts = audio.get("word_tts", "")
            def_tts = audio.get("def_trans_tts", "")
            
            new_card = WordCard(
                user_id=user.id,
                korean_word=data.get("word"),
                source="demo",
                pos=data.get("pos_type"),
                definition=data.get("definition_korean"),
                definition_translated=data.get("definition_translated"),
                pronunciation=data.get("pronunciation"),
                example_sentences=data.get("examples", []),
                tts_audio_path=word_tts.replace("\\", "/") if word_tts else None,
                def_trans_audio_path=def_tts.replace("\\", "/") if def_tts else None,
            )
            db.add(new_card)
        db.commit()

    access_token = create_access_token(data={"sub": user.username})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "데모 계정이 성공적으로 생성되었습니다."
    }
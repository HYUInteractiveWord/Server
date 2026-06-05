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
    dbody: UserCreate, db: Session = Depends(get_db)
):
    """
    [앱 연동용] 데모 유저를 즉시 생성하고 10000 XP를 부여한 뒤, 
    해당 언어의 데모 단어장을 이식하고 바로 로그인할 수 있는 토큰을 반환합니다.
    """
    if db.query(User).filter(User.username == dbody.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == dbody.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=dbody.username,
        email=dbody.email,
        hashed_password=hash_password(dbody.password),
        preferred_language=dbody.preferred_language,
        xp=10000, 
        rank="Emerald",
        max_word_slots=100
    )
    db.add(user)
    db.flush()

    for m in INITIAL_DAILY_MISSIONS:
        db.add(Mission(user_id=user.id, **m))
    db.commit()
    db.refresh(user)
    
    # 2. 언어별 경로 설정 로직
    target_lang = dbody.preferred_language.strip().lower().split("-")[0]
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
            
        print(f"DEBUG: 로드된 데이터 타입: {type(cards_data)}")
        print(f"DEBUG: 전체 데이터 개수: {len(cards_data) if isinstance(cards_data, list) else '리스트 아님'}")
        if not isinstance(cards_data, list):
            print("ERROR: JSON 데이터가 리스트 형식이 아닙니다.")
        else:
            for i, data in enumerate(cards_data):
                try:
                    # 중복 체크
                    if db.query(WordCard).filter(
                        WordCard.user_id == user.id, 
                        WordCard.korean_word == data.get("word")
                    ).first():
                        continue

                    audio = data.get("audio", {})
                    new_card = WordCard(
                        user_id=user.id,
                        korean_word=data.get("word"),
                        source="demo",
                        pos=data.get("pos_type"),
                        definition=data.get("definition_korean"),
                        definition_translated=data.get("definition_translated"),
                        pronunciation=data.get("pronunciation"),
                        example_sentences=data.get("examples", []),
                        tts_audio_path=audio.get("word_tts", "").replace("\\", "/") if audio.get("word_tts") else None,
                        def_trans_audio_path=audio.get("def_trans_tts", "").replace("\\", "/") if audio.get("def_trans_tts") else None,
                    )
                    db.add(new_card)
                except Exception as e:
                    print(f"ERROR: {i}번째 단어 추가 실패: {e}")
                    continue 
            db.commit()
            print("INFO: 모든 단어 처리 완료 및 커밋 성공")
    return user
@router.delete("/delete")
def delete_current_user(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.commit()
    return {"message": "User deleted successfully"}
@router.patch("/delete", response_model=UserResponse)
def update_user_info(
    body: dict, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if "preferred_language" in body:
        new_lang = body["preferred_language"]
        current_user.preferred_language = new_lang
        
        db.query(WordCard).filter(WordCard.user_id == current_user.id).delete()
        
    db.commit()
    db.refresh(current_user)
    return current_user
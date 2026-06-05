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
from datetime import datetime, timezone, timedelta
import shutil
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
    today = datetime.now(timezone(timedelta(hours=9))).date()
    for m in INITIAL_DAILY_MISSIONS:
        db.add(Mission(user_id=user.id, last_reset_date=today, **m))
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
        
    if os.path.exists(demo_dir):
        # 유저 전용 기본 디렉토리 설정
        user_base_dir = os.path.join("static", "tts", f"user_{user.id}")
        os.makedirs(user_base_dir, exist_ok=True)

        for root, dirs, files in os.walk(demo_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    print(f"DEBUG: 처리 중인 파일 -> {file_path}")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data_content = json.load(f)
                            cards_data = data_content if isinstance(data_content, list) else [data_content]
                            
                        for data in cards_data:
                            word = data.get("word")
                            if not word: continue

                            # 중복 체크
                            if db.query(WordCard).filter(
                                WordCard.user_id == user.id,
                                WordCard.korean_word == word
                            ).first():
                                continue

                            # 1. 단어별 폴더 생성 (static/tts/user_{id}/단어/)
                            word_dir = os.path.join(user_base_dir, word)
                            os.makedirs(word_dir, exist_ok=True)

                            # 2. 오디오 파일 복사 및 경로 변환 함수
                            def copy_and_get_new_path(original_path_str):
                                if not original_path_str: return None
                                # 소스 파일 절대 경로 추정 (demo_dir 기반)
                                src_path = os.path.normpath(original_path_str)
                                if not os.path.exists(src_path):
                                    print(f"  WARNING: 원본 파일 없음 -> {src_path}")
                                    return None
                                
                                # 복사할 파일명 (예: 승차_word.mp3)
                                filename = os.path.basename(original_path_str)
                                dest_path = os.path.join(word_dir, filename)
                                
                                # 복사 수행
                                shutil.copy2(src_path, dest_path)
                                # DB에 저장될 경로 (static/... 으로 시작하는 상대 경로)
                                return dest_path.replace("\\", "/")

                            # 오디오 파일 처리
                            audio = data.get("audio", {})
                            word_tts = copy_and_get_new_path(audio.get("word_tts"))
                            def_tts = copy_and_get_new_path(audio.get("def_trans_tts"))
                            
                            # 예문 오디오 처리
                            examples_data = data.get("examples", [])
                            # 예문 오디오도 있다면 동일하게 복사하여 경로 교체
                            for ex in examples_data:
                                if isinstance(ex, dict):
                                    if ex.get("audio_path"):
                                        ex["audio_path"] = copy_and_get_new_path(ex["audio_path"])
                                    if ex.get("trans_audio_path"):
                                        ex["trans_audio_path"] = copy_and_get_new_path(ex["trans_audio_path"])

                            # DB 레코드 생성
                            new_card = WordCard(
                                user_id=user.id,
                                korean_word=word,
                                source="demo",
                                pos=data.get("pos_type"),
                                definition=data.get("definition_korean"),
                                definition_translated=data.get("definition_translated"),
                                pronunciation=data.get("pronunciation"),
                                example_sentences=examples_data,
                                tts_audio_path=word_tts,
                                def_trans_audio_path=def_tts,
                            )
                            db.add(new_card)
                            
                    except Exception as e:
                        print(f"ERROR: 파일 {file} 처리 중 오류 발생: {e}")
        db.commit()
        print("INFO: 유저 전용 폴더로 데이터 이관 완료")
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
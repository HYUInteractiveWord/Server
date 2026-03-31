from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from app.db import Base, engine
from app.api.routes import auth, words, scan, pronunciation, missions

# DB 테이블 생성 (개발용 - 프로덕션은 Alembic 마이그레이션 사용)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="InteractiveWord API",
    description="인터렉티브 단어장 백엔드 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중 전체 허용, 배포 시 프론트엔드 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTS 오디오 파일 정적 서빙
Path("static/tts").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(auth.router, prefix="/api")
app.include_router(words.router, prefix="/api")
app.include_router(scan.router, prefix="/api")
app.include_router(pronunciation.router, prefix="/api")
app.include_router(missions.router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}

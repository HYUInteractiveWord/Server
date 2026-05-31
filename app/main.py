from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # 추가
from pathlib import Path

from sqlalchemy import text
from app.db import Base, engine
from app.api.routes import auth, words, scan, pronunciation, missions, dictionary

Base.metadata.create_all(bind=engine)

# Safe migration: add preferred_language to existing databases
with engine.connect() as _conn:
    _conn.execute(text(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_language VARCHAR DEFAULT 'ko'"
    ))
    _conn.commit()

app = FastAPI(
    title="InteractiveWord API",
    description="인터렉티브 단어장 백엔드 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 폴더 설정
Path("static/tts").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth.router, prefix="/api")
app.include_router(words.router, prefix="/api")
app.include_router(scan.router, prefix="/api")
app.include_router(pronunciation.router, prefix="/api")
app.include_router(missions.router, prefix="/api")
app.include_router(dictionary.router, prefix="/api")

@app.get("/")
@app.get("/test")
async def serve_frontend():
    return FileResponse("static/test.html")

@app.get("/health")
def health_check():
    return {"status": "ok"}
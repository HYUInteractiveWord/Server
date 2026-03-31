# InteractiveWord Backend Server

인터렉티브 단어장 어플리케이션의 백엔드 API 서버입니다.

## 기술 스택

- **Framework**: FastAPI
- **Database**: PostgreSQL (Docker)
- **ORM**: SQLAlchemy
- **Auth**: JWT (python-jose)

## 시작하기

### 1. 환경 변수 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 필요한 값을 채워주세요. `DATABASE_URL`과 `SECRET_KEY`는 필수입니다.

### 2. 실행

**Windows:**
```powershell
start.bat
```

**수동 실행:**
```bash
docker-compose up -d          # DB 컨테이너 시작
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 3. API 문서

서버 실행 후 [http://localhost:8000/docs](http://localhost:8000/docs) 접속

## 프로젝트 구조

```
app/
├── main.py                 # FastAPI 앱 진입점
├── db.py                   # DB 연결
├── api/
│   ├── deps.py             # 공통 의존성 (인증)
│   └── routes/
│       ├── auth.py         # 회원가입/로그인
│       ├── words.py        # 단어카드 CRUD
│       ├── scan.py         # 오디오 스캔 결과 수신
│       ├── pronunciation.py # 발음 분석 결과 수신
│       └── missions.py     # 미션/경험치
├── models/                 # SQLAlchemy DB 모델
├── schemas/                # Pydantic 요청/응답 스키마
├── services/
│   ├── dictionary.py       # 국립국어원 사전 API
│   ├── tts.py              # TTS 오디오 생성
│   └── gamification.py     # XP/랭크/레벨 로직
└── core/
    ├── config.py           # 환경 변수
    └── security.py         # 비밀번호 해싱, JWT
```

## 주요 API 엔드포인트

| Method | URL | 설명 |
|--------|-----|------|
| POST | `/api/auth/register` | 회원가입 |
| POST | `/api/auth/login` | 로그인 (JWT 발급) |
| GET | `/api/words/` | 내 단어카드 목록 |
| POST | `/api/words/` | 단어카드 생성 |
| POST | `/api/scan/upload` | 오디오 파일 업로드 |
| POST | `/api/scan/process` | STT 결과 수신 및 처리 |
| POST | `/api/pronunciation/submit` | 발음 분석 결과 수신 |
| GET | `/api/missions/daily` | 오늘의 미션 |

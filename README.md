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

## 프론트 폴더 확인 이후 추가 구현 / 수정 사항 (2026.04.10)

### 1. 사용자 정보 조회 API 추가
프론트엔드의 Home / Profile 화면에서 로그인된 사용자 정보를 표시할 수 있도록, 현재 로그인한 사용자 정보를 조회하는 API를 추가했습니다.

- **Method**: `GET`
- **URL**: `/api/auth/me`

#### 응답 예시

```json
{
  "id": 1,
  "username": "testuser1",
  "email": "test1@example.com",
  "xp": 0,
  "rank": "Bronze",
  "max_word_slots": 20,
  "created_at": "2026-04-10T00:00:00"
}
```

---

### 2. 사전 검색 전용 API 추가
검색 결과를 먼저 보여준 뒤, 사용자가 저장 여부를 선택할 수 있도록 사전 검색만 수행하는 전용 API를 추가했습니다.

- **Method**: `GET`
- **URL**: `/api/dictionary/search?word=단어`

#### 응답 예시

```json
{
  "word": "사과",
  "pos": "명사",
  "definition": "먹는 열매"
}
```

---

### 3. TTS 경로 형식 수정
기존에는 TTS 파일 경로를 상대 경로로 반환했지만, 프론트엔드에서 바로 사용할 수 있도록 절대 URL 형식으로 수정했습니다.

#### 변경 전

```text
static/tts/파일명.mp3
```

#### 변경 후

```text
http://주소/static/tts/파일명.mp3
```

---

### 4. 로그인 요청 형식 정리
테스트 과정에서 Swagger OAuth2 Authorize와의 호환을 위해 한때 `OAuth2PasswordRequestForm` 기반의 `form-data` 방식으로 변경하여 확인했습니다.

최종적으로는 프론트엔드 연동 방식에 맞춰 다시 **JSON Body 방식**으로 복구했습니다.
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

.env 파일을 열어 필요한 값을 채워주세요. DATABASE_URL과 SECRET_KEY는 필수입니다.
(STT 사용을 위해 WHISPER_MODEL="medium", FFMPEG_BIN 등의 설정도 함께 확인해 주세요.)

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

서버 실행 후 http://localhost:8000/docs 접속

## 프로젝트 구조
외부 models폴더에 gemma4 gguf 모델 반드시 저장!

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
├── models/                 # SQLAlchemy DB 모델 - 유저, 미션, 발음레코드, 단어카드, 발음평가
├── schemas/                # Pydantic 요청/응답 스키마
├── services/
│   ├── dictionary.py       # 국립국어원 사전 API
│   ├── tts.py              # TTS 오디오 생성
│   ├── gamification.py     # XP/랭크/레벨 로직
│   ├── pipeline.py         # 단어카드 생성, 음성인식 전체 로직!!중요
    ├── evaluation.py       # 전체 음성평가 로직 통합 및 최종점수 계산(whisper 활용)
    ├── ytdlp.py            # 유튜브 스캔을 위한 기능(보안이슈 존재)
    ├── pronunciation/      # 음성 평가를 위한 4개 파이프라인 코드(분석>점수>시각화>서비스)
└── core/
    ├── config.py           # 환경 변수
    └── security.py         # 비밀번호 해싱, JWT
```

## 주요 API 엔드포인트

| Method | URL | 설명 |
POST	/api/auth/register	회원가입
POST	/api/auth/login	로그인 (JWT 발급)
GET	/api/auth/me	내 정보 조회 (XP, 랭크 등)
GET	/api/words/	내 단어카드 목록
POST	/api/words/	단어카드 생성
GET	/api/dictionary/search	사전 검색 (외국어/한국어 자동 분기)
POST	/api/scan/upload	오디오 파일 업로드 (STT 스캔)
POST	/api/scan/process	스캔된 단어 DB 저장 및 카드 생성
DELETE	/api/scan/cleanup	서버 내 임시 오디오/이미지 파일 즉시 정리
POST	/api/pronunciation/submit	발음 분석 결과 수신

## 프론트 폴더 확인 이후 추가 구현 / 수정 사항 (2026.04.10)

1. 사용자 정보 조회 API 추가
프론트엔드의 Home / Profile 화면에서 로그인된 사용자 정보를 표시할 수 있도록, 현재 로그인한 사용자 정보를 조회하는 API를 추가했습니다.

Method: GET

URL: /api/auth/me

응답 예시
JSON
{
  "id": 1,
  "username": "testuser1",
  "email": "test1@example.com",
  "xp": 0,
  "rank": "Bronze",
  "max_word_slots": 20,
  "created_at": "2026-06-13T00:00:00"
}
2. 다국어 지원 스마트 사전 검색 API 추가
사용자가 입력한 검색어가 한국어인지 외국어(영어/러시아어 등)인지 정규식으로 자동 감지하여 라우팅을 분기합니다. 외국어 입력 시 LLM 파이프라인을 가동하여 가장 알맞은 한국어 단어를 매칭하고, 프론트엔드에 대표 단어를 교체하여 응답합니다.

Method: GET

URL: /api/dictionary/search?word=단어

3. NLP 파이프라인 동음이의어 및 환각(Hallucination) 완벽 제어
다의어 검증 강화: 기초사전 API 호출 시 영어 번역(translated=y)을 함께 가져와 LLM이 문맥(expected_meaning)과 대조하도록 수정했습니다. (예: horse 검색 시 언어 '말'이 아닌 동물 '말' 추출)

Chain of Thought (CoT) 프롬프트: LLM이 뜻을 선택할 때 무작정 고르지 않고, reasoning 필드에 먼저 이유를 작성하도록 강제하여 AI의 환각과 성급한 매칭을 차단했습니다.

불용어 필터링: 기초사전에서 반환된 품사가 학습에 부적합한 '어미', '조사', '접사'일 경우 LLM 검증 이전에 원천적으로 탈락(Drop)시키는 로직을 추가했습니다.

4. STT(음성 인식) 엔진 최적화 및 평가 고도화
Demucs 선택적 스킵 (skip_demucs): 미디어 스캔 시에는 Demucs를 통해 배경음을 제거하지만, 마이크 입력 시에는 목소리 뭉개짐 현상을 방지하기 위해 Demucs 필터를 건너뛰도록 처리했습니다.

Whisper 신뢰도 반영: 사용자의 발음을 평가할 때 단순 물리적 오차뿐만 아니라 Whisper 모델의 인식 결과 신뢰도(Penalty Factor)를 추출하여 최종 점수에 곱연산으로 반영합니다.

5. 서버 디스크 용량 누수 방지 및 클린업
TTS 오디오 프리뷰, 발음 평가 파일 등 임시 생성되는 파일들로 인해 서버 디스크 용량이 꽉 차는 것을 방지하는 로직을 적용했습니다.

임시 파일 지연 삭제: 파일이 클라이언트로 전송된 직후 백그라운드에서 안전하게 임시 오디오 파일 및 폴더를 즉시 삭제합니다.

수동 청소 라우터 (DELETE /api/scan/cleanup): 관리자가 원할 때 언제든 static/tts/temp 폴더를 즉각적으로 비울 수 있는 전용 청소 API를 제공합니다.

6. 오디오 경로 형식 수정 및 통신 규격 안정화
절대 URL 반환: 프론트엔드에서 바로 재생할 수 있도록 TTS 파일 경로를 상대 경로(static/tts/...)에서 절대 URL(http://주소/static/tts/...) 형식으로 수정했습니다.

로그인 규격 복구: 테스트 과정에서 사용하던 form-data 방식을 클라이언트 연동에 가장 적합한 JSON Body 방식으로 안정화했습니다.

라우터 404 경로 해결: 앱과 서버 간의 API 호출 시 상대경로/절대경로 맵핑 불일치 문제를 명확한 prefix 적용을 통해 해결하여 통신 안정성을 확보했습니다.
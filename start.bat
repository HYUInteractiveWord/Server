@echo off
echo [1/3] Starting PostgreSQL container...
docker-compose up -d

echo [2/3] Activating virtual environment...
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)

echo [3/3] Starting FastAPI server...
uvicorn app.main:app --reload

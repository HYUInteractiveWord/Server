@echo off
chcp 949 > nul

echo ==================================================
echo [1/4] Starting PostgreSQL container
echo ==================================================
docker-compose up -d

if %errorlevel% neq 0 (
    echo.
    echo Docker 실행에 실패했습니다.
    echo 1. Docker Desktop 프로그램이 켜져 있는지 확인
    echo 2. 현재 폴더에 docker-compose.yml 파일이 있는지 확인
    pause
    exit /b
)
echo Waiting 3 seconds for Database to be ready
timeout /t 3 /nobreak > nul

echo.
echo ==================================================
echo [2/4] Checking Virtual Environment
echo ==================================================
if not exist .venv (
    echo Creating virtual environment
    python -m venv .venv
    call .venv\Scripts\activate
    
    echo Installing base dependencies
    pip install -r requirements.txt
    echo.
    echo Installing build tools
    pip install ninja
    
    echo.
    echo Initializing MSVC Environment
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    
    echo.
    echo Installing LLM Engine
    set "FORCE_CMAKE=1"
    set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CXX_FLAGS=/utf-8 -DCMAKE_C_FLAGS=/utf-8 -G Ninja"
    pip install --no-cache-dir llama-cpp-python[server]
) else (
    call .venv\Scripts\activate
)

echo.
echo ==================================================
echo [3/4] Starting Local LLM Server
echo ==================================================
for /f "tokens=1,2 delims==" %%A in (.env) do (
    if "%%A"=="LLM_MODEL_NAME" set "LLM_MODEL_NAME=%%B"
)
echo %LLM_MODEL_NAME%.gguf
rem 2. 하드코딩된 파일명 대신 %LLM_MODEL_NAME% 변수를 사용
start "Local LLM API Server" cmd /k "call .venv\Scripts\activate & set PYTHONIOENCODING=utf-8 & python -m llama_cpp.server --model ./models/%LLM_MODEL_NAME%.gguf --host 0.0.0.0 --port 8001 --n_ctx 4096 --n_gpu_layers -1"
timeout /t 3 /nobreak > nul

echo.
echo ==================================================
echo [4/4] Starting FastAPI server
echo ==================================================
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000
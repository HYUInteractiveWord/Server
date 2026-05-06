#!/bin/bash
# 서버 시작 스크립트 (Linux / EC2)
set -e

echo "=================================================="
echo "[1/4] PostgreSQL 컨테이너 시작"
echo "=================================================="
docker-compose up -d
echo "DB 준비 대기 중 (3초)..."
sleep 3

echo ""
echo "=================================================="
echo "[2/4] 가상환경 확인"
echo "=================================================="
if [ ! -d ".venv" ]; then
    echo "가상환경 생성 중..."
    python3.11 -m venv .venv
    source .venv/bin/activate

    echo "기본 패키지 설치 중..."
    pip install --upgrade pip
    pip install -r requirements.txt

    echo "llama-cpp-python (CUDA) 빌드 및 설치 중... (시간이 걸릴 수 있습니다)"
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir "llama-cpp-python[server]"
else
    source .venv/bin/activate

    python -c "import llama_cpp" 2>/dev/null || {
        echo "llama-cpp-python 미설치. CUDA 빌드로 설치 중..."
        CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir "llama-cpp-python[server]"
    }
fi

echo ""
echo "=================================================="
echo "[3/4] LLM 서버 시작 (포트 8001)"
echo "=================================================="
set -a
source .env
set +a

mkdir -p logs

nohup python -m llama_cpp.server \
    --model "./models/${LLM_MODEL_NAME}.gguf" \
    --host 127.0.0.1 \
    --port 8001 \
    --n_ctx 4096 \
    --n_gpu_layers -1 \
    > logs/llm_server.log 2>&1 &

LLM_PID=$!
echo "LLM 서버 PID: $LLM_PID"
echo $LLM_PID > logs/llm_server.pid
echo "LLM 서버 준비 대기 중 (10초)..."
sleep 10

echo ""
echo "=================================================="
echo "[4/4] FastAPI 서버 시작 (포트 8000)"
echo "=================================================="
uvicorn app.main:app --host 0.0.0.0 --port 8000

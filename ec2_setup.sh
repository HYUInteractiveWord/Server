#!/bin/bash
# EC2 초기 환경 세팅 스크립트 (최초 1회만 실행)
# 대상: Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Amazon Linux 2023)
set -e

echo "=================================================="
echo "[1/6] 시스템 패키지 업데이트"
echo "=================================================="
sudo dnf update -y
sudo dnf install -y git ffmpeg java-17-amazon-corretto cmake gcc gcc-c++ python3.11 python3.11-pip python3.11-devel

echo ""
echo "=================================================="
echo "[2/6] Docker 설치 및 시작"
echo "=================================================="
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

echo ""
echo "=================================================="
echo "[3/6] Docker Compose 설치"
echo "=================================================="
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo ""
echo "=================================================="
echo "[4/6] CUDA 확인"
echo "=================================================="
nvidia-smi
nvcc --version

echo ""
echo "=================================================="
echo "[5/6] 로그 디렉토리 생성"
echo "=================================================="
mkdir -p logs
mkdir -p models
mkdir -p app/static/tts

echo ""
echo "=================================================="
echo "[6/6] 완료"
echo "=================================================="
echo ""
echo "이제 다음 단계를 순서대로 실행하세요:"
echo "  1. cp .env.example .env  -> .env 파일 수정 (API 키 입력)"
echo "  2. bash download_model.sh  -> GGUF 모델 다운로드"
echo "  3. bash start.sh  -> 서버 시작"
echo ""
echo "주의: docker 그룹 적용을 위해 재접속(SSH logout/login) 필요할 수 있습니다."

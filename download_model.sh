#!/bin/bash
# HuggingFace에서 GGUF 모델 다운로드
# 실행 전: .env 파일의 LLM_MODEL_NAME 확인
set -e

# .env에서 모델명 읽기
set -a
source .env
set +a

# -----------------------------------------------
# HuggingFace 레포지토리 정보 (필요시 수정)
# -----------------------------------------------
HF_REPO="unsloth/gemma-4-E4B-it-GGUF"
MODEL_FILE="${LLM_MODEL_NAME}.gguf"
# -----------------------------------------------

DEST="./models/${MODEL_FILE}"

if [ -f "$DEST" ]; then
    echo "모델 파일이 이미 존재합니다: $DEST"
    exit 0
fi

echo "모델 다운로드 시작: ${MODEL_FILE}"
echo "레포: ${HF_REPO}"
echo ""

pip show huggingface_hub > /dev/null 2>&1 || pip install huggingface_hub

huggingface-cli download "${HF_REPO}" "${MODEL_FILE}" \
    --local-dir ./models \
    --local-dir-use-symlinks False

echo ""
echo "다운로드 완료: $DEST"
ls -lh "$DEST"

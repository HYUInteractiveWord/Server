from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import os
import numpy as np
import tempfile
import math
from pathlib import Path

from app.db import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.models.pronunciation_record import PronunciationRecord
from app.schemas.pronunciation import PronunciationResponse
from app.services.gamification import calculate_xp_gain
from app.services.evaluation import PronunciationEvaluator
from app.services.pipeline import _get_whisper_model
from app.core.config import settings
from app.services.pronunciation.service import save_all_pitch_graphs

router = APIRouter(prefix="/pronunciation", tags=["pronunciation"])

def make_serializable(obj):
    if isinstance(obj, (np.float32, np.float64, float)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
        
    if isinstance(obj, np.ndarray):
        arr = obj.copy()
        arr[~np.isfinite(arr)] = 0.0
        return arr.tolist()
        
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]
        
    return obj


@router.post("/submit")
async def submit_pronunciation(
    word_card_id: int = Form(...),
    korean_word: str = Form(...),
    tts_audio_path: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    audio_bytes = await file.read()
    
    clean_path = tts_audio_path.lstrip("/\\")
    full_tts_path = Path.cwd() / clean_path
    
    if not full_tts_path.exists():
        filename = os.path.basename(clean_path)
        search_root = Path("static/tts")
        found = False
        for p in search_root.rglob(filename):
            full_tts_path = p
            found = True
            break
            
        if not found:
            raise HTTPException(status_code=404, detail="서버에서 기준 TTS 파일을 찾을 수 없습니다.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_user:
        tmp_user.write(audio_bytes)
        user_audio_path = tmp_user.name

    try:
        whisper_model = _get_whisper_model(settings.WHISPER_MODEL)
        evaluator = PronunciationEvaluator(whisper_model=whisper_model)
        
        eval_result = evaluator.evaluate(
            target_word=korean_word,
            tts_audio_path=str(full_tts_path),
            user_audio_bytes=audio_bytes
        )
        
        final_score = eval_result["final_score"]
        analysis_data = eval_result["analysis_data"]
        
        user_pitch = make_serializable(analysis_data["plot_data"]["f0_usr"])
        ref_pitch = make_serializable(analysis_data["plot_data"]["f0_ref"])
        dtw_dist = make_serializable(analysis_data["pronunciation_detail"]["normalized_distance"])

        record = PronunciationRecord(
            user_id=current_user.id,
            word_card_id=word_card_id,
            score=final_score,
            user_pitch_data=user_pitch,
            reference_pitch_data=ref_pitch,
            dtw_distance=dtw_dist,
        )
        db.add(record)

        xp_gained = calculate_xp_gain(final_score, is_new_best=True)
        current_user.xp += xp_gained

        db.commit()
        db.refresh(record)

        graph_output_dir = "static/tts/graphs"
        graph_data = save_all_pitch_graphs(
            tts_path=str(full_tts_path),
            user_path=user_audio_path,
            output_dir=graph_output_dir
        )

        formatted_graphs = {}
        if graph_data and "graph_paths" in graph_data:
            for key, val in graph_data["graph_paths"].items():
                formatted_graphs[key] = str(val)

        detailed_scores = analysis_data["scores"]

        plot_data = analysis_data["plot_data"]
        raw_graph_json = {
            "tts_time": make_serializable(plot_data["t_ref"]),
            "tts_pitch": make_serializable(plot_data["f0_ref"]),
            "user_time": make_serializable(plot_data["t_usr"]),
            "user_pitch": make_serializable(plot_data["f0_usr"]),
        }


        return {
            "record_id": record.id,
            "score": float(final_score),
            "is_new_best": True,
            "xp_gained": int(xp_gained),
            "word_card_level": 0,
            "graphs": graph_data["graph_paths"],

            "details": {
                "pronunciation": float(detailed_scores["pronunciation_score"]),
                "formant": float(detailed_scores["formant_score"]),
                "pitch": float(detailed_scores["pitch_score"]),
                "timing": float(detailed_scores["duration_score"]),
                "is_intensity_good": bool(detailed_scores["intensity_pass"])
            },

            "raw_graph_data": raw_graph_json
        }
    
    finally:
        if os.path.exists(user_audio_path):
            os.remove(user_audio_path)


@router.get("/{word_card_id}/history")
def get_pronunciation_history(
    word_card_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = db.query(PronunciationRecord).filter(
        PronunciationRecord.word_card_id == word_card_id,
        PronunciationRecord.user_id == current_user.id,
    ).order_by(PronunciationRecord.recorded_at.desc()).all()
    return records


@router.post("/evaluate_test")
async def evaluate_pronunciation_test(
    target_word: str = Form(...),
    tts_audio_path: str = Form(...),
    file: UploadFile = File(...)
):
    audio_bytes = await file.read()
    clean_path = tts_audio_path.lstrip("/\\")
    target_path = Path.cwd() / clean_path
    
    if not target_path.exists():
        filename = os.path.basename(clean_path)
        search_root = Path("static/tts/test_user")
        found = False
        for p in search_root.rglob(filename):
            target_path = p
            found = True
            break
        
        if not found:
            raise HTTPException(
                status_code=404, 
                detail=f"서버에서 파일을 찾을 수 없습니다. 경로를 확인하세요: {target_path}"
            )

    full_tts_path = target_path

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_user:
        tmp_user.write(audio_bytes)
        user_audio_path = tmp_user.name

    try:
        whisper_model = _get_whisper_model(settings.WHISPER_MODEL)
        evaluator = PronunciationEvaluator(whisper_model=whisper_model)
        
        eval_result = evaluator.evaluate(
            target_word=target_word,
            tts_audio_path=str(full_tts_path),
            user_audio_bytes=audio_bytes
        )

        graph_output_dir = "static/tts/graphs"
        graph_data = save_all_pitch_graphs(
            tts_path=str(full_tts_path),
            user_path=user_audio_path,
            output_dir=graph_output_dir
        )

        detailed_scores = eval_result["analysis_data"]["scores"]
        plot_data = eval_result["analysis_data"]["plot_data"]
        
        raw_graph_json = {
            "tts_time": make_serializable(plot_data["t_ref"]),
            "tts_pitch": make_serializable(plot_data["f0_ref"]),
            "user_time": make_serializable(plot_data["t_usr"]),
            "user_pitch": make_serializable(plot_data["f0_usr"]),
        }

        return make_serializable({
            "status": "success",
            "target_word": target_word,
            "evaluation": eval_result,
            "graphs": graph_data["graph_paths"],

            "details": {
                "pronunciation": float(detailed_scores["pronunciation_score"]),
                "formant": float(detailed_scores["formant_score"]),
                "pitch": float(detailed_scores["pitch_score"]),
                "timing": float(detailed_scores["duration_score"]),
                "is_intensity_good": bool(detailed_scores["intensity_pass"])
            },
            "raw_graph_data": raw_graph_json

        })
    
    finally:
        if os.path.exists(user_audio_path):
            os.remove(user_audio_path)
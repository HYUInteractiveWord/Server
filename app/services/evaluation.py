import os
import tempfile
import whisper
import re
import numpy as np

# 팀원이 작성한 기존 분석 모듈 임포트
from app.services.pronunciation.service import analyze_pronunciation 

class PronunciationEvaluator:
    def __init__(self, whisper_model: whisper.Whisper):
        self.whisper_model = whisper_model

    def _get_whisper_penalty_factor(self, user_audio_path: str, target_word: str) -> float:
        """
        Whisper 인식 결과를 바탕으로 페널티 비율(0.0 ~ 1.0) 계산
        - 타겟 단어가 정확히 포함됨 (1순위): 감점 없음 (1.0)
        - 타겟 단어 누락 (다른 단어로 인식됨): 인식된 발화의 평균 신뢰도만큼 감점 곱연산
        """
        try:
            result = self.whisper_model.transcribe(
                user_audio_path, language="ko", word_timestamps=True
            )
            
            recognized_text = result.get("text", "")
            
            clean_target = re.sub(r'[^가-힣a-zA-Z0-9]', '', target_word)
            clean_recognized = re.sub(r'[^가-힣a-zA-Z0-9]', '', recognized_text)
            
            # 1순위로 인식된 경우 감점 없음
            if clean_target in clean_recognized:
                return 1.0
            
            # 다른 단어로 인식된 경우 해당 구간 모델의 평균 Probability 산출
            total_prob = 0.0
            word_count = 0
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    total_prob += word_info.get("probability", 0.0)
                    word_count += 1
                    
            avg_prob = (total_prob / word_count) if word_count > 0 else 0.5
            
            # 산출된 신뢰도를 반환하여 최종 점수에 곱함
            return float(avg_prob)
        except Exception as e:
            print(f"[Whisper Error] 신뢰도 추출 실패: {e}")
            return 1.0

    def evaluate(self, target_word: str, tts_audio_path: str, user_audio_bytes: bytes) -> dict:
        """물리적 유사도 점수와 AI 신뢰도 감점을 결합하여 최종 점수 반환"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_user:
            tmp_user.write(user_audio_bytes)
            user_audio_path = tmp_user.name

        try:
            analysis_result = analyze_pronunciation(tts_audio_path, user_audio_path)
            physical_score = analysis_result["scores"]["final_score"]
            
            penalty_factor = self._get_whisper_penalty_factor(user_audio_path, target_word)
            
            final_score = physical_score * penalty_factor
            
            return {
                "target_word": target_word,
                "physical_score": round(physical_score, 2),
                "penalty_factor": round(penalty_factor, 2),
                "final_score": round(final_score, 2),
                "analysis_data": analysis_result
            }
        finally:
            if os.path.exists(user_audio_path):
                os.remove(user_audio_path)
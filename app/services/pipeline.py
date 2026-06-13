import os
import uuid
import subprocess
import tempfile
import json
import asyncio
import requests
import re
import html
import xml.etree.ElementTree as ET
import whisper
import edge_tts
import torch
import time
import sys

from konlpy.tag import Okt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Audio Processing Pipeline (Demucs + Whisper)
_model_cache: dict[str, whisper.Whisper] = {}

def _get_whisper_model(model_size: str) -> whisper.Whisper:
    if model_size not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_cache[model_size] = whisper.load_model(model_size, device=device)
    return _model_cache[model_size]

def _run_demucs(input_path: str, out_dir: str) -> str | None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmd = [sys.executable, "-m", "demucs", "-d", device, "--two-stems=vocals", "--out", out_dir, input_path]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        return None
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = build_path(out_dir, "htdemucs", base_name, "vocals.wav")
    return vocals_path if os.path.exists(vocals_path) else None

def _run_whisper(audio_path: str, model_size: str) -> str:
    model = _get_whisper_model(model_size)
    result = model.transcribe(audio_path, language="ko")
    return result["text"]

def extract_text_from_audio(audio_bytes: bytes, ffmpeg_bin: str, whisper_model_size: str) -> str:
    """오디오 바이트를 받아 Demucs → Whisper 실행 후 Raw 텍스트 반환"""
    if ffmpeg_bin:
        os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = build_path(tmpdir, f"{uuid.uuid4().hex}.wav")
        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        vocals_path = _run_demucs(input_path, tmpdir)
        if vocals_path is None:
            vocals_path = input_path  # Demucs 실패 시 원본으로 fallback

        return _run_whisper(vocals_path, whisper_model_size)

def build_path(base, *parts):
    path = os.path.join(base, *parts)
    return path.replace("\\", "/")


# 2. NLP & LLM Pipeline
class KoreanLearningPipeline:
    def __init__(self, term_api_key: str, dict_api_key: str, model_name: str = "gemma-4"):
        self.term_api_key = term_api_key
        self.dict_api_key = dict_api_key
        
        self.llm = ChatOpenAI(
            base_url="http://localhost:8001/v1",
            api_key="empty", 
            model=model_name,
            temperature=0.0 
        )
        self.okt = Okt()
        self.json_parser = JsonOutputParser()
        
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def _get_lang_str(self, target_language: str) -> str:
        """언어 코드를 LLM 프롬프트용 지시어로 변환"""
        lang_map = {
            "en": "English",
            "ru": "Russian (русский язык)"
        }
        return lang_map.get(target_language.lower(), "English (영어)")

    async def correct_stt_text(self, text: str) -> str:
        """1. STT 텍스트 후보정 (비동기 처리) - 프롬프트 및 후처리 강화"""
        if not text.strip():
            return ""

        prompt = PromptTemplate.from_template(
            "당신은 한국어 음성 인식(STT) 오탈자 교정기입니다. 문맥을 추론하여 아예 다른 단어로 창조하지 마시고, 오직 '발음이 비슷하게 잘못 적힌 글자'만 올바른 맞춤법으로 고치세요.\n\n"
            "[예시]\n"
            "입력: 이 아름다운 해바라기 죠화는 어디서 줴작한 거\n"
            "출력: 이 아름다운 해바라기 조화는 어디서 제작한 거\n\n"
            "[절대 규칙]\n"
            "1. 원본의 의미를 임의로 유추하여 완전히 다른 뜻의 단어로 바꾸면 절대 안 됩니다.\n"
            "2. 문장의 원래 의미와 시제를 100% 원본과 동일하게 유지하세요.\n"
            "3. '원본:', '보정본:', '출력:' 등의 설명이나 접두사를 절대 붙이지 마세요. 오직 교정된 텍스트 딱 하나만 출력하세요.\n\n"
            "입력: {text}\n출력:"
        )
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"text": text})
            result = response.content.strip()
            
            for prefix in ["보정본:", "출력:", "원본:"]:
                if prefix in result:
                    result = result.split(prefix)[-1].strip()
            
            return result.replace('"', '').replace("'", "")
        except Exception as e:
            print(f"  [Error] STT correction failed: {e}", flush=True)
            return text


    
    async def extract_core_vocabulary(self, text: str):
        """2. 핵심 어휘 추출 (비동기 처리 및 JSON 강제 프롬프트 강화)"""
        prompt = PromptTemplate.from_template(
            "당신은 한국어 교육 전문가입니다. 텍스트에서 외국인 학습자용 '핵심 어휘(명사, 동사, 형용사)'만 추출하세요.\n"
            "- 대명사, 의존명사 절대 제외.\n"
            "- 동사/형용사는 사전형(원형)으로 변환 (예: '샀다' -> '사다').\n\n"
            "텍스트: {text}\n\n"
            "반드시 아래와 같은 JSON 배열(문자열 리스트) 형식으로만 응답하고, 다른 사족은 절대 붙이지 마세요.\n"
            '["단어1", "단어2"]'
        )
        
        chain = prompt | self.llm 
        
        try:
            response = await chain.ainvoke({"text": text})
            llm_raw_output = response.content.strip() 
            
            cleaned_output = re.sub(r'```json\n?|```', '', llm_raw_output).strip()
            
            try:
                extracted_words = json.loads(cleaned_output)
            except Exception:
                extracted_words = re.findall(r'[가-힣]+', cleaned_output)
            return extracted_words, llm_raw_output
            
        except Exception as e:
            print(f"  [Error] Vocab extraction failed: {e}", flush=True)
            return [], f"Error: {str(e)}"

    async def fetch_basic_dict_data(self, word: str, expected_meaning: str = None) -> dict:
        """기초사전 API에서 다의어 포함 뜻풀이 수집"""
        url = "https://krdict.korean.go.kr/api/search"
        params = {
            "key": self.dict_api_key, 
            "q": word, 
            "part": "word", 
            "sort": "popular",  
            "advanced": "y", 
            "method": "exact",
            "num": 10           
        }
        print(f"  [Dict API] '{word}' 기초사전 검색 중...", flush=True)
        
        def _sync_request():
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.text

        try:
            response_text = await asyncio.to_thread(_sync_request)
            
            root = ET.fromstring(response_text)
            total = int(root.findtext('total', '0'))
            
            if total > 0:
                items = root.findall('.//item')[:10]
                senses = []
                for item in items:
                    pos = item.findtext('pos', '품사 없음')
                    # 한 item 내에 sense가 여러 개 있을 수 있으므로 모두 탐색
                    for sense in item.findall('sense'):
                        definition = sense.findtext('definition', '정의 없음')
                        clean_def = self._clean_text(definition)
                        
                        if not any(s['definition'] == clean_def for s in senses):
                            senses.append({"pos": pos, "definition": clean_def})
                            
                print(f"  [Dict API] '{word}' 뜻 {len(senses)}개 발견 완료!", flush=True)
                return {"status": "success", "word": word, "senses": senses}
            
            print(f"  [Dict API] '{word}' 사전에 등록되지 않은 단어입니다.", flush=True)
            return {"status": "fail", "word": word}
            
        except Exception as e:
            print(f"  [Error] fetch_basic_dict_data failed for '{word}': {e}", flush=True)
            return {"status": "error", "word": word}


    async def select_best_definition(self, word: str, senses: list, context_text: str) -> dict | None:
        """다의어 중 원본 외국어 뜻(문맥)에 가장 부합하는 뜻 선택 (일치하는게 없으면 None 반환)"""
        prompt = PromptTemplate.from_template(
            "당신은 이중언어 번역 및 문맥 분석기입니다. 사용자가 찾고자 하는 단어의 핵심 의미는 다음과 같습니다.\n"
            "[목표 의미 및 문맥]: {context_text}\n\n"
            "아래는 한국어 단어 '{word}'의 사전적 뜻풀이 후보들입니다.\n"
            "[후보]\n{candidates}\n\n"
            "위 후보 중 [목표 의미]와 가장 정확하게 일치하는 뜻의 번호(index)를 찾으세요.\n"
            "중요: 만약 후보 중에 [목표 의미]와 일치하거나 유사한 뜻이 **단 하나도 없다면**, 억지로 고르지 말고 반드시 -1을 반환하세요.\n"
            "반드시 아래의 JSON 형식으로만 응답하세요.\n"
            "{{\n  \"best_index\": 0\n}}"
        )
        candidates_str = "".join([f"{i}. [{s['pos']}] {s['definition']}\n" for i, s in enumerate(senses)])
        chain = prompt | self.llm | self.json_parser
        
        try:
            result = await chain.ainvoke({"word": word, "context_text": context_text, "candidates": candidates_str})
            # 기본값을 0에서 -1로 변경
            best_idx = result.get("best_index", -1) 
            
            # 일치하는 뜻이 없어 -1을 반환했거나, 유효하지 않은 인덱스면 None 반환
            if best_idx == -1 or not (0 <= best_idx < len(senses)):
                return None
                
            return senses[best_idx]
        except Exception as e:
            print(f"  [Error] select_best_definition failed: {e}", flush=True)
            return None
    async def fetch_basic_dict_data(self, word: str, expected_meaning: str = None) -> dict:
        """기초사전 API에서 다의어 포함 뜻풀이 수집"""
        url = "https://krdict.korean.go.kr/api/search"
        params = {
            "key": self.dict_api_key, 
            "q": word, 
            "part": "word", 
            "sort": "popular",  # 'dict' 대신 'popular'를 써서 대중적인 뜻
            "advanced": "y", 
            "method": "exact",
            "num": 10           # 다의어가 잘리지 않게 후보군을 10개로
        }
        print(f"  [Dict API] '{word}' 기초사전 검색 중...", flush=True)
        
        def _sync_request():
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.text

        try:
            response_text = await asyncio.to_thread(_sync_request)
            
            root = ET.fromstring(response_text)
            total = int(root.findtext('total', '0'))
            
            if total > 0:
                items = root.findall('.//item')[:10]
                senses = []
                for item in items:
                    pos = item.findtext('pos', '품사 없음')
                    # 한 item 내에 sense가 여러 개 있을 수 있으므로 모두 탐색
                    for sense in item.findall('sense'):
                        definition = sense.findtext('definition', '정의 없음')
                        clean_def = self._clean_text(definition)
                        
                        if not any(s['definition'] == clean_def for s in senses):
                            senses.append({"pos": pos, "definition": clean_def})
                            
                print(f"  [Dict API] '{word}' 뜻 {len(senses)}개 발견 완료!", flush=True)
                return {"status": "success", "word": word, "senses": senses}
            
            print(f"  [Dict API] '{word}' 사전에 등록되지 않은 단어입니다.", flush=True)
            return {"status": "fail", "word": word}
            
        except Exception as e:
            print(f"  [Error] fetch_basic_dict_data failed for '{word}': {e}", flush=True)
            return {"status": "error", "word": word}


    async def select_best_definition(self, word: str, senses: list, context_text: str) -> dict | None:
        """다의어 중 원본 외국어 뜻(문맥)에 가장 부합하는 뜻 선택 (일치하는게 없으면 None 반환)"""
        
        # 💡 [핵심] LLM이 뜻풀이를 안 읽고 대충 true를 던지는 것을 막기 위해 'reasoning(추론)' 과정을 강제합니다.
        prompt = PromptTemplate.from_template(
            "당신은 이중언어 번역 및 문맥 분석기입니다. 사용자가 찾고자 하는 단어의 핵심 의미는 다음과 같습니다.\n"
            "[목표 의미 및 문맥]: {context_text}\n\n"
            "아래는 한국어 단어 '{word}'의 사전적 뜻풀이 후보들입니다.\n"
            "[후보]\n{candidates}\n\n"
            "평가 지침:\n"
            "1. 위 후보들의 '뜻풀이(definition)'를 주의 깊게 읽고, [목표 의미]와 정확하게 일치하는지 비교하여 'reasoning'에 먼저 그 이유를 작성하세요.\n"
            "2. 단어의 발음이나 한자가 같더라도, 제시된 뜻풀이가 [목표 의미]와 다르면 절대 고르면 안 됩니다.\n"
            "3. 일치하는 뜻풀이가 있다면 'is_match'를 true로 하고, 해당 번호(index)를 'best_index'에 적으세요.\n"
            "4. 일치하는 뜻풀이가 단 하나도 없다면 'is_match'를 반드시 false로 하고, 'best_index'는 -1로 적으세요.\n\n"
            "반드시 아래의 JSON 형식으로 응답하세요:\n"
            "{{\n  \"reasoning\": \"뜻풀이 0번은 장애물을 뜻하므로 horse와 다릅니다.\",\n  \"is_match\": false,\n  \"best_index\": -1\n}}"
        )
        candidates_str = "".join([f"{i}. [{s['pos']}] {s['definition']}\n" for i, s in enumerate(senses)])
        chain = prompt | self.llm | self.json_parser
        
        try:
            result = await chain.ainvoke({"word": word, "context_text": context_text, "candidates": candidates_str})
            
            # 💡 서버 로그에서 LLM이 무슨 생각을 했는지 직접 확인할 수 있습니다.
            print(f"  [Dict Filter] '{word}' 판단 이유: {result.get('reasoning', '')}", flush=True)
            
            is_match = result.get("is_match", False)
            if not is_match:
                print(f"  [Dict Filter] '{word}' 최종 탈락 (is_match=False)", flush=True)
                return None
                
            best_idx = result.get("best_index", -1) 
            
            if 0 <= best_idx < len(senses):
                return senses[best_idx]
            else:
                return None
                
        except Exception as e:
            print(f"  [Error] select_best_definition failed: {e}", flush=True)
            return None

    async def filter_with_dict(self, extracted_words: list, context_text: str, expected_meaning: str = None) -> dict:
        """3. 기초사전 검증 (어미, 조사 등 학습에 부적합한 품사 원천 차단)"""
        valid_candidates = {}
        unique_words = list(set(extracted_words))
        
        excluded_pos = ["품사 없음", "어미", "조사", "접사", "접두사", "접미사"]
        
        for word in unique_words:
            dict_info = await self.fetch_basic_dict_data(word, expected_meaning=expected_meaning)
            
            if dict_info.get("status") == "success":
                valid_senses = [
                    s for s in dict_info.get("senses", []) 
                    if s["pos"] and s["pos"] not in excluded_pos
                ]
                
                if not valid_senses:
                    print(f"  [Dict Filter] 탈락: '{word}' (학습에 부적합한 품사만 존재함)", flush=True)
                    continue
                
                best_sense = await self.select_best_definition(word, valid_senses, context_text)
                
                if best_sense is not None:
                    valid_candidates[word] = best_sense
                else:
                    print(f"  [Dict Filter] 탈락: '{word}' (사전에 목표 의미와 일치하는 뜻이 없습니다)", flush=True)
                    
            await asyncio.sleep(0.2)
            
        return valid_candidates


    async def fetch_on_term_category(self, original_word: str, target_definition: str = "") -> str:
        """온용어 세부 분류 태그 검색 (의미 기반 스마트 매칭)"""
        nouns = self.okt.nouns(original_word)
        search_keyword = nouns[0] if nouns else original_word
        
        url = "https://kli.korean.go.kr/term/api/search.do"
        params = {
            "key": self.term_api_key, 
            "apiSearchWord": search_keyword, 
            "start": 1, 
            "num": 10
        }
        
        def _sync_request():
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, params=params, headers=headers, timeout=10, verify=False)
            response.raise_for_status()
            return response.json()

        try:
            data = await asyncio.to_thread(_sync_request)
            channel = data.get("channel", {})
            
            if int(channel.get("total", 0)) == 0:
                return ""
                
            if int(channel.get("total", 0)) > 0:
                return_objects = channel.get("return_object", [])
                if return_objects and return_objects[0].get("returnCode") == 1:
                    result_list = return_objects[0].get("resultlist", [])
                    if result_list:
                        
                        if target_definition and len(result_list) > 1:
                            candidates_str = ""
                            for i, item in enumerate(result_list):
                                cat_main = item.get("category_main", "")
                                cat_sub = item.get("category_sub", "")
                                item_def = item.get("definition", "정의 없음")
                                candidates_str += f"{i}. [{cat_main} - {cat_sub}] {item_def}\n"
                            
                            prompt = PromptTemplate.from_template(
                                "당신은 한국어 의미 분석기입니다. 단어 '{word}'에 대해 우리가 찾고자 하는 [목표 뜻]은 다음과 같습니다.\n"
                                "[목표 뜻]: {target_definition}\n\n"
                                "[온용어 사전 후보]\n"
                                "{candidates}\n\n"
                                "위 후보 중 [목표 뜻]과 의미가 가장 일치하는 번호(index)를 찾으세요. JSON 형식으로 응답하세요.\n"
                                "{{\n  \"best_index\": 0\n}}"
                            )
                            chain = prompt | self.llm | self.json_parser
                            try:
                                result = await chain.ainvoke({
                                    "word": original_word, 
                                    "target_definition": target_definition, 
                                    "candidates": candidates_str
                                })
                                best_idx = result.get("best_index", 0)
                                target_data = result_list[best_idx] if 0 <= best_idx < len(result_list) else result_list[0]
                            except Exception:
                                target_data = result_list[0]
                        else:
                            target_data = result_list[0]
                            
                        category_main = target_data.get("category_main", "")
                        category_sub = target_data.get("category_sub", "")
                        return f"{category_main} - {category_sub}".strip(" -")
            return ""
        except Exception as e:
            print(f"  온용어 검색 오류: {e}", flush=True)
            return ""

    async def process_with_llm(self, word_raw: str, definition: str, pos: str, target_language: str = "en") -> dict:
        """LLM 번역 및 품사별 문법 특화 예문 생성 (타겟 언어 원어 지시어 적용)"""
        target_lang_str = self._get_lang_str(target_language)
        lang_code = target_language.strip().lower().split("-")[0]
        if lang_code == "ru":
            romanization_instruction = (
                '"Напишите произношение ТОЛЬКО ОДНОГО главного слова, указанного в поле [단어], русскими буквами (кириллицей). '
                'КАТЕГОРИЧЕСКИ ЗАПРЕЩАЕТСЯ писать произношение сгенерированных предложений! '
                'Только одно слово. Без латиницы, без хангыля, без скобок. Пример: иджа"'
            )
            trans_def_instruction = (
                '"[Точный перевод в одно слово] / [Простое объяснение на русском языке]. '
                'Не используйте корейский язык в этом поле. Обязательно разделяйте перевод и объяснение косой чертой (/). '
                'Пример: стул / предмет мебели с приподнятой поверхностью, опирающийся на ножки"'
            )
        else:
            romanization_instruction = (
                '"Write the pronunciation of ONLY the single main word indicated in [단어] using the Latin alphabet. '
                'ABSOLUTELY DO NOT write the pronunciation of the generated example sentences! '
                'Just the one word. No Hangul, no parentheses. Example: uija"'
            )
            trans_def_instruction = (
                '"[Exact 1:1 translation word] / [Simple explanation sentence in English]. '
                'Do not use Korean at all in this field. You must separate the translation and explanation with a slash (/). '
                'Example: chair / a piece of furniture with a raised surface supported by legs"'
            )
        
        # 1. 동사 분기 프롬프트 (과거, 현재, 미래 시제 반영)
        if "동사" in pos or "Verb" in pos:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 단어를 사용하여 초급 한국어 학습자를 위한 쉽고 명확한 예문 3개를 생성하되, 반드시 아래 시제 조건에 맞춰 빌드하세요.\n"
                "- 첫 번째 예문: 해당 동사의 '과거 시제' 형태가 들어간 예문\n"
                "- 두 번째 예문: 해당 동사의 '현재 시제' 형태가 들어간 예문\n"
                "- 세 번째 예문: 해당 동사의 '미래 시제(~을 것이다, ~겠습니다 등)' 형태가 들어간 예문\n\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                f'  "translated_definition": {trans_def_instruction},\n'
                '  "easy_examples": [\n'
                '    {{"korean": "과거 시제 반영 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "현재 시제 반영 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "미래 시제 반영 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                f'  "romanization": {romanization_instruction}\n'
                "}}"
            )
            
        # 2. 명사 분기 프롬프트 (주어, 서술어, 목적어 위치 구조체 반영)
        elif "명사" in pos or "Noun" in pos:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 명사를 사용하여 초급 한국어 학습자를 위한 쉽고 명확한 예문 3개를 생성하되, 문장 내 성분 및 위치 규칙에 맞추어 빌드하세요.\n"
                "- 첫 번째 예문: 해당 명사가 문장 내에서 '주어(이/가, 은/는 결합)'로 사용된 예문\n"
                "- 두 번째 예문: 해당 명사가 문장 내에서 '서술어(이체/이다 결합 또는 보어 자리)'로 사용된 예문\n"
                "- 세 번째 예문: 해당 명사가 문장 내에서 '목적어(을/를 결합)'로 사용된 예문\n\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                f'  "translated_definition": {trans_def_instruction},\n'
                '  "easy_examples": [\n'
                '    {{"korean": "명사가 주어로 쓰인 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "명사가 서술어로 쓰인 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "명사가 목적어로 쓰인 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                f'  "romanization": {romanization_instruction}\n'
                "}}"
            )
            
        # 3. 형용사 분기 프롬프트 (수식형과 문장 성분 배치 결합)
        elif "형용사" in pos or "Adjective" in pos:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 형용사를 사용하여 초급 한국어 학습자를 위한 쉽고 명확한 예문 3개를 생성하되, 문장 내 결합 위치에 따라 다양하게 배치하세요.\n"
                "- 첫 번째 예문: 해당 형용사가 명사 앞에서 명사를 직접 수식하는 '관형사형/수식어 위치(예: 예쁜 꽃)'로 쓰인 예문\n"
                "- 두 번째 예문: 해당 형용사가 문장의 맨 끝에서 종결어미와 결합하여 '기본 서술어 위치(예: 꽃이 예쁘다)'로 쓰인 예문\n"
                "- 세 번째 예문: 해당 형용사가 문장 중간에서 연결어미나 부사형 어미 등과 결합하여 '다양한 변형 구조 위치(예: 예쁘게 자라다, 예쁘고 좋다)'로 쓰인 예문\n\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                f'  "translated_definition": {trans_def_instruction},\n'
                '  "easy_examples": [\n'
                '    {{"korean": "명사 수식형 구조의 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "문장 끝 서술형 구조의 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "연결 및 부사 구조적 변형 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                f'  "romanization": {romanization_instruction}\n'
                "}}"
            )
            
        # 4. 부사 분기 프롬프트
        elif "부사" in pos or "Adverb" in pos:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 부사를 사용하여 초급 한국어 학습자를 위한 쉽고 명확한 예문 3개를 생성하되, 수식하는 대상과 문장 내 위치에 따라 다양하게 배치하세요.\n"
                "- 첫 번째 예문: 해당 부사가 '동사' 바로 앞에서 동작의 상태나 정도를 꾸며주는 예문 (예: '빨리' 걷다, '잘' 먹다)\n"
                "- 두 번째 예문: 해당 부사가 '형용사'나 '다른 부사' 앞에서 그 정도를 강조하는 예문 (예: '아주' 예쁘다, '너무' 빨리)\n"
                "- 세 번째 예문: 문장 맨 앞이나 중간에서 문맥 전체의 분위기를 전환하거나 강조하는 예문\n\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                f'  "translated_definition": {trans_def_instruction},\n'
                '  "easy_examples": [\n'
                '    {{"korean": "동사 수식형 예문", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "형용사/부사 강조형 예문", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "문장 전체/분위기 수식형 예문", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                f'  "romanization": {romanization_instruction}\n'
                "}}"
            )
            
        # 5. Fallback
        else:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 단어를 문맥 내에 배치하여 결합 형태가 서로 다른 유용하고 명확한 예문 3개를 생성하세요.\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                f'  "translated_definition": {trans_def_instruction},\n'
                '  "easy_examples": [\n'
                '    {{"korean": "활용 예문 1", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "활용 예문 2", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "활용 예문 3", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                f'  "romanization": {romanization_instruction}\n'
                "}}"
            )

        prompt = PromptTemplate.from_template(prompt_text)
        chain = prompt | self.llm | self.json_parser
        try:
            return await chain.ainvoke({
                "word_raw": word_raw, 
                "definition": definition, 
                "pos": pos,
                "target_lang_str": target_lang_str
            })
        except Exception as e:
            print(f"  [Error] process_with_llm failed: {e}", flush=True)
            return {"translated_definition": "", "easy_examples": [], "romanization": ""}
    async def generate_tts(self, text: str, output_path: str, lang: str = "ko"):
        if not text: return
        clean_text = text.replace("/", ", ")
        clean_text = re.sub(r'[*_~\[\]\(\)<>]', '', clean_text) 
        clean_text = clean_text.strip()
        
        if not clean_text: return
        voice_map = {
            "ko": "ko-KR-SunHiNeural",
            "ru": "ru-RU-SvetlanaNeural",
            "en": "en-US-AriaNeural"
        }
        voice = voice_map.get(lang, "ko-KR-SunHiNeural")
        communicate = edge_tts.Communicate(clean_text, voice)
        await communicate.save(output_path)

    async def phase1_analyze(self, raw_stt_text: str) -> dict:
        """Phase 1: 텍스트 수신 -> 후보정 -> 단어 추출 -> 검증 -> 후보군 반환"""
        t_start = time.time()
        
        corrected_text = await self.correct_stt_text(raw_stt_text)
        print(f"  [Log] STT Correction Time: {time.time() - t_start:.2f}s", flush=True)
        
        t_step = time.time()
        extracted_words, llm_raw_output = await self.extract_core_vocabulary(corrected_text)
        print(f"  [Log] Vocab Extraction Time: {time.time() - t_step:.2f}s", flush=True)
        
        t_step = time.time()
        valid_candidates = await self.filter_with_dict(extracted_words, corrected_text)
        print(f"  [Log] Dict Filtering Time: {time.time() - t_step:.2f}s", flush=True)
        
        print(f"  [Log] Phase 1 Total: {time.time() - t_start:.2f}s", flush=True)
        return {
            "raw_text": raw_stt_text,
            "corrected_text": corrected_text,
            "llm_raw_output": llm_raw_output,    
            "extracted_words": extracted_words,   
            "candidates": valid_candidates
        }

    async def search_dictionary_candidates(self, query: str, source_lang: str = "러시아어 또는 영어") -> dict:
        """사전 검색: 외국어 -> 한국어 단어 후보 추출 및 사전 검증"""
        print(f"\n[Dict Search] '{query}' ({source_lang}) 한국어 단어 매칭 중...", flush=True)
        
        prompt = PromptTemplate.from_template(
            "당신은 한국어 교육 전문가입니다. 학습자가 입력한 {source_lang} 단어/문장 '{query}'에 해당하는 "
            "가장 자연스러운 한국어 기초 어휘(명사, 동사/형용사는 반드시 사전형) 1~3개를 추천해주세요.\n\n"
            "반드시 아래의 JSON 배열(문자열 리스트) 형식으로만 응답하세요.\n"
            '["단어1", "단어2"]'
        )
        chain = prompt | self.llm 
        
        try:
            response = await chain.ainvoke({"source_lang": source_lang, "query": query})
            llm_raw_output = response.content.strip() 
            cleaned_output = re.sub(r'```json\n?|```', '', llm_raw_output).strip()
            
            try:
                extracted_words = json.loads(cleaned_output)
            except Exception:
                extracted_words = re.findall(r'[가-힣]+', cleaned_output)
                
            context_text = f"이 단어는 {source_lang} '{query}'의 의미를 가집니다."
            

            valid_candidates = await self.filter_with_dict(extracted_words, context_text, expected_meaning=query)
            
            print(f"  사전검색 완료. 후보군: {list(valid_candidates.keys())}", flush=True)
            return valid_candidates
            
        except Exception as e:
            print(f"  사전검색 failed: {e}", flush=True)
            return {}

    async def generate_word_preview(self, word: str, definition: str, pos: str, output_dir: str, target_language: str = "en") -> dict:
        """
        [최적화본] LLM 번역 및 뜻 TTS 생성을 생략하고, 표제어와 표제어 음성만 즉시 생성
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Preview] '{word}' 고속 프리뷰 데이터 생성 중...", flush=True)
        word_audio_path = build_path(output_dir, f"temp_{word}_word.mp3")
        await self.generate_tts(word, word_audio_path, lang="ko")
        
        return {
            "word": word,
            "target_language": target_language,
            "pos_type": pos,
            "definition_korean": definition,
            "definition_translated": definition, # 번역 대신 원문 그대로 전달 (어차피 앱에서 안 보여줌)
            "pronunciation": "",                # 로마자 표기도 생략 (LLM이 필요하므로)
            "audio_path": word_audio_path,
            "def_trans_audio_path": None        # 뜻 음성 경로는 없음으로 처리
        }

    async def verify_spoken_word(self, audio_bytes: bytes, ffmpeg_bin: str, whisper_model_size: str, target_word: str) -> dict:
        """STT 추출 및 LLM 보정 후 타겟 단어 일치 여부 검증"""
        print(f"\n[Verification] '{target_word}' 음성 검증 시작...", flush=True)
        
        try:
            raw_text = await asyncio.to_thread(
                extract_text_from_audio, audio_bytes, ffmpeg_bin, whisper_model_size
            )
            raw_text = raw_text.strip()
            print(f"  -> STT 원문 (소음 정제 완료): {raw_text}", flush=True)
        except Exception as e:
            print(f"   STT 추출 실패: {e}", flush=True)
            return {"is_match": False, "target_word": target_word, "spoken_raw": "인식 오류", "spoken_corrected": ""}

        word_count = len(raw_text.split())
        if not raw_text or word_count >= 4:
            print(f"  -> ⚠️ 무음 또는 환각 감지됨 (어절 수: {word_count}). 검증 실패 처리.", flush=True)
            return {
                "is_match": False,
                "target_word": target_word,
                "spoken_raw": "음성이 인식되지 않았거나 잡음입니다.",
                "spoken_corrected": "다시 한 번 또렷하게 말씀해 주세요."
            }

        corrected_text = await self.correct_stt_text(raw_text)
        print(f"  -> LLM 보정문: {corrected_text}", flush=True)
        
        clean_target = re.sub(r'[^가-힣a-zA-Z0-9]', '', target_word)
        clean_spoken = re.sub(r'[^가-힣a-zA-Z0-9]', '', corrected_text)
        
        is_match = clean_target in clean_spoken
        
        if is_match:
            print(f"  검증 성공! (타겟 단어를 정확히 발음함)", flush=True)
        else:
            print(f"  검증 실패 (인식된 단어와 다름)", flush=True)
            
        return {
            "is_match": is_match,
            "target_word": target_word,
            "spoken_raw": raw_text,
            "spoken_corrected": corrected_text
        }

    async def phase2_generate(self, selected_words: dict, output_dir: str, target_language: str = "en") -> list:
        """Phase 2: 선택된 단어 수신 -> 품사별 맞춤 문법 분석 -> 다국어 오디오(단어뜻, 예문뜻) 세트 전체 디스크 생성"""
        os.makedirs(output_dir, exist_ok=True)
        final_cards = []
        t_start = time.time()

        for word, info in selected_words.items():
            word_dir = build_path(output_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            semantic_category = await self.fetch_on_term_category(word, info["definition"])
            
            await asyncio.sleep(0.2)
            
            # 품사 분기 세부 프롬프트 로직 가동
            llm_result = await self.process_with_llm(word, info["definition"], info["pos"], target_language)
            translated_def_text = llm_result.get("translated_definition", "")
            
            # 1. 한국어 표제어 음성 생성
            word_audio_path = build_path(word_dir, f"{word}_word.mp3")
            await self.generate_tts(word, word_audio_path, lang="ko")
            
            # 2. 번역 뜻 외국어(러시아어/영어 등) 원어 음성 MP3 실제 생성 저장
            def_trans_audio_path = build_path(word_dir, f"{word}_def_trans.mp3")
            await self.generate_tts(translated_def_text, def_trans_audio_path, lang=target_language)
            
            # 3. 3가지 품사 특화형 예문 및 예문 번역어 MP3 상호 생성 처리
            all_examples = []
            example_audio_paths = []
            
            for i, ex in enumerate(llm_result.get("easy_examples", [])):
                kor_text = ex.get("korean", "")
                trans_text = ex.get("translation", "")
                
                # 한국어 특화 예문형 MP3 생성
                ex_audio_path = build_path(word_dir, f"{word}_ex_{i+1}.mp3")
                await self.generate_tts(kor_text, ex_audio_path, lang="ko")
                example_audio_paths.append(ex_audio_path)
                
                # 번역 예문 외국어 원어 MP3 생성 추가
                trans_audio_path = build_path(word_dir, f"{word}_trans_ex_{i+1}.mp3")
                await self.generate_tts(trans_text, trans_audio_path, lang=target_language)
                
                all_examples.append({
                    "type": "llm_generated", 
                    "korean": kor_text, 
                    "translation": trans_text,
                    "audio_path": ex_audio_path,        # 한국어 예문 오디오 경로
                    "trans_audio_path": trans_audio_path # 외국어 번역 예문 오디오 경로
                })

            word_card = {
                "word": word,
                "target_language": target_language,
                "pronunciation": llm_result.get("romanization", ""),
                "pos_type": info["pos"], 
                "semantic_category": semantic_category,
                "definition_korean": info["definition"],
                "definition_translated": translated_def_text,
                "examples": all_examples,
                "audio": {
                    "word_tts": word_audio_path, 
                    "def_trans_tts": def_trans_audio_path,
                    "examples_tts": example_audio_paths
                }
            }
            final_cards.append(word_card)

            with open(build_path(word_dir, f"{word}_card.json"), 'w', encoding='utf-8') as f:
                json.dump(word_card, f, ensure_ascii=False, indent=4)
        
                
        print(f"  [Log] Phase 2 Total: {time.time() - t_start:.2f}s", flush=True)
        return final_cards
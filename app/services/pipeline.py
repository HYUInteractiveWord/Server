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
        """1. STT 텍스트 후보정 (비동기 처리)"""
        prompt = PromptTemplate.from_template(
            "당신은 한국어 음성 인식(STT) 오탈자 교정기입니다. 문맥을 추론하여 아예 다른 단어로 창조하지 마시고, 오직 '발음이 비슷하게 잘못 적힌 글자'만 올바른 맞춤법으로 고치세요.\n\n"
            "[예시]\n"
            "원본: 이 아름다운 해바라기 죠화는 어디서 줴작한 거\n"
            "보정본: 이 아름다운 해바라기 조화는 어디서 제작한 거\n"
            "원본: 어제 칭구랑 가치 바블 머거따\n"
            "보정본: 어제 친구랑 같이 밥을 먹었다\n\n"
            "[절대 규칙]\n"
            "1. 원본의 의미를 임의로 유추하여 완전히 다른 뜻의 단어로 바꾸면 절대 안 됩니다.\n"
            "2. 문장의 원래 의미와 시제를 100% 원본과 동일하게 유지하세요.\n"
            "3. 다른 설명 없이 교정된 텍스트만 출력하세요.\n\n"
            "원본: {text}\n보정본:"
        )
        chain = prompt | self.llm
        try:
            response = await chain.ainvoke({"text": text})
            return response.content.strip()
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

    async def fetch_basic_dict_data(self, word: str) -> dict:
        """기초사전 API에서 다의어 포함 뜻풀이 수집"""
        url = "https://krdict.korean.go.kr/api/search"
        params = {"key": self.dict_api_key, "q": word, "part": "word", "sort": "dict", "translated": "n", "advanced": "y", "method": "exact"}
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
                items = root.findall('.//item')[:5]
                senses = []
                for item in items:
                    pos = item.findtext('pos', '품사 없음')
                    sense = item.find('sense')
                    if sense is not None:
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

    async def select_best_definition(self, word: str, senses: list, context_text: str) -> dict:
        """다의어 문맥 분석 및 선택 (비동기 처리)"""
        prompt = PromptTemplate.from_template(
            "당신은 한국어 문맥 분석기입니다. 아래 문장에서 '{word}'라는 단어의 의미를 분석하세요.\n"
            "[문맥]: {context_text}\n"
            "[후보]\n{candidates}\n"
            "위 후보 중 알맞은 정의의 번호(index)를 찾으세요. JSON 형식으로 응답하세요.\n"
            "{{\n  \"reasoning\": \"이유\",\n  \"best_index\": 0\n}}"
        )
        candidates_str = "".join([f"{i}. [{s['pos']}] {s['definition']}\n" for i, s in enumerate(senses)])
        chain = prompt | self.llm | self.json_parser
        try:
            result = await chain.ainvoke({"word": word, "context_text": context_text, "candidates": candidates_str})
            best_idx = result.get("best_index", 0)
            return senses[best_idx] if 0 <= best_idx < len(senses) else senses[0]
        except Exception as e:
            print(f"  [Error] select_best_definition failed: {e}", flush=True)
            return senses[0]

    async def filter_with_dict(self, extracted_words: list, context_text: str) -> dict:
        """3. 기초사전 검증 ('품사 없음' 제외 및 다의어 필터링)"""
        valid_candidates = {}
        unique_words = list(set(extracted_words))
        
        for word in unique_words:
            dict_info = await self.fetch_basic_dict_data(word)
            
            if dict_info.get("status") == "success":
                valid_senses = [s for s in dict_info.get("senses", []) if s["pos"] and s["pos"] != "품사 없음"]
                if not valid_senses:
                    continue
                
                if len(valid_senses) > 1:
                    best_sense = await self.select_best_definition(word, valid_senses, context_text)
                else:
                    best_sense = valid_senses[0]
                    
                valid_candidates[word] = best_sense
                
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
        """LLM 번역 및 품사별 문법 특화 예문 생성 (동사/명사/형용사별 분기 조건 적용)"""
        target_lang_str = self._get_lang_str(target_language)
        
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
                '  "translated_definition": "[정의]를 목표 언어로 쉽게 설명한 문장",\n'
                '  "easy_examples": [\n'
                '    {{"korean": "과거 시제 반영 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "현재 시제 반영 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "미래 시제 반영 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                '  "romanization": "단어의 로마자 발음 표기"\n'
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
                '  "translated_definition": "[정의]를 목표 언어로 쉽게 설명한 문장",\n'
                '  "easy_examples": [\n'
                '    {{"korean": "명사가 주어로 쓰인 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "명사가 서술어로 쓰인 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "명사가 목적어로 쓰인 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                '  "romanization": "단어의 로마자 발음 표기"\n'
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
                '  "translated_definition": "[정의]를 목표 언어로 쉽게 설명한 문장",\n'
                '  "easy_examples": [\n'
                '    {{"korean": "명사 수식형 구조의 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "문장 끝 서술형 구조의 문장", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "연결 및 부사 구조적 변형 문장", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                '  "romanization": "단어의 로마자 발음 표기"\n'
                "}}"
            )
            
        # 4. Fallback (기타 부사, 관형사 등 예외 품사용 기본 3개 구성 자동 방어)
        else:
            prompt_text = (
                "[단어]: {word_raw} ({pos})\n"
                "[정의]: {definition}\n"
                "[목표 언어]: {target_lang_str}\n\n"
                "당신은 국어교육학 전문가입니다. 위 단어를 문맥 내에 배치하여 결합 형태가 서로 다른 유용하고 명확한 예문 3개를 생성하세요.\n"
                "반드시 아래 구조의 JSON 형식으로만 응답하세요:\n"
                "{{\n"
                '  "translated_definition": "[정의]를 목표 언어로 쉽게 설명한 문장",\n'
                '  "easy_examples": [\n'
                '    {{"korean": "활용 예문 1", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "활용 예문 2", "translation": "목표 언어 번역"}},\n'
                '    {{"korean": "활용 예문 3", "translation": "목표 언어 번역"}}\n'
                '  ],\n'
                '  "romanization": "단어의 로마자 발음 표기"\n'
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

    async def generate_tts(self, text: str, output_path: str):
        if not text: return
        communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
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
            valid_candidates = await self.filter_with_dict(extracted_words, context_text)
            
            print(f"  사전검색 완료. 후보군: {list(valid_candidates.keys())}", flush=True)
            return valid_candidates
            
        except Exception as e:
            print(f"  사전검색 failed: {e}", flush=True)
            return {}

    async def generate_word_preview(self, word: str, definition: str, pos: str, output_dir: str, target_language: str = "en") -> dict:
        """사전에서 선택한 단어의 임시 프리뷰 생성 (발음기호, 번역, 단어 음성만 빠르게 생성)"""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Preview] '{word}' 임시 확인용 데이터 생성 중...", flush=True)
        
        target_lang_str = self._get_lang_str(target_language)

        prompt = PromptTemplate.from_template(
            "당신은 한국어 교육 전문가입니다.\n"
            "[단어]: {word}\n[품사]: {pos}\n[정의]: {definition}\n[목표 언어]: {target_lang_str}\n\n"
            "이 단어에 대해 아래 JSON을 생성하세요:\n"
            "1. translated_definition (정의를 '목표 언어'로 번역 또는 쉽게 설명)\n"
            "2. romanization (단어의 로마자 발음 표기)\n"
            "{{\n"
            '  "translated_definition": "...",\n'
            '  "romanization": "..."\n'
            "}}"
        )
        chain = prompt | self.llm | self.json_parser
        try:
            llm_result = await chain.ainvoke({
                "word": word, 
                "definition": definition, 
                "pos": pos,
                "target_lang_str": target_lang_str
            })
        except Exception as e:
            print(f"  [Error] Preview LLM failed: {e}", flush=True)
            llm_result = {"translated_definition": "", "romanization": ""}
            
        word_audio_path = build_path(output_dir, f"temp_{word}_word.mp3")
        await self.generate_tts(word, word_audio_path)
        
        return {
            "word": word,
            "target_language": target_language,
            "pos_type": pos,
            "definition_korean": definition,
            "definition_translated": llm_result.get("translated_definition", ""),
            "pronunciation": llm_result.get("romanization", ""),
            "audio_path": word_audio_path
        }

    async def verify_spoken_word(self, audio_bytes: bytes, ffmpeg_bin: str, whisper_model_size: str, target_word: str) -> dict:
        """사용자가 녹음한 음성을 분석하여 타겟 단어와 일치하는지 검증"""
        print(f"\n[Verification] '{target_word}' 음성 검증 시작...", flush=True)
        
        try:
            raw_text = await asyncio.to_thread(
                extract_text_from_audio, audio_bytes, ffmpeg_bin, whisper_model_size
            )
            print(f"  -> STT 원문: {raw_text}", flush=True)
        except Exception as e:
            print(f"  ❌ STT 추출 실패: {e}", flush=True)
            return {"is_match": False, "target_word": target_word, "spoken_raw": "", "spoken_corrected": ""}

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
        """Phase 2: 선택된 단어 딕셔너리 수신 -> LLM 다국어 처리 -> TTS 생성 -> 파일 저장"""
        os.makedirs(output_dir, exist_ok=True)
        final_cards = []
        t_start = time.time()

        for word, info in selected_words.items():
            word_dir = build_path(output_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            semantic_category = await self.fetch_on_term_category(word, info["definition"])
            
            await asyncio.sleep(0.2)
            
            # 수정한 품사별 맞춤 프롬프트 실행부 호출 (항상 3가지 예문 객체 획득)
            llm_result = await self.process_with_llm(word, info["definition"], info["pos"], target_language)
            
            all_examples = [{"type": "llm_generated", "korean": ex.get("korean", ""), "translation": ex.get("translation", "")} 
                            for ex in llm_result.get("easy_examples", [])]
            
            word_audio_path = build_path(word_dir, f"{word}_word.mp3")
            await self.generate_tts(word, word_audio_path)
            
            example_audio_paths = []
            for i, ex_obj in enumerate(all_examples):
                ex_audio_path = build_path(word_dir, f"{word}_ex_{i+1}.mp3")
                await self.generate_tts(ex_obj["korean"], ex_audio_path)
                example_audio_paths.append(ex_audio_path)

            word_card = {
                "word": word,
                "target_language": target_language,
                "pronunciation": llm_result.get("romanization", ""),
                "pos_type": info["pos"], 
                "semantic_category": semantic_category,
                "definition_korean": info["definition"],
                "definition_translated": llm_result.get("translated_definition", ""),
                "examples": all_examples,
                "audio": {"word_tts": word_audio_path, "examples_tts": example_audio_paths}
            }
            final_cards.append(word_card)

            with open(build_path(word_dir, f"{word}_card.json"), 'w', encoding='utf-8') as f:
                json.dump(word_card, f, ensure_ascii=False, indent=4)
        

        if final_cards:
            with open(build_path(output_dir, "all_vocab_cards.json"), 'w', encoding='utf-8') as f:
                json.dump(final_cards, f, ensure_ascii=False, indent=4)
                
        print(f"  [Log] Phase 2 Total: {time.time() - t_start:.2f}s", flush=True)
        return final_cards
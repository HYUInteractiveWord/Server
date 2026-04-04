import os
import json
import asyncio
import requests
import re
import html
import xml.etree.ElementTree as ET
from konlpy.tag import Okt

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import edge_tts

class KoreanLearningPipeline:
    def __init__(self, term_api_key: str, dict_api_key: str, model_name: str = "gemma3:4b"):
        """초기화 및 모듈 설정"""
        self.term_api_key = term_api_key
        self.dict_api_key = dict_api_key
        self.llm = ChatOllama(model=model_name, temperature=0.0) # 온도를 0으로 낮춰 창의성 완전 억제
        self.okt = Okt()
        
    def _clean_text(self, text: str) -> str:
        """API 응답에 포함된 HTML 태그 및 엔티티 제거"""
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()

    def correct_stt_text(self, text: str) -> str:
        """1. STT 텍스트 후보정 (Few-Shot 기법으로 발음 교정 강제)"""
        print("\n[1] 음성 인식 텍스트 후보정 중...")
        prompt = PromptTemplate.from_template(
            "당신은 한국어 음성 인식(STT) 오탈자 교정기입니다. 문맥을 추론하여 아예 다른 단어로 창조하지 마시고, 오직 '발음이 비슷하게 잘못 적힌 글자'만 올바른 맞춤법으로 고치세요.\n\n"
            "[예시]\n"
            "원본: 이 아름다운 해바라기 조화는 어디서 줴작한 거\n"
            "보정본: 이 아름다운 해바라기 조화는 어디서 제작한 거\n"
            "원본: 어제 칭구랑 가치 바블 머거따\n"
            "보정본: 어제 친구랑 같이 밥을 먹었다\n\n"
            "[절대 규칙]\n"
            "1. 원본의 의미를 임의로 유추하여 '샀다'나 '사셨는지'처럼 완전히 다른 뜻의 단어로 바꾸면 절대 안 됩니다.\n"
            "2. 문장의 원래 의미와 시제를 100% 원본과 동일하게 유지하세요.\n"
            "3. 다른 설명 없이 교정된 텍스트만 출력하세요.\n\n"
            "원본: {text}\n보정본:"
        )
        chain = prompt | self.llm
        return chain.invoke({"text": text}).content.strip()

    def extract_core_vocabulary(self, text: str) -> list:
        """2. LLM을 활용한 1차 중요 단어 추출 (원형 복원 예시 강화)"""
        print("\n[2] LLM 핵심 단어 추출 중 (의미 없는 단어 1차 필터링)...")
        prompt = PromptTemplate.from_template(
            "당신은 한국어 교육 전문가입니다. 다음 텍스트에서 외국인 학습자에게 제공할 가치가 있는 '핵심 어휘(명사, 동사, 형용사)'만 추출하세요.\n"
            "- '이, 그, 저, 어디, 거, 것, 누구'와 같은 대명사, 의존명사는 절대 포함하지 마세요.\n"
            "- 동사와 형용사는 반드시 사전형(원형)으로 변환하세요.\n"
            "  (예시: '제작한' -> '제작하다', '예쁜' -> '예쁘다', '샀다' -> '사다')\n\n"
            "텍스트: {text}\n\n"
            "반드시 아래의 JSON 배열(문자열 리스트) 형식으로만 응답하세요.\n"
            '["단어1", "단어2", "단어3"]'
        )
        chain = prompt | self.llm | JsonOutputParser()
        try:
            return chain.invoke({"text": text})
        except Exception as e:
            print(f"핵심 단어 추출 중 오류 발생: {e}")
            return []

    def fetch_basic_dict_data(self, word: str) -> dict:
        """기초사전 API에서 단어 검색 (다의어 지원을 위해 여러 뜻풀이 수집)"""
        url = "https://krdict.korean.go.kr/api/search"
        params = {
            "key": self.dict_api_key,
            "q": word,
            "part": "word",
            "sort": "dict",
            "translated": "n",
            "advanced": "y",
            "method": "exact"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
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
                
                return {"status": "success", "senses": senses}
            return {"status": "fail"}
        except Exception as e:
            return {"status": "error"}

    def select_best_definition(self, word: str, senses: list, context_text: str) -> dict:
        print(f"  -> '{word}'의 다의어(의미 {len(senses)}개) 감지됨. 문맥 분석을 수행합니다...")
        
        prompt = PromptTemplate.from_template(
            "당신은 한국어 문맥 분석기입니다. 아래의 문장에서 '{word}'라는 단어가 어떤 의미로 사용되었는지 분석하세요.\n\n"
            "[문맥]: {context_text}\n\n"
            "[사전적 정의 후보]\n"
            "{candidates}\n\n"
            "위 후보 중 문맥에 가장 알맞은 정의의 번호(index)를 찾으세요. (예: 문맥에 '해바라기' 같은 꽃이 있다면 '조화'는 '가짜 꽃'입니다.)\n"
            "반드시 아래 JSON 형식으로만 응답하세요. 답을 고르기 전에 reasoning에 이유를 먼저 적으세요.\n"
            "{{\n"
            '  "reasoning": "문맥상 이 단어가 어떤 의미로 쓰였는지 짧은 분석",\n'
            '  "best_index": 0\n'
            "}}"
        )
        
        candidates_str = ""
        for i, s in enumerate(senses):
            candidates_str += f"{i}. [{s['pos']}] {s['definition']}\n"
            
        chain = prompt | self.llm | JsonOutputParser()
        try:
            result = chain.invoke({
                "word": word,
                "context_text": context_text,
                "candidates": candidates_str
            })
            
            # 추론 결과 출력 (디버깅용)
            print(f"     * LLM 추론: {result.get('reasoning', '추론 없음')}")
            
            best_idx = result.get("best_index", 0)
            if 0 <= best_idx < len(senses):
                return senses[best_idx]
            return senses[0]
        except Exception as e:
            return senses[0]

    def filter_with_dict(self, extracted_words: list, context_text: str) -> dict:
        """3. 기초사전 검증 ('품사 없음' 제외 및 다의어 문맥 필터링)"""
        print("\n[3] 한국어 기초사전 검증 중 ('품사 없음' 및 다의어 필터링)...")
        valid_candidates = {}
        unique_words = list(set(extracted_words))
        
        for word in unique_words:
            dict_info = self.fetch_basic_dict_data(word)
            if dict_info.get("status") == "success":
                senses = dict_info.get("senses", [])
                valid_senses = [s for s in senses if s["pos"] and s["pos"] != "품사 없음"]
                
                if not valid_senses:
                    print(f"[제외됨] {word} (유효한 품사 없음)")
                    continue
                
                if len(valid_senses) == 1:
                    best_sense = valid_senses[0]
                else:
                    best_sense = self.select_best_definition(word, valid_senses, context_text)
                    
                valid_candidates[word] = best_sense
                short_def = best_sense['definition'][:30] + "..." if len(best_sense['definition']) > 30 else best_sense['definition']
                print(f"[확인됨] {word} ({best_sense['pos']}) : {short_def}")
            else:
                print(f"[제외됨] {word} (사전 검색 실패)")
                
        return valid_candidates

    def select_words_from_user(self, valid_candidates: dict) -> list:
        """4. 사용자 단어 선택 인터페이스"""
        if not valid_candidates:
            print("\n추출 및 사전에 등록된 유효한 단어 후보가 없습니다.")
            return []

        print("\n[4] 기초사전에서 검증된 단어 목록입니다. 단어장에 추가할 단어를 선택하세요.")
        words_list = list(valid_candidates.items())
        
        for idx, (word, info) in enumerate(words_list):
            short_def = info['definition'][:30] + "..." if len(info['definition']) > 30 else info['definition']
            print(f"{idx + 1}. {word} [{info['pos']}] : {short_def}")
            
        selected_indices = input("\n원하는 단어의 번호를 쉼표(,)로 구분하여 입력하세요 (예: 1,3): ")
        
        selected_words = []
        try:
            indices = [int(i.strip()) - 1 for i in selected_indices.split(",")]
            selected_words = [words_list[i] for i in indices if 0 <= i < len(words_list)]
        except ValueError:
            print("잘못된 입력입니다.")
            
        return selected_words

    def fetch_on_term_category(self, original_word: str) -> str:
        """5. 온용어 세부 분류 태그 검색"""
        nouns = self.okt.nouns(original_word)
        search_keyword = nouns[0] if nouns else original_word
        
        print(f"\n[5] 온용어 API에서 '{search_keyword}' 분야 분류표 검색 중...")
        url = "https://kli.korean.go.kr/term/api/search"
        params = {"key": self.term_api_key, "q": search_keyword, "target": "JSON"}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            channel = data.get("channel", {})
            if channel.get("total", 0) > 0:
                return_objects = channel.get("return_object", [])
                if return_objects and return_objects[0].get("returnCode") == 1:
                    result_list = return_objects[0].get("resultlist", [])
                    if result_list:
                        target_data = result_list[0]
                        category_main = target_data.get("category_main", "")
                        category_sub = target_data.get("category_sub", "")
                        return f"{category_main} - {category_sub}".strip(" -")
            return ""
        except Exception as e:
            return ""

    def process_with_llm(self, word_raw: str, definition: str, pos: str) -> dict:
        """6. LLM 의미 번역 및 예문 생성"""
        print(f"\n[6] '{word_raw}' 의미 번역 및 예문 생성 중 (LLM)...")
        prompt = PromptTemplate.from_template(
            "당신은 한국어 교육 전문가입니다. 아래 데이터를 바탕으로 학습용 항목을 분석 및 생성하세요.\n\n"
            "[입력 데이터]\n"
            "- 단어: {word_raw}\n"
            "- 품사: {pos}\n"
            "- 사전적 정의: {definition}\n\n"
            "[요구사항]\n"
            "1. english_definition: '사전적 정의'를 자연스러운 영어로 번역하세요.\n"
            "2. easy_examples: 초급 한국어 학습자를 위해 이 단어를 활용한 아주 쉽고 짧은 한국어 예문 2개를 생성하고 영어 번역을 제공하세요.\n"
            "3. romanization: 단어의 로마자 발음을 영문으로 적으세요 (예: go-geup-seu-reop-da).\n\n"
            "반드시 JSON 형식으로만 응답하세요.\n"
            "{{\n"
            '  "english_definition": "...",\n'
            '  "easy_examples": [\n'
            '    {{"korean": "예문 1", "english": "예문 1 번역"}},\n'
            '    {{"korean": "예문 2", "english": "예문 2 번역"}}\n'
            '  ],\n'
            '  "romanization": "..."\n'
            "}}"
        )
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            return chain.invoke({
                "word_raw": word_raw,
                "definition": definition,
                "pos": pos
            })
        except Exception as e:
            return {"english_definition": "", "easy_examples": [], "romanization": ""}

    async def generate_tts(self, text: str, output_path: str):
        """gTTS/Edge-TTS 표준 발음 음성 생성"""
        if not text:
            return
        communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
        await communicate.save(output_path)

    async def run(self, stt_text: str, output_dir: str = "./output"):
        os.makedirs(output_dir, exist_ok=True)
        final_results = []

        # 1. 텍스트 교정
        corrected_text = self.correct_stt_text(stt_text)
        print(f"-> 보정된 텍스트: {corrected_text}")

        # 2. 핵심 어휘 추출
        extracted_words = self.extract_core_vocabulary(corrected_text)
        
        # 3. 기초사전 검증 (다의어 처리 시 문맥 파악을 위해 corrected_text 파라미터 전달)
        valid_candidates = self.filter_with_dict(extracted_words, corrected_text)
        
        # 4. 사용자 선택
        selected_words = self.select_words_from_user(valid_candidates)
        
        for word, dict_info in selected_words:
            definition = dict_info["definition"]
            pos = dict_info["pos"]
            
            # 5. 개별 단어 폴더 생성
            word_dir = os.path.join(output_dir, word)
            os.makedirs(word_dir, exist_ok=True)

            # 6. 온용어 세부 분류 추출
            semantic_category = self.fetch_on_term_category(word)
            if semantic_category:
                print(f"  -> 분류 확인됨: {semantic_category}")
            
            # 7. LLM 번역 및 예문 생성
            llm_result = self.process_with_llm(word, definition, pos)
            
            all_examples = []
            for ex in llm_result.get("easy_examples", []):
                all_examples.append({
                    "type": "llm_generated",
                    "korean": ex.get("korean", ""),
                    "english": ex.get("english", "")
                })
            
            # 8. TTS 생성
            word_audio_path = os.path.join(word_dir, f"{word}_word.mp3")
            await self.generate_tts(word, word_audio_path)
            
            example_audio_paths = []
            for i, ex_obj in enumerate(all_examples):
                ex_audio_path = os.path.join(word_dir, f"{word}_ex_{i+1}.mp3")
                await self.generate_tts(ex_obj["korean"], ex_audio_path)
                example_audio_paths.append(ex_audio_path)

            # JSON 구조화
            word_card = {
                "word": word,
                "pronunciation": llm_result.get("romanization", ""),
                "pos_type": pos, 
                "semantic_category": semantic_category,
                "definition_korean": definition,
                "definition_english": llm_result.get("english_definition", ""),
                "examples": all_examples,
                "audio": {
                    "word_tts": word_audio_path,
                    "examples_tts": example_audio_paths
                }
            }
            final_results.append(word_card)

            individual_json_path = os.path.join(word_dir, f"{word}_card.json")
            with open(individual_json_path, 'w', encoding='utf-8') as f:
                json.dump(word_card, f, ensure_ascii=False, indent=4)

        if final_results:
            output_json_path = os.path.join(output_dir, "all_vocab_cards.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)
            print(f"\n[완료] 데이터 저장 완료.")
            
        return final_results
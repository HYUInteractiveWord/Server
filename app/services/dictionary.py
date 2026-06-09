import xml.etree.ElementTree as ET
import httpx
from app.core.config import settings

def fallback_word_info(korean_word: str) -> dict:
    return {
        "pos": "명사",
        "definition": f"'{korean_word}'에 대한 사전 뜻 정보를 찾지 못했습니다. 임시 단어 카드로 저장할 수 있습니다.",
        "examples": [
            {
                "korean": f"{korean_word}를 배웠어요.",
                "english": f"I learned the word {korean_word}.",
            },
            {
                "korean": f"{korean_word}를 사용해서 문장을 만들었어요.",
                "english": f"I made a sentence using {korean_word}.",
            },
        ],
    }

def _strip_xml_namespace(root):
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

def _text(elem) -> str | None:
    if elem is None or elem.text is None:
        return None
    value = elem.text.strip()
    return value or None

def _first_text(root, tag_names: list[str]) -> str | None:
    for tag in tag_names:
        value = _text(root.find(f".//{tag}"))
        if value:
            return value
    return None

def _find_child_text(parent, tag_names: list[str]) -> str | None:
    if parent is None:
        return None
    for tag in tag_names:
        value = _text(parent.find(tag))
        if value:
            return value
    return None

def _parse_xml(raw_text: str):
    root = ET.fromstring(raw_text.strip())
    _strip_xml_namespace(root)
    return root

def _find_error(root) -> str | None:
    error_code = _first_text(root, ["error_code"])
    message = _first_text(root, ["message"])
    if error_code or message:
        return f"{error_code or ''} {message or ''}".strip()
    return None

# 수정- expected_meaning을 받아 번역 결과와 대조하여 동음이의어를 구분
def _get_search_target_code(raw_text: str, korean_word: str, expected_meaning: str = None) -> str | None:
    root = _parse_xml(raw_text)
    error = _find_error(root)
    if error:
        print(f"[Dict API] search error: {error}", flush=True)
        return None

    items = root.findall(".//item")
    if not items:
        print("[Dict API] search item 없음", flush=True)
        return None

    # 한국어가 정확히 일치하는 후보군 먼저 추리기
    exact_matches = []
    for item in items:
        word = _find_child_text(item, ["word"])
        if word == korean_word:
            exact_matches.append(item)

    if not exact_matches:
        exact_matches = items # 완전히 일치하는게 없으면 전체 후보 사용

    selected = None

    # 외국어 뜻이 있다면 번역 결과와 대조
    if expected_meaning:
        expected_lower = expected_meaning.lower().strip()
        for item in exact_matches:
            # item 하위의 모든 <trans_word> (번역어) 태그 탐색
            trans_words = [t.text.lower() for t in item.findall(".//trans_word") if t.text]
            # 예상하는 의미가 번역어 목록에 포함되어 있는지 확인
            if any(expected_lower in tw for tw in trans_words):
                selected = item
                print(f"[Dict API] 문맥(영어) 매칭 성공! '{expected_meaning}' 포함됨.", flush=True)
                break

    # 일치하는 뜻을 못 찾았거나 expected_meaning이 없으면 늘 하던대로 첫 번째 선택
    if selected is None:
        selected = exact_matches[0]

    target_code = _find_child_text(selected, ["target_code", "word_no", "targetCode"])
    word = _find_child_text(selected, ["word"])

    print(f"[Dict API] search 선택 word={word}, target_code={target_code}", flush=True)
    return target_code

def _parse_view_info(raw_text: str, korean_word: str) -> dict | None:
    root = _parse_xml(raw_text)
    error = _find_error(root)
    if error:
        print(f"[Dict API] view error: {error}", flush=True)
        return None

    word = _first_text(root, ["word"]) or korean_word
    pos = _first_text(root, ["pos", "word_unit", "part_of_speech"]) or "명사"

    definition = _first_text(root, ["definition", "dfn", "sense_definition"])

    if not definition:
        sense = root.find(".//sense")
        if sense is not None:
            definition = _find_child_text(sense, ["definition", "dfn"])

    definition_english = None
    translation = root.find(".//translation")
    if translation is not None:
        definition_english = _find_child_text(
            translation,
            ["trans_dfn", "trans_word", "translation", "definition"],
        )

    if not definition:
        tags = []
        for elem in list(root.iter())[:60]:
            if elem.tag not in tags:
                tags.append(elem.tag)
        print(f"[Dict API] view에서 definition 못 찾음. tags={tags}", flush=True)
        return None

    return {
        "pos": pos,
        "definition": definition,
        "definition_english": definition_english,
        "examples": [
            {
                "korean": f"{word}를 배웠어요.",
                "english": f"I learned the word {word}.",
            },
            {
                "korean": f"{word}를 사용해서 문장을 만들었어요.",
                "english": f"I made a sentence using {word}.",
            },
        ],
    }

# 수정- expected_meaning 파라미터 추가 및 API 호출 시 번역 옵션 추가
def fetch_word_info(korean_word: str, expected_meaning: str = None) -> dict:
    """
    한국어기초사전 API:
    - expected_meaning: 원본 외국어 단어(예: "horse")를 넘겨주면 동음이의어 분간에 사용.
    """
    if not settings.DICT_API_KEY or settings.DICT_API_KEY == "dummy-key":
        print("[Dict API] DICT_API_KEY dummy. fallback 사용", flush=True)
        return fallback_word_info(korean_word)

    try:
        with httpx.Client(timeout=15.0, verify=False) as client:
            search_resp = client.get(
                "https://krdict.korean.go.kr/api/search",
                params={
                    "key": settings.DICT_API_KEY,
                    "q": korean_word,
                    "part": "word",
                    "sort": "popular",
                    "num": 10,
                    "translated": "y",   # 💡 번역 결과 포함
                    "trans_lang": "1",   # 💡 영어로 번역
                },
                headers={"User-Agent": "InteractiveWord/1.0"},
            )
            search_resp.raise_for_status()

            # expected_meaning 전달
            target_code = _get_search_target_code(search_resp.text, korean_word, expected_meaning)

            direct_info = None
            try:
                direct_info = _parse_view_info(search_resp.text, korean_word)
            except Exception:
                direct_info = None

            if direct_info and not target_code:
                print(f"[Dict API] '{korean_word}' search 직접 뜻 발견: {direct_info['definition']}", flush=True)
                return direct_info

            if not target_code:
                print(f"[Dict API] '{korean_word}' target_code 없음. fallback 사용", flush=True)
                return fallback_word_info(korean_word)

            view_resp = client.get(
                "https://krdict.korean.go.kr/api/view",
                params={
                    "key": settings.DICT_API_KEY,
                    "method": "target_code",
                    "q": target_code,
                    "translated": "y",
                    "trans_lang": "1",
                },
                headers={"User-Agent": "InteractiveWord/1.0"},
            )
            view_resp.raise_for_status()

            parsed = _parse_view_info(view_resp.text, korean_word)
            if parsed:
                print(f"[Dict API] '{korean_word}' view 뜻 발견: {parsed['definition']}", flush=True)
                return parsed

            print(f"[Dict API] '{korean_word}' view 파싱 실패. fallback 사용", flush=True)
            return fallback_word_info(korean_word)

    except Exception as e:
        print(f"사전검색 failed: {e}", flush=True)
        return fallback_word_info(korean_word)
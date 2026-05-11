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


def _get_search_target_code(raw_text: str, korean_word: str) -> str | None:
    root = _parse_xml(raw_text)
    error = _find_error(root)
    if error:
        print(f"[Dict API] search error: {error}", flush=True)
        return None

    items = root.findall(".//item")
    if not items:
        print("[Dict API] search item 없음", flush=True)
        return None

    selected = None

    # 정확히 같은 단어 우선
    for item in items:
        word = _find_child_text(item, ["word"])
        if word == korean_word:
            selected = item
            break

    if selected is None:
        selected = items[0]

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

    # 가장 중요한 값: 뜻풀이
    definition = _first_text(root, ["definition", "dfn", "sense_definition"])

    # 한국어기초사전 view 구조에서 sense 아래에 있을 가능성도 처리
    if not definition:
        sense = root.find(".//sense")
        if sense is not None:
            definition = _find_child_text(sense, ["definition", "dfn"])

    # 영어 번역 뜻
    definition_english = None
    translation = root.find(".//translation")
    if translation is not None:
        definition_english = _find_child_text(
            translation,
            ["trans_dfn", "trans_word", "translation", "definition"],
        )

    if not definition:
        # 디버깅용: 실제 태그 구조 일부 출력
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


def fetch_word_info(korean_word: str) -> dict:
    """
    한국어기초사전 API:
    1) /api/search 로 target_code 찾기
    2) /api/view 로 상세 뜻 조회
    실패하면 fallback 반환
    """
    if not settings.DICT_API_KEY or settings.DICT_API_KEY == "dummy-key":
        print("[Dict API] DICT_API_KEY dummy. fallback 사용", flush=True)
        return fallback_word_info(korean_word)

    try:
        with httpx.Client(timeout=15.0, verify=False) as client:
            # 1. search: target_code 확보
            search_resp = client.get(
                "https://krdict.korean.go.kr/api/search",
                params={
                    "key": settings.DICT_API_KEY,
                    "q": korean_word,
                    "part": "word",
                    "sort": "popular",
                    "num": 10,
                },
                headers={"User-Agent": "InteractiveWord/1.0"},
            )
            search_resp.raise_for_status()

            target_code = _get_search_target_code(search_resp.text, korean_word)

            # search 응답 자체에 definition이 있을 수도 있으므로 view 전에 한번 시도
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

            # 2. view: 상세 뜻 조회
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

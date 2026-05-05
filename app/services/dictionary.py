import httpx
from app.core.config import settings


def fallback_word_info(korean_word: str) -> dict:
    return {
        "pos": "명사",
        "definition": f"'{korean_word}'에 대한 사전 뜻 정보를 찾지 못했습니다. 임시 단어 카드로 저장할 수 있습니다.",
        "examples": [
            {
                "korean": f"{korean_word}를 배웠어요.",
                "english": f"I learned the word {korean_word}."
            },
            {
                "korean": f"{korean_word}를 사용해서 문장을 만들었어요.",
                "english": f"I made a sentence using {korean_word}."
            }
        ],
    }


def fetch_word_info(korean_word: str) -> dict:
    """
    국립국어원 한국어기초사전 API로 단어 정보를 조회한다.
    실패하거나 결과가 비어 있으면 프론트 테스트가 가능하도록 fallback 정보를 반환한다.
    """
    if not settings.DICT_API_KEY or settings.DICT_API_KEY == "dummy-key":
        return fallback_word_info(korean_word)

    url = "https://krdict.korean.go.kr/api/search"
    params = {
        "key": settings.DICT_API_KEY,
        "q": korean_word,
        "translated": "y",
        "trans_lang": "1",
        "sort": "popular",
        "num": 1,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        items = data.get("channel", {}).get("item", [])
        if not items:
            return fallback_word_info(korean_word)

        item = items[0]
        senses = item.get("sense", [])
        first_sense = senses[0] if senses else {}

        definition = first_sense.get("definition")
        pos = item.get("pos")

        if not definition:
            return fallback_word_info(korean_word)

        return {
            "pos": pos or "명사",
            "definition": definition,
            "examples": [
                {
                    "korean": f"{korean_word}를 배웠어요.",
                    "english": f"I learned the word {korean_word}."
                },
                {
                    "korean": f"{korean_word}를 사용해서 문장을 만들었어요.",
                    "english": f"I made a sentence using {korean_word}."
                }
            ],
        }

    except Exception as e:
        print(f"사전검색 failed: {e}")
        return fallback_word_info(korean_word)

import httpx
from app.core.config import settings


def fetch_word_info(korean_word: str) -> dict:
    """
    국립국어원 한국어기초사전 Open API로 단어 정보 조회.
    API 키 미설정 시 빈 dict 반환.
    """
    if not settings.KRDICT_API_KEY:
        return {}

    url = "https://krdict.korean.go.kr/api/search"
    params = {
        "key": settings.KRDICT_API_KEY,
        "q": korean_word,
        "translated": "y",
        "trans_lang": "1",  # 영어
        "sort": "popular",
        "num": 1,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        item = data.get("channel", {}).get("item", [{}])[0]
        return {
            "pos": item.get("pos"),
            "definition": item.get("sense", [{}])[0].get("definition"),
            "examples": [],
        }
    except Exception:
        return {}

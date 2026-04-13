from fastapi import APIRouter, HTTPException, Query
from app.schemas.dictionary import DictionarySearchResponse
from app.services.dictionary import fetch_word_info

router = APIRouter(prefix="/dictionary", tags=["dictionary"])


@router.get("/search", response_model=DictionarySearchResponse)
def search_dictionary(word: str = Query(..., min_length=1)):
    word = word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="word is required")

    info = fetch_word_info(word)

    return DictionarySearchResponse(
        word=word,
        pos=info.get("pos"),
        definition=info.get("definition"),
    )
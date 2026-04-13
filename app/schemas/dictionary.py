from pydantic import BaseModel


class DictionarySearchResponse(BaseModel):
    word: str
    pos: str | None = None
    definition: str | None = None
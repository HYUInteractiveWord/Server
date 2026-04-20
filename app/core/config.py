from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    DICT_API_KEY: str
    TERM_API_KEY: str
    LLM_MODEL_NAME: str = "gemma-4-E4B-it-Q5_K_M"
    OPENAI_API_KEY: str = "empty"
    FFMPEG_BIN: str = r"C:\ffmpeg\bin"  # ffmpeg 설치 경로
    WHISPER_MODEL: str = "small"         # base | small | medium

    class Config:
        env_file = ".env"


settings = Settings()

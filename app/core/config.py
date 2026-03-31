from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    KRDICT_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    FFMPEG_BIN: str = r"C:\ffmpeg\bin"  # ffmpeg 설치 경로
    WHISPER_MODEL: str = "small"         # base | small | medium

    class Config:
        env_file = ".env"


settings = Settings()

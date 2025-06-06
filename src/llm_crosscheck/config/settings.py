from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "LLM CrossCheck AI Service"
    APP_DESCRIPTION: str = "A service to validate outputs from LLMs"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_METRICS: bool = True
    ALLOWED_ORIGINS: list[str] = ["*"]
    ALLOWED_METHODS: list[str] = ["*"]
    ALLOWED_HEADERS: list[str] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10


def get_settings() -> Settings:
    return Settings()

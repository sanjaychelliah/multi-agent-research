"""Central configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    # Pipeline
    MAX_SUBTASKS: int = int(os.getenv("MAX_SUBTASKS", "4"))

    # Storage
    METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", "./metrics.db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def has_tavily(cls) -> bool:
        return bool(cls.TAVILY_API_KEY)

    @classmethod
    def validate(cls) -> None:
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Copy .env.example → .env and fill it in.")
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required.")


cfg = Config()

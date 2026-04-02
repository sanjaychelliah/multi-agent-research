"""Central configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM
    # Supported providers: openai | anthropic | openrouter | ollama
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Provider API keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # OpenRouter base URL (override if using a self-hosted proxy)
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Ollama base URL (default: local)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

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
        provider = cls.LLM_PROVIDER
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required for provider=openai. "
                "Copy .env.example → .env and fill it in."
            )
        if provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for provider=anthropic.")
        if provider == "openrouter" and not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for provider=openrouter.")
        if provider == "ollama":
            pass  # no key required for local Ollama
        if provider not in ("openai", "anthropic", "openrouter", "ollama"):
            raise ValueError(
                f"Unknown LLM_PROVIDER '{provider}'. "
                "Choose from: openai | anthropic | openrouter | ollama"
            )


cfg = Config()

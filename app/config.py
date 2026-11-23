"""All settings live here. Change providers/models in .env — no code changes needed."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    app_name: str = "Chat with Docs"
    database_url: str = "sqlite:///./data/app.db"
    chroma_dir: str = "./data/chroma"
    upload_dir: str = "./data/uploads"
    max_upload_mb: int = 20

    # --- Auth ---
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # --- LLM (the "one setting" switch) ---
    # Providers: gemini | groq | claude | openai_compatible
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    # Backup provider used automatically when the primary fails (e.g. daily limit hit).
    # Leave llm_fallback_provider empty to disable.
    llm_fallback_provider: str = "groq"
    llm_fallback_model: str = "llama-3.3-70b-versatile"

    gemini_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    # For any other OpenAI-compatible server (e.g. Ollama: http://localhost:11434/v1)
    openai_compatible_base_url: str = "http://localhost:11434/v1"
    openai_compatible_api_key: str = "ollama"

    # --- Embeddings (free, run locally) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- RAG ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()

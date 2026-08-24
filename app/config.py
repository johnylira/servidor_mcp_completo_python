"""Configurações centralizadas, carregadas de variáveis de ambiente."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Valida a configuração e define valores padrão seguros para desenvolvimento."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str | None = None
    rag_data_dir: Path = Path("/data")
    rag_collection: str = "mcp_rag"
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
    """Retorna uma única instância de configurações e cria o diretório de dados."""
    settings = Settings()
    settings.rag_data_dir.mkdir(parents=True, exist_ok=True)
    return settings

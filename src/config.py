from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )


settings = Settings()

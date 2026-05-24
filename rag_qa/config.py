from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    base_storage_dir: Path
    uploads_dir: Path
    index_dir: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        base_storage_dir = Path(os.getenv("RAG_STORAGE_DIR", "storage")).resolve()
        uploads_dir = base_storage_dir / "uploads"
        index_dir = base_storage_dir / "faiss_index"

        uploads_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip(),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip(),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            retrieval_k=int(os.getenv("RAG_RETRIEVAL_K", "6")),
            base_storage_dir=base_storage_dir,
            uploads_dir=uploads_dir,
            index_dir=index_dir,
        )

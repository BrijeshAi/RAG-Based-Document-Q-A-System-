from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_qa.config import AppConfig
from rag_qa.models import IndexBuildStats


class IndexManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.manifest_path = self.config.index_dir / "manifest.json"

    # def _embeddings(self) -> OpenAIEmbeddings:
    #     return OpenAIEmbeddings(
    #         model=self.config.embedding_model,
    #         api_key=self.config.openai_api_key,
    #     )
    def _embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def index_exists(self) -> bool:
        return (
            (self.config.index_dir / "index.faiss").exists()
            and (self.config.index_dir / "index.pkl").exists()
        )

    def build_and_save(self, documents: list[Document]) -> IndexBuildStats:
        if not documents:
            raise ValueError("Cannot build an index without documents.")

        vector_store = FAISS.from_documents(documents, self._embeddings())
        vector_store.save_local(str(self.config.index_dir))
        self._write_manifest(documents=documents)

        source_files = {document.metadata.get("filename", "unknown") for document in documents}
        return IndexBuildStats(
            chunk_count=len(documents),
            source_file_count=len(source_files),
        )

    def load(self) -> FAISS:
        if not self.index_exists():
            raise FileNotFoundError("No saved FAISS index was found.")

        return FAISS.load_local(
            folder_path=str(self.config.index_dir),
            embeddings=self._embeddings(),
            allow_dangerous_deserialization=True,
        )

    def clear_index(self) -> None:
        if self.config.index_dir.exists():
            shutil.rmtree(self.config.index_dir)
        self.config.index_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None

        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, documents: list[Document]) -> None:
        filenames = sorted({document.metadata.get("filename", "unknown") for document in documents})
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.config.embedding_model,
            "chat_model": self.config.chat_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "retrieval_k": self.config.retrieval_k,
            "chunk_count": len(documents),
            "source_file_count": len(filenames),
            "filenames": filenames,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

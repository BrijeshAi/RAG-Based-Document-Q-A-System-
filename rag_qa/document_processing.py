from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rag_qa.models import FileProcessingSummary, ProcessingSummary


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "document.pdf"


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _uploaded_file_to_path(uploaded_file: UploadedFile, target_dir: Path) -> Path:
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha1(file_bytes).hexdigest()[:10]
    safe_name = sanitize_filename(uploaded_file.name)
    stored_name = f"{Path(safe_name).stem}_{digest}.pdf"
    destination = target_dir / stored_name
    destination.write_bytes(file_bytes)
    return destination


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_page_text(
    *,
    filename: str,
    stored_path: Path,
    page_number: int,
    page_text: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    chunks = splitter.split_text(page_text)
    documents: list[Document] = []

    for chunk_index, chunk in enumerate(chunks):
        metadata = {
            "filename": filename,
            "stored_path": str(stored_path),
            "page_number": page_number,
            "chunk_index": chunk_index,
            "citation": f"{filename}, page {page_number}",
        }
        documents.append(Document(page_content=chunk, metadata=metadata))

    return documents


def save_uploaded_files_and_create_chunks(
    *,
    uploaded_files: Iterable[UploadedFile],
    uploads_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], ProcessingSummary]:
    reset_directory(uploads_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_documents: list[Document] = []
    file_summaries: list[FileProcessingSummary] = []
    total_pages = 0

    for uploaded_file in uploaded_files:
        stored_path = _uploaded_file_to_path(uploaded_file, uploads_dir)
        original_filename = uploaded_file.name
        reader = PdfReader(str(stored_path))

        file_chunk_count = 0
        file_page_count = 0

        for page_number, page in enumerate(reader.pages, start=1):
            extracted_text = _clean_text(page.extract_text() or "")
            if not extracted_text:
                continue

            page_documents = _chunk_page_text(
                filename=original_filename,
                stored_path=stored_path,
                page_number=page_number,
                page_text=extracted_text,
                splitter=splitter,
            )
            if not page_documents:
                continue

            all_documents.extend(page_documents)
            file_chunk_count += len(page_documents)
            file_page_count += 1
            total_pages += 1

        file_summaries.append(
            FileProcessingSummary(
                filename=original_filename,
                page_count=file_page_count,
                chunk_count=file_chunk_count,
            )
        )

    if not all_documents:
        raise ValueError("No extractable text was found in the uploaded PDFs.")

    summary = ProcessingSummary(
        file_count=len(file_summaries),
        page_count=total_pages,
        chunk_count=len(all_documents),
        file_summaries=file_summaries,
    )
    return all_documents, summary

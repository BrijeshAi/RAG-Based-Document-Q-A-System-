from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FileProcessingSummary:
    filename: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class ProcessingSummary:
    file_count: int
    page_count: int
    chunk_count: int
    file_summaries: list[FileProcessingSummary]


@dataclass(frozen=True)
class Citation:
    source_id: str
    filename: str
    page_number: int
    chunk_index: int


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    filename: str
    page_number: int
    chunk_index: int
    preview: str
    raw_score: float
    relative_relevance: float


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    answer_found: bool
    confidence_label: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


@dataclass(frozen=True)
class IndexBuildStats:
    chunk_count: int
    source_file_count: int

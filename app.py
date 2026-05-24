from __future__ import annotations

import streamlit as st

from rag_qa.config import AppConfig
from rag_qa.document_processing import save_uploaded_files_and_create_chunks
from rag_qa.models import ProcessingSummary
from rag_qa.qa import AnswerResult, RAGPipeline
from rag_qa.vector_store import IndexManager


def initialize_session_state() -> None:
    st.session_state.setdefault("chat_messages", [])


def render_sidebar(config: AppConfig, index_manager: IndexManager) -> None:
    manifest = index_manager.load_manifest()

    st.sidebar.header("Configuration")
    st.sidebar.caption("Environment variables override these defaults.")
    st.sidebar.code(
        "\n".join(
            [
                f"Embedding model: {config.embedding_model}",
                f"Chat model: {config.chat_model}",
                f"Chunk size: {config.chunk_size}",
                f"Chunk overlap: {config.chunk_overlap}",
                f"Top-K retrieval: {config.retrieval_k}",
            ]
        ),
        language="text",
    )

    api_key_ready = bool(config.openai_api_key)
    if api_key_ready:
        st.sidebar.success("`OPENAI_API_KEY` detected.")
    else:
        st.sidebar.error("Missing `OPENAI_API_KEY`. Add it to your environment or `.env` file.")

    st.sidebar.divider()
    st.sidebar.subheader("Index status")

    if manifest:
        st.sidebar.success("Local FAISS index found.")
        st.sidebar.write(
            f"Files: **{manifest.get('source_file_count', 0)}**  \n"
            f"Chunks: **{manifest.get('chunk_count', 0)}**  \n"
            f"Built: **{manifest.get('created_at', 'unknown')}**"
        )
        filenames = manifest.get("filenames", [])
        if filenames:
            st.sidebar.caption("Indexed documents")
            st.sidebar.write(", ".join(sorted(filenames)))
    else:
        st.sidebar.info("No saved index yet.")

    if st.sidebar.button("Clear saved index", use_container_width=True):
        index_manager.clear_index()
        st.session_state.chat_messages = []
        st.sidebar.success("Saved index cleared.")
        st.rerun()

    if st.sidebar.button("Reset chat session", use_container_width=True):
        st.session_state.chat_messages = []
        st.sidebar.success("Chat history cleared.")
        st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def format_answer_markdown(result: AnswerResult) -> str:
    if not result.citations:
        return result.answer

    citation_lines = [
        f"- `{citation.source_id}`: **{citation.filename}**, page **{citation.page_number}**"
        for citation in result.citations
    ]
    return f"{result.answer}\n\n**Citations**\n" + "\n".join(citation_lines)


def render_retrieval_details(result: AnswerResult) -> None:
    with st.expander("Retrieved source chunks", expanded=False):
        st.caption(
            "Relative relevance is normalized within this query only. Higher values mean the chunk ranked closer to the top."
        )
        for chunk in result.retrieved_chunks:
            st.markdown(
                f"**{chunk.source_id}** | `{chunk.filename}` | page **{chunk.page_number}** | "
                f"relative relevance **{chunk.relative_relevance:.2f}**"
            )
            st.write(chunk.preview)


def build_index_ui(config: AppConfig, index_manager: IndexManager) -> None:
    st.subheader("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload large batches of PDFs. Rebuilding replaces the previous saved index.",
    )

    if st.button("Build / Rebuild index", type="primary", use_container_width=True):
        if not config.openai_api_key:
            st.error("Set `OPENAI_API_KEY` before building the index.")
            return

        if not uploaded_files:
            st.warning("Upload at least one PDF before building the index.")
            return

        with st.status("Building index...", expanded=True) as status:
            try:
                status.write("Saving uploaded files...")
                documents, summary = save_uploaded_files_and_create_chunks(
                    uploaded_files=uploaded_files,
                    uploads_dir=config.uploads_dir,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )

                status.write("Creating embeddings and FAISS index...")
                build_stats = index_manager.build_and_save(documents=documents)

                status.update(label="Index ready", state="complete")
                st.success(
                    f"Indexed {summary.file_count} PDFs into {build_stats.chunk_count} chunks. "
                    f"Saved index at `{config.index_dir}`."
                )
                render_processing_summary(summary)
            except Exception as exc:  # noqa: BLE001
                status.update(label="Index build failed", state="error")
                st.error(f"Failed to build the index: {exc}")


def render_processing_summary(summary: ProcessingSummary) -> None:
    with st.expander("Document processing summary", expanded=False):
        st.write(
            f"Files processed: **{summary.file_count}**  \n"
            f"Pages read: **{summary.page_count}**  \n"
            f"Chunks created: **{summary.chunk_count}**"
        )
        for detail in summary.file_summaries:
            st.markdown(
                f"- `{detail.filename}`: {detail.page_count} pages, {detail.chunk_count} chunks"
            )


def qa_ui(config: AppConfig, index_manager: IndexManager) -> None:
    st.subheader("2. Ask questions")
    has_index = index_manager.index_exists()

    if not has_index:
        st.info("Build an index to start asking questions.")
        return

    render_chat_history()

    question = st.chat_input("Ask a question about the indexed documents")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        print("STEP 1: Creating pipeline")

        pipeline = RAGPipeline(config=config, index_manager=index_manager)

        print("STEP 2: Pipeline created")

        history = [
            message
            for message in st.session_state.chat_messages[:-1]
            if message["role"] in {"user", "assistant"}
        ]

        print("STEP 3: Calling answer_question")

        result = pipeline.answer_question(
            question=question,
            chat_history=history
        )

        print("STEP 4: answer_question completed")
    except Exception as exc:  # noqa: BLE001
        with st.chat_message("assistant"):
            st.error(f"Unable to answer right now: {exc}")
        return

    assistant_markdown = format_answer_markdown(result)
    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_markdown})

    with st.chat_message("assistant"):
        st.markdown(assistant_markdown)
        st.caption(f"Confidence: **{result.confidence_label}**")
        render_retrieval_details(result)


def main() -> None:
    st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
    st.title("RAG-Based Document Q&A")
    st.caption("Upload PDFs, build a FAISS index, and get grounded answers with citations.")

    initialize_session_state()

    config = AppConfig.from_env()
    index_manager = IndexManager(config=config)

    render_sidebar(config=config, index_manager=index_manager)
    build_index_ui(config=config, index_manager=index_manager)
    st.divider()
    qa_ui(config=config, index_manager=index_manager)


if __name__ == "__main__":
    main()

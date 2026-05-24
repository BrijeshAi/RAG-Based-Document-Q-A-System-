# Short Summary in Points

## What the project does

- Builds a RAG-based document Q&A app using Streamlit, LangChain, FAISS, and OpenAI.
- Lets users upload multiple PDFs and ask natural-language questions about them.
- Returns answers grounded in document content with source citations.

## Main workflow

- Upload PDFs from the Streamlit UI.
- Extract text from each PDF page.
- Split text into chunks.
- Add metadata like filename, page number, and chunk index.
- Generate embeddings using OpenAI.
- Store embeddings in a local FAISS index.
- Retrieve top relevant chunks for each question.
- Generate answers only from retrieved context.
- Show citations in the final answer.

## Important files

- `app.py`: Streamlit entry point and UI flow.
- `rag_qa/config.py`: environment variables and app settings.
- `rag_qa/document_processing.py`: PDF saving, text extraction, cleaning, and chunking.
- `rag_qa/vector_store.py`: FAISS build, load, clear, and manifest handling.
- `rag_qa/qa.py`: retrieval + LLM answer generation + citation validation.
- `rag_qa/models.py`: shared dataclasses for summaries, citations, and results.

## Key features

- Supports multiple PDF uploads.
- Preserves source metadata for citations.
- Saves and reloads FAISS index locally.
- Includes chat history support.
- Includes clear index and reset chat actions.
- Shows retrieved chunk details and relative relevance.
- Uses environment variables for secrets and configuration.

## How hallucination is reduced

- The model sees only retrieved chunks as context.
- The prompt tells it to avoid outside knowledge.
- It must say when the answer is not in the documents.
- Only valid retrieved source IDs are accepted as citations.

## Output shown to the user

- Final grounded answer
- Citation list with filename and page number
- Confidence label
- Retrieved chunk previews in an expandable section

## Why the structure is good

- Easy to read
- Easy to extend
- Clear separation between UI, ingestion, vector storage, and Q&A logic
- Good baseline for production-oriented internal apps

## Good next enhancements

- OCR support for scanned PDFs
- More file types like DOCX and TXT
- Database-backed chat history
- Managed vector database
- Authentication and user-specific indexes

from typing import Dict, List

import chromadb

from config import CHROMA_DB_PATH, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS
from ingest.parser import extract_section_text, table_to_text
from utils.retrieval import get_embeddings_model


def sliding_window_chunk(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split text into overlapping chunks. Uses 4 chars ~ 1 token approximation."""
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Break at sentence boundary when possible
        if end < len(text):
            search_start = max(start, end - 200)
            sentence_end = max(
                text.rfind(". ", search_start, end),
                text.rfind("! ", search_start, end),
                text.rfind("? ", search_start, end),
                text.rfind("\n\n", search_start, end),
            )
            if sentence_end > start:
                end = sentence_end + 1

        chunks.append(text[start:end].strip())
        start = end - overlap_chars

    return chunks


def create_chunks(pdf_content: Dict, file_name: str) -> List[Dict]:
    """Section-wise chunking with metadata."""
    chunks = []
    toc_with_pages = pdf_content["toc_with_pages"]

    for entry in toc_with_pages:
        section_title = entry["section"]
        page_start = entry["page_start"]
        page_end = entry["page_end"]

        section_text = extract_section_text(pdf_content["pages_text"], page_start, page_end)
        if not section_text.strip():
            continue

        section_chunks = sliding_window_chunk(section_text)

        for chunk_text in section_chunks:
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "file_name": file_name,
                    "chunk_type": "text",
                    "section_title": section_title,
                    "page_start": page_start,
                    "page_end": page_end,
                })

    # Tables as separate chunks
    for table in pdf_content["tables"]:
        table_text = table_to_text(table["data"])
        if not table_text.strip():
            continue

        table_page = table["page"]
        matching_section = next(
            (e for e in toc_with_pages if e["page_start"] <= table_page <= e["page_end"]),
            None,
        )

        chunks.append({
            "text": table_text,
            "file_name": file_name,
            "chunk_type": "table",
            "section_title": matching_section["section"] if matching_section else "Unknown",
            "page_start": table_page,
            "page_end": table_page,
        })

    return chunks


def store_chunks_in_db(chunks: List[Dict], ingestion_id: str):
    """Embed and store chunks in ChromaDB."""
    if not chunks:
        return

    embeddings_model = get_embeddings_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=f"ingestion_{ingestion_id}")

    metadatas = [
        {
            "file_name": c["file_name"],
            "chunk_type": c["chunk_type"],
            "section_title": c["section_title"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
        }
        for c in chunks
    ]

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=[f"{ingestion_id}_chunk_{i}" for i in range(len(chunks))],
    )

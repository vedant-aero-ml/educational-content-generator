import logging
from typing import Dict, List, Tuple

import chromadb

from config import (
    CHROMA_DB_PATH,
    DEFAULT_TOP_K,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder singleton
_CROSS_ENCODER = None


def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        from config import RERANKER_MODEL
        logger.info(f"Loading cross-encoder model: {RERANKER_MODEL}")
        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL)
    return _CROSS_ENCODER


def get_embeddings_model():
    """Return Gemini embeddings model via LangChain."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def retrieve_context(
    ingestion_id: str,
    query: str,
    topic: str = None,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[List[str], List[Dict]]:
    """
    3-stage retrieval funnel:
    1. Dense retrieval (fetch top_k * 5 candidates)
    2. Coarse filtering (by section title if topic specified, with fail-safe)
    3. Cross-encoder reranking
    """
    logger.info(f"Retrieval start: ingestion_id={ingestion_id}, topic='{topic}', top_k={top_k}")

    embeddings_model = get_embeddings_model()
    search_query = topic if topic else query
    query_embedding = embeddings_model.embed_query(search_query)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = chroma_client.get_collection(name=f"ingestion_{ingestion_id}")
    except Exception as e:
        logger.error(f"Collection not found: {e}")
        return [], []

    # Stage 1: Dense retrieval
    initial_k = top_k * 5
    results = collection.query(query_embeddings=[query_embedding], n_results=initial_k)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    logger.info(f"Stage 1: Retrieved {len(documents)} candidates")

    if not documents:
        return [], []

    # Stage 2: Coarse filtering by topic
    if topic and metadatas:
        filtered = [
            (doc, meta) for doc, meta in zip(documents, metadatas)
            if topic.lower() in meta.get("section_title", "").lower()
        ]
        # Fail-safe: only apply filter if it retains enough results
        if len(filtered) >= 3:
            documents, metadatas = zip(*filtered)
            documents, metadatas = list(documents), list(metadatas)
            logger.info(f"Stage 2: Filtered to {len(documents)} chunks")
        else:
            logger.info(f"Stage 2: Filter too restrictive ({len(filtered)} matches), keeping all")

    # Skip reranking if few candidates
    if len(documents) <= top_k:
        return documents[:top_k], metadatas[:top_k]

    # Stage 3: Cross-encoder reranking
    try:
        cross_encoder = get_cross_encoder()
        pairs = [(search_query, doc) for doc in documents]
        scores = cross_encoder.predict(pairs)

        scored = sorted(zip(scores, documents, metadatas), key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for _, doc, _ in scored[:top_k]]
        reranked_meta = [meta for _, _, meta in scored[:top_k]]
        logger.info(f"Stage 3: Reranked, returning top {len(reranked_docs)}")
        return reranked_docs, reranked_meta
    except Exception as e:
        logger.error(f"Reranking failed ({e}), returning unranked top {top_k}")
        return documents[:top_k], metadatas[:top_k]

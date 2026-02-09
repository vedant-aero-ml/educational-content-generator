import os
import uuid
import time
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import GEMINI_LLM_MODEL
from models import (
    IngestResponse,
    GenerateRequest,
    GenerateResponse,
    TOCEntry,
    IngestStats,
    ParsedIntent,
    GenerateMetadata,
)
from ingest.parser import extract_pdf_content
from ingest.chunker import create_chunks, store_chunks_in_db
from agents.generator import (
    parse_intent,
    generate_mcqs,
    generate_fill_blanks,
    generate_summaries,
)
from utils.retrieval import retrieve_context
from agents.evaluation import evaluate_batch
from utils.log_handler import request_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Educational Content Generator",
    description="RAG-based system for generating MCQs, fill-in-the-blanks, and summaries from PDFs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "healthy", "service": "Educational Content Generator", "version": "1.0.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest PDF: parse -> chunk -> embed -> store in ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    ingestion_id = str(uuid.uuid4())

    with request_logger(ingestion_id):
        logger.info(f"=== INGEST START: {file.filename} | ingestion_id={ingestion_id} ===")

        # Save uploaded file
        upload_dir = "./uploaded_pdfs"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = f"{upload_dir}/{ingestion_id}_{file.filename}"

        try:
            with open(pdf_path, "wb") as f:
                content = await file.read()
                f.write(content)
            logger.info(f"Saved PDF to {pdf_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Parse PDF
        try:
            pdf_content = extract_pdf_content(pdf_path)
            logger.info(f"Parsed PDF: {len(pdf_content['pages_text'])} pages, {len(pdf_content['toc'])} TOC entries, {len(pdf_content['tables'])} tables")
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

        # Chunk
        try:
            chunks = create_chunks(pdf_content, file.filename)
            n_text = len([c for c in chunks if c["chunk_type"] == "text"])
            n_tables = len([c for c in chunks if c["chunk_type"] == "table"])
            logger.info(f"Chunking complete: {len(chunks)} total chunks ({n_text} text, {n_tables} table)")
        except Exception as e:
            logger.error(f"Failed to chunk PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to chunk PDF: {str(e)}")

        # Embed & store
        try:
            store_chunks_in_db(chunks, ingestion_id)
            logger.info(f"Stored {len(chunks)} chunks in ChromaDB collection ingestion_{ingestion_id}")
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store chunks: {str(e)}")

        toc = [TOCEntry(section=entry["section"]) for entry in pdf_content["toc"]]
        response = IngestResponse(
            ingestion_id=ingestion_id,
            file_name=file.filename,
            pages=len(pdf_content["pages_text"]),
            toc=toc,
            summary=f"Successfully ingested {len(chunks)} chunks from {len(pdf_content['pages_text'])} pages.",
            stats=IngestStats(n_chunks=len(chunks), n_tables=n_tables, n_equations=0),
        )

        logger.info(f"=== INGEST COMPLETE: {ingestion_id} | {len(chunks)} chunks from {len(pdf_content['pages_text'])} pages ===")
        return response


@app.post("/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """Generate learning content: parse intent -> retrieve -> generate -> evaluate."""
    request_id = str(uuid.uuid4())

    with request_logger(request_id):
        logger.info(f"=== GENERATE START: request_id={request_id} | ingestion_id={request.ingestion_id} ===")
        logger.info(f"User prompt: {request.user_prompt}")

        # Step 1: Parse intent
        try:
            intent = parse_intent(request.user_prompt)
            logger.info(f"Parsed intent: mode={intent['mode']}, topic={intent.get('topic')}, n={intent['n']}, difficulty={intent['difficulty']}")
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to parse intent: {str(e)}")

        # Step 2: Retrieve context
        retrieval_start = time.time()
        try:
            context_chunks, context_metadata = retrieve_context(
                ingestion_id=request.ingestion_id,
                query=request.user_prompt,
                topic=intent.get("topic"),
                top_k=5,
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve context: {str(e)}")
        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(context_chunks)} chunks in {retrieval_time:.0f}ms")

        if not context_chunks:
            logger.error(f"No content found for ingestion_id: {request.ingestion_id}")
            raise HTTPException(status_code=404, detail=f"No content found for ingestion_id: {request.ingestion_id}")

        # Step 3: Generate content
        generation_start = time.time()
        try:
            if intent["mode"] == "mcq":
                questions = generate_mcqs(context_chunks, context_metadata, intent)
            elif intent["mode"] == "fill_blank":
                questions = generate_fill_blanks(context_chunks, context_metadata, intent)
            elif intent["mode"] in ["summary", "summary_per_section"]:
                questions = generate_summaries(context_chunks, context_metadata, intent)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown mode: {intent['mode']}")
        except HTTPException:
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Generation failed: {error_msg}")
            if "404" in error_msg:
                raise HTTPException(status_code=502, detail=f"LLM model not found: {error_msg}")
            elif "429" in error_msg:
                raise HTTPException(status_code=429, detail=f"LLM rate limit exceeded: {error_msg}")
            else:
                raise HTTPException(status_code=500, detail=f"Content generation failed: {error_msg}")
        generation_time = (time.time() - generation_start) * 1000
        logger.info(f"Generated {len(questions)} items in {generation_time:.0f}ms")

        # Step 4: Evaluate with RAG Triad (single batched call)
        eval_start = time.time()
        context_text = "\n\n".join(context_chunks)
        try:
            evaluations = evaluate_batch(
                questions=questions,
                context=context_text,
                topic=intent.get("topic", "the document"),
            )
            for q, eval_result in zip(questions, evaluations):
                q["evaluator"] = eval_result
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Evaluation failed: {error_msg}")
            if "429" in error_msg:
                raise HTTPException(status_code=429, detail=f"LLM rate limit exceeded during evaluation: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {error_msg}")
        eval_time = (time.time() - eval_start) * 1000
        logger.info(f"Evaluation complete in {eval_time:.0f}ms")

        response = GenerateResponse(
            request_id=request_id,
            generated_learning_content=questions,
            metadata=GenerateMetadata(
                parsed_intent=ParsedIntent(**intent),
                retrieval_time_ms=int(retrieval_time),
                generation_time_ms=int(generation_time),
                model=GEMINI_LLM_MODEL,
            ),
        )

        logger.info(f"=== GENERATE COMPLETE: {request_id} | {len(questions)} items, retrieval={retrieval_time:.0f}ms, generation={generation_time:.0f}ms, eval={eval_time:.0f}ms ===")
        return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

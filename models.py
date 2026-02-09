from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class TOCEntry(BaseModel):
    section: str


class IngestStats(BaseModel):
    n_chunks: int
    n_tables: int
    n_equations: int


class IngestResponse(BaseModel):
    ingestion_id: str
    file_name: str
    pages: int
    toc: List[TOCEntry]
    summary: str
    stats: IngestStats


class GenerateRequest(BaseModel):
    ingestion_id: str
    user_prompt: str = Field(..., description="Free-form text describing what to generate")


class MCQQuestion(BaseModel):
    id: str
    question: str
    options: Dict[str, str]
    correct: str
    explanation: str
    difficulty: str
    evaluator: Optional[Dict[str, Any]] = None


class FillBlankQuestion(BaseModel):
    id: str
    question: str
    correct: str
    explanation: str
    difficulty: str
    evaluator: Optional[Dict[str, Any]] = None


class Summary(BaseModel):
    id: str
    summary: str
    section: str
    evaluator: Optional[Dict[str, Any]] = None


class ParsedIntent(BaseModel):
    mode: str
    topic: Optional[str]
    n: int
    difficulty: str
    global_scope: bool


class GenerateMetadata(BaseModel):
    parsed_intent: ParsedIntent
    retrieval_time_ms: int
    generation_time_ms: int
    model: str


class GenerateResponse(BaseModel):
    request_id: str
    generated_learning_content: List[Dict[str, Any]]
    metadata: GenerateMetadata

import uuid
import json
import logging
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, GEMINI_LLM_MODEL, GENERATION_TEMPERATURE
from agents.prompts import (
    INTENT_PARSER_PROMPT,
    MCQ_GENERATOR_PROMPT,
    FILL_BLANK_PROMPT,
    SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


def _parse_llm_json(response) -> any:
    """Extract and parse JSON from an LLM response, stripping markdown wrappers."""
    content = response.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()
    return json.loads(content)


def _get_llm(temperature=None):
    return ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature if temperature is not None else GENERATION_TEMPERATURE,
    )


def _format_context(context_chunks: List[str], context_metadata: List[Dict]) -> str:
    return "\n\n".join(
        f"[Page {meta.get('page', 'unknown')}, Section: {meta.get('section_title', 'unknown')}]\n{text}"
        for text, meta in zip(context_chunks, context_metadata)
    )


def _unwrap_list(data, key: str) -> list:
    """Handle LLM returning either a list or a dict with a nested list."""
    if isinstance(data, dict) and key in data:
        return data[key]
    return data


def parse_intent(user_prompt: str) -> Dict:
    logger.info(f"Parsing intent: '{user_prompt}'")

    llm = _get_llm(temperature=0.0)
    prompt = INTENT_PARSER_PROMPT.format(user_prompt=user_prompt)

    try:
        response = llm.invoke(prompt)
        intent = _parse_llm_json(response)
        logger.info(f"Parsed intent: {intent}")
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}, using fallback")
        intent = {
            "mode": "mcq",
            "topic": None,
            "n": None,
            "difficulty": None,
            "global_scope": True,
        }

    if intent.get("n") is None:
        intent["n"] = 5 if intent.get("topic") else 1
    if intent.get("difficulty") is None:
        intent["difficulty"] = "mixed"

    logger.info(f"Final intent: {intent}")
    return intent


def generate_mcqs(
    context_chunks: List[str],
    context_metadata: List[Dict],
    intent: Dict,
) -> List[Dict]:
    if not context_chunks:
        return []

    logger.info(f"Generating {intent['n']} MCQs")

    llm = _get_llm()
    formatted_context = _format_context(context_chunks, context_metadata)

    prompt = MCQ_GENERATOR_PROMPT.format(
        retrieved_chunks=formatted_context,
        num_questions=intent["n"],
        difficulty=intent["difficulty"],
    )

    response = llm.invoke(prompt)
    questions = _unwrap_list(_parse_llm_json(response), "questions")
    logger.info(f"Generated {len(questions)} MCQs")

    for q in questions:
        q["id"] = str(uuid.uuid4())

    return questions


def generate_fill_blanks(
    context_chunks: List[str],
    context_metadata: List[Dict],
    intent: Dict,
) -> List[Dict]:
    if not context_chunks:
        return []

    logger.info(f"Generating {intent['n']} fill-blank questions")

    llm = _get_llm()
    formatted_context = _format_context(context_chunks, context_metadata)

    prompt = FILL_BLANK_PROMPT.format(
        retrieved_chunks=formatted_context,
        num_questions=intent["n"],
        difficulty=intent["difficulty"],
    )

    response = llm.invoke(prompt)
    questions = _unwrap_list(_parse_llm_json(response), "questions")
    logger.info(f"Generated {len(questions)} fill-blank questions")

    for q in questions:
        q["id"] = str(uuid.uuid4())

    return questions


def generate_summaries(
    context_chunks: List[str],
    context_metadata: List[Dict],
    intent: Dict,
) -> List[Dict]:
    llm = _get_llm()
    summaries = []

    if intent["mode"] == "summary_per_section":
        sections = {}
        for chunk, meta in zip(context_chunks, context_metadata):
            section = meta.get("section_title", "Unknown")
            sections.setdefault(section, []).append(chunk)

        for section, chunks in sections.items():
            prompt = SUMMARY_PROMPT.format(
                section_text="\n\n".join(chunks),
                section_title=section,
            )
            response = llm.invoke(prompt)
            summary_obj = _parse_llm_json(response)
            summary_obj["id"] = str(uuid.uuid4())
            summaries.append(summary_obj)
    else:
        prompt = SUMMARY_PROMPT.format(
            section_text="\n\n".join(context_chunks),
            section_title=intent.get("topic", "Document"),
        )
        response = llm.invoke(prompt)
        summary_obj = _parse_llm_json(response)
        summary_obj["id"] = str(uuid.uuid4())
        summaries.append(summary_obj)

    return summaries

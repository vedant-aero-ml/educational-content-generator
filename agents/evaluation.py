import json
import logging
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, GEMINI_JUDGE_MODEL, JUDGE_TEMPERATURE
from agents.prompts import RAG_TRIAD_PROMPT

logger = logging.getLogger(__name__)


def _parse_llm_json(response) -> any:
    """Extract and parse JSON from an LLM response, stripping markdown wrappers."""
    content = response.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()
    return json.loads(content)


def evaluate_batch(
    questions: List[Dict],
    context: str,
    topic: str = "the document",
) -> List[Dict]:
    """Evaluate all questions in a single LLM call."""
    if not questions:
        return []

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_JUDGE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=JUDGE_TEMPERATURE,
    )

    questions_json = json.dumps(questions, indent=2)
    context_truncated = context[:4000]

    prompt = RAG_TRIAD_PROMPT.format(
        context=context_truncated,
        topic=topic,
        questions_json=questions_json,
    )

    response = llm.invoke(prompt)
    evaluations = _parse_llm_json(response)

    if not isinstance(evaluations, list):
        evaluations = [evaluations]

    # Pad or trim to match question count
    while len(evaluations) < len(questions):
        evaluations.append({
            "quality_score": 0.0,
            "is_supported": False,
            "issues": ["evaluation_missing"],
        })

    return evaluations[:len(questions)]

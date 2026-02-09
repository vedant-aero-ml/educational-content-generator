INTENT_PARSER_PROMPT = """You are an intent parser. Extract structured information from user prompts.

User prompt: "{user_prompt}"

Extract and return ONLY valid JSON:
{{
  "mode": "mcq" | "fill_blank" | "summary" | "summary_per_section",
  "topic": "<topic string or null if not specified>",
  "n": <number of items requested, or null>,
  "difficulty": "easy" | "medium" | "hard" | "mixed" | null,
  "global_scope": true | false
}}

Rules:
- "mcq", "multiple choice", "questions" -> mode: "mcq"
- "fill in the blank", "cloze" -> mode: "fill_blank"
- "summary" -> mode: "summary"
- "for all sections separately" -> mode: "summary_per_section"
- Extract numbers like "5", "10" as n
- If topic not specified, set topic: null, global_scope: true
- If difficulty not mentioned, set null

Return ONLY the JSON, no other text."""


MCQ_GENERATOR_PROMPT = """You are an educational content generator. Based ONLY on the context below, generate {num_questions} multiple-choice questions.

CONTEXT FROM TEXTBOOK:
{retrieved_chunks}

REQUIREMENTS:
1. Questions must be DIRECTLY answerable from the context (extractive approach)
2. Each question has:
   - Clear question text
   - 4 options (A, B, C, D)
   - Exactly one correct answer
   - 3 plausible distractors (common misconceptions, not random)
3. Include 2-sentence explanation citing the context
4. Difficulty level: {difficulty}
5. For tables: ask about values, trends, relationships
6. For equations: ask about solving, identifying variables, or concepts

OUTPUT ONLY VALID JSON (array of question objects):
[
  {{
    "question": "What is the solution to 2x + 4 = 12?",
    "options": {{
      "A": "x = 3",
      "B": "x = 4",
      "C": "x = 5",
      "D": "x = 6"
    }},
    "correct": "B",
    "explanation": "According to the context, to solve 2x + 4 = 12, subtract 4 from both sides to get 2x = 8, then divide by 2 to get x = 4.",
    "difficulty": "{difficulty}"
  }}
]

Generate exactly {num_questions} questions. Return ONLY the JSON array."""


FILL_BLANK_PROMPT = """You are an educational content generator. Based ONLY on the context below, generate {num_questions} fill-in-the-blank questions.

CONTEXT FROM TEXTBOOK:
{retrieved_chunks}

REQUIREMENTS:
1. Create sentences from the context with ONE key term/number removed
2. Use ____ to mark the blank
3. The correct answer must appear verbatim in the context
4. Include 2-sentence explanation
5. Difficulty: {difficulty}

OUTPUT ONLY VALID JSON:
[
  {{
    "question": "The quadratic formula is x = ____",
    "correct": "(-b +/- sqrt(b^2-4ac)) / 2a",
    "explanation": "From the context, the quadratic formula solves ax^2 + bx + c = 0 and is given by x = (-b +/- sqrt(b^2-4ac)) / 2a.",
    "difficulty": "{difficulty}"
  }}
]

Generate exactly {num_questions} questions. Return ONLY the JSON array."""


SUMMARY_PROMPT = """You are an educational content summarizer. Based on the context below, create a concise summary.

CONTEXT:
{section_text}

REQUIREMENTS:
1. 3-5 sentences covering main concepts
2. Include key definitions and formulas mentioned
3. Clear and student-friendly language

OUTPUT ONLY VALID JSON:
{{
  "summary": "...",
  "section": "{section_title}"
}}"""


RAG_TRIAD_PROMPT = """You are an educational content quality evaluator. Assess each generated question using the RAG Triad framework.

RETRIEVED CONTEXT:
{context}

GENERATED QUESTIONS:
Topic: {topic}
{questions_json}

For EACH question, evaluate on three dimensions (0.0 to 1.0 scale):

1. CONTEXT RELEVANCE: Does the retrieved context contain information relevant to the topic "{topic}"?
   - 1.0 = Highly relevant, directly addresses topic
   - 0.5 = Partially relevant, tangential information
   - 0.0 = Irrelevant, no connection to topic

2. GROUNDEDNESS: Is the question and answer derived ONLY from the retrieved context?
   - 1.0 = Fully grounded, all facts from context
   - 0.5 = Mostly grounded with minor inferences
   - 0.0 = Not grounded, fabricated information

3. ANSWER RELEVANCE: Is the question well-formed, clear, and solvable from the context?
   - 1.0 = Excellent question, clear and educational
   - 0.5 = Acceptable but could be improved
   - 0.0 = Poor question, confusing or unsolvable

Return a JSON array with one evaluation object per question, in the same order. Each object must have:
- context_relevance_score, groundedness_score, answer_relevance_score (floats)
- quality_score (average of the three)
- is_supported (true if quality_score > 0.6)
- reasoning (2-3 sentence explanation)
- issues (list of specific problems, empty if none)

Return ONLY the JSON array."""

import os
from dotenv import load_dotenv

load_dotenv()

# Google AI configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Chunking configuration
MAX_CHUNK_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 50

# Retrieval configuration
DEFAULT_TOP_K = 5

# Generation configuration
GEMINI_LLM_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_TEMPERATURE = 0.2

# Judge model configuration
GEMINI_JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_TEMPERATURE = 0.0

# Reranker configuration
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Logging configuration
LOG_DIR = "./logs"


import os
import logging
logger = logging.getLogger(__name__)# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")


LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY is missing from environment or .env file.")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

GENERATION_CONFIG = {
    "max_length": 512,
    "temperature": 0,
    "top_p": 0.95,
    "repetition_penalty": 1.15
}
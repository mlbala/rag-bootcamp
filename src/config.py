from pathlib import Path

from dotenv import load_dotenv

# Resolves to the root of RAG-Bootcamp/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root, whatever the current working directory is
load_dotenv(PROJECT_ROOT / ".env")

# Top-level data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_PARSING_DIR = DATA_DIR / "data_parsing"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Raw data subdirectories
RAW_DB_DIR = RAW_DATA_DIR / "databases"
RAW_JSON_DIR = RAW_DATA_DIR / "json_files"
RAW_PDF_DIR = RAW_DATA_DIR / "pdf"
RAW_STRUCTURED_DIR = RAW_DATA_DIR / "structured_files"
RAW_TEXT_DIR = RAW_DATA_DIR / "text"
RAW_WORD_DIR = RAW_DATA_DIR / "word_files"

# Root-level vector store directory (shared by notebooks/, src/ and apps/)
VECTOR_STORE_DIR = PROJECT_ROOT / "vectorstore"
CHROMA_PATH = VECTOR_STORE_DIR / "chroma_db"
FAISS_PATH = VECTOR_STORE_DIR / "faiss_index"

# Make sure the key directories exist
for directory in [DATA_PARSING_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, VECTOR_STORE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

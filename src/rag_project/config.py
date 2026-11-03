from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent.parent 

DATA_DIR = BASE_DIR / "data"  
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"      
PROCESSED_DIR = DATA_DIR / "processed"   
INDEX_DIR = DATA_DIR / "indexes"     
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"     

BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNK_MAP_PATH = INDEX_DIR / "chunks.json"

# Create folders if they don't exist
for folder in (RAW_PDF_DIR, PROCESSED_DIR, INDEX_DIR):
    folder.mkdir(parents=True, exist_ok=True)


CHUNK_SIZE = 1000
CHUNK_OVERLAP=200

EMBEDDING_MODEL_NAME= "BAAI/bge-small-en-v1.5"

RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL_NAME="openai/gpt-oss-120b"
BM25_TOP_K = 20 
DENSE_TOP_K = 20
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_K = 5
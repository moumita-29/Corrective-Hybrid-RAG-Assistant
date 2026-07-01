"""Central configuration for the RAG application.

All model names, paths, and tunable parameters live here.
Change these values to swap models or adjust behavior.
"""

# --- Embedding Model (Swapped for speed) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- LLM (Groq API) ---
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 5

# --- FAISS ---
FAISS_INDEX_PATH = "faiss_index"

# --- Reranker (Swapped for a much faster CPU cross-encoder) ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Hybrid Search ---
BM25_K = 10   # Candidates from each retriever before fusion
RRF_K = 60    # RRF constant (higher = less weight on top ranks)

# --- Memory ---
MEMORY_WINDOW = 5   # Number of past conversation turns to remember

# --- Fallback Mechanism ---
FALLBACK_THRESHOLD = 40.0   # Minimum confidence score (%) required to use RAG context

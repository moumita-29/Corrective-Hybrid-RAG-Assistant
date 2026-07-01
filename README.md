# Hybrid Search RAG Assistant

## Project Overview
The Hybrid Search RAG (Retrieval-Augmented Generation) Assistant is a powerful, locally-run AI application designed to intelligently query and chat with your PDF documents. It uses a combination of semantic search (FAISS) and keyword search (BM25) merged via Reciprocal Rank Fusion, followed by a Cross-Encoder reranking step to retrieve the most highly relevant context for your questions. When the system detects low-quality retrieval, it employs a smart fallback mechanism to rely on general AI knowledge.

## Features
- **Multi-PDF Upload**: Drag and drop multiple PDF documents simultaneously.
- **Hybrid Search**: Combines Dense (FAISS) and Sparse (BM25) vector retrieval for superior accuracy.
- **Cross-Encoder Reranking**: Re-evaluates search results for precise, context-aware answers.
- **Smart Fallback Mechanism**: Automatically switches to the LLM's general knowledge if document context is irrelevant or low confidence.
- **Source Citations**: Transparently displays exactly which document and page the AI used to formulate its response.
- **Retrieval Inspector**: A dedicated debug tab to view exact retrieved chunks and their scores.
- **Evaluation Dashboard**: Tracks query history, average retrieval time, and confidence metrics.

## Architecture
The application follows a standard RAG pipeline enhanced with advanced retrieval techniques. 

```mermaid
graph TD
    A[User PDF Documents] -->|Chunking| B(LangChain PyPDFLoader)
    B --> C[(FAISS Vector Store)]
    B --> D[(BM25 Sparse Index)]
    
    E[User Query] --> F{Hybrid Retrieval}
    F -->|Semantic Search| C
    F -->|Keyword Search| D
    
    C --> G[Dense Candidates]
    D --> H[Sparse Candidates]
    
    G --> I(Reciprocal Rank Fusion - RRF)
    H --> I
    
    I --> J[Fused Candidates]
    J --> K(Cross-Encoder Reranker)
    
    K --> Q[Document Grader]
    Q --> R[Knowledge Refinement]
    
    R --> L{Confidence Score Check}
    
    L -- Low Score --> S[Query Rewrite]
    S --> T[Second Retrieval]
    T --> L
    
    L -- High Score --> N[Context Injected into Prompt]
    L -- Repeated Failure --> M[Fallback: General Knowledge]
    
    M --> O[Qwen2.5 LLM]
    N --> O
    
    O --> U{Answer Verification}
    U -- FAIL --> O
    U -- PASS --> P[Streamed Response to User]
```

1. **Document Processing**: PDFs are loaded, split into overlapping chunks, and vectorized.
2. **Hybrid Retrieval**:
   - FAISS retrieves chunks based on semantic similarity.
   - BM25 retrieves chunks based on exact keyword matches.
3. **Fusion & Reranking**: Results are merged using Reciprocal Rank Fusion (RRF) and scored by a Cross-Encoder model.
4. **Generation**: The highest-scoring chunks are injected into a prompt for the Hugging Face LLM (Qwen2.5) to stream a response. If scores are too low, the pipeline falls back to a general knowledge query.

## Tech Stack
- **Frontend / Framework**: [Streamlit](https://streamlit.io/)
- **LLM**: Qwen2.5-7B-Instruct (via Hugging Face Inference API)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Reranker**: sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector Database**: FAISS
- **Keyword Search**: rank_bm25
- **Document Loading & Chunking**: LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`)

## Installation

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd hybrid-rag
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your environment**:
   Create a `.env` file in the root directory and add your Hugging Face token:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

## How to Run

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to the URL provided in the terminal (usually `http://localhost:8501`).
3. Upload one or more PDFs using the sidebar or main uploader.
4. Start chatting!


## Future Improvements
- **Agentic Workflows**: Introduce multi-step reasoning capabilities for complex queries.
- **Additional File Formats**: Support for DOCX, TXT, and Markdown files.
- **Persistent Vector Store**: Save and load FAISS indices across sessions to avoid re-processing large documents.
- **Customizable Prompts**: Allow users to tweak the system prompt via the UI for specialized tasks.

## Author
**Moumita Paul**  
*IIIT Lucknow*

"""Hybrid retriever: FAISS + BM25 → Reciprocal Rank Fusion → Reranking.

This is the core of the hybrid search pipeline:
1. FAISS (semantic search) — finds documents with similar meaning
2. BM25 (keyword search)  — finds documents with matching terms
3. Reciprocal Rank Fusion — merges both ranked lists fairly
4. Cross-encoder reranker — rescores the merged candidates
"""

from rag.vector_store import similarity_search
from rag.reranker import rerank
from config import TOP_K, BM25_K, RRF_K


def is_comparison_query(query):
    """Check if query is asking to compare or summarize multiple documents."""
    query_lower = query.lower()
    keywords = [
        "compare", "difference", "differences", "summarize", "between", "both", 
        "versus", "vs", "multiple", "all documents", "all uploaded", "across all", 
        "each document", "every document", "various documents", "overall"
    ]
    return any(keyword in query_lower for keyword in keywords)

def hybrid_search(query, vector_store, bm25_index, k=TOP_K):
    """Perform hybrid search combining semantic and keyword retrieval.

    Args:
        query: The user's question.
        vector_store: FAISS vector store instance.
        bm25_index: BM25Index instance (can be None for FAISS-only).
        k: Number of final results to return after reranking.

    Returns:
        List of (Document, reranker_score) tuples.
    """
    # Step 1: Get candidates from both retrievers
    # For comparison queries, we need a massive initial candidate pool so one document 
    # doesn't crowd out the others before the round-robin diversity filter runs.
    search_k = BM25_K * 5 if is_comparison_query(query) else BM25_K
    
    faiss_results = similarity_search(vector_store, query, k=search_k)
    bm25_results = bm25_index.search(query, k=search_k) if bm25_index else []

    # Step 2: Merge with Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(faiss_results, bm25_results, k=RRF_K)

    if not fused:
        return []

    # Step 3: Candidate Selection & Reranking
    if is_comparison_query(query):
        from collections import defaultdict
        
        # 3A: Diverse selection for the Reranker Pool (ensure all PDFs get a chance to be reranked)
        docs_by_source_fused = defaultdict(list)
        for doc, score in fused:
            source = doc.metadata.get("source", "Unknown")
            docs_by_source_fused[source].append(doc)
            
        fused_docs = []
        while len(fused_docs) < 25 and docs_by_source_fused:
            sources_to_remove = []
            for source, docs in docs_by_source_fused.items():
                if docs and len(fused_docs) < 25:
                    fused_docs.append(docs.pop(0))
                if not docs:
                    sources_to_remove.append(source)
            for source in sources_to_remove:
                del docs_by_source_fused[source]
        
        total_indexed_pdfs = len(set(doc.metadata.get("source", "Unknown") for doc, _ in fused))
        pdfs_before_reranking = len(set(doc.metadata.get("source", "Unknown") for doc in fused_docs))
        
        reranked = rerank(query, fused_docs, top_k=25)
        
        # Step 4: Diverse document selection (Post-Reranking)
        from collections import defaultdict
        docs_by_source = defaultdict(list)
        for doc, score in reranked:
            source = doc.metadata.get("source", "Unknown")
            docs_by_source[source].append((doc, score))
            
        final_reranked = []
        
        # Round-robin selection across different sources
        while len(final_reranked) < k and docs_by_source:
            sources_to_remove = []
            for source, docs in docs_by_source.items():
                if docs and len(final_reranked) < k:
                    final_reranked.append(docs.pop(0))
                if not docs:
                    sources_to_remove.append(source)
            for source in sources_to_remove:
                del docs_by_source[source]
                
        # Do NOT re-sort by score. We want to preserve the interleaved round-robin 
        # ordering (Doc A, Doc B, Doc A, Doc B) so that the LLM sees diverse documents 
        # early in its context window and doesn't get overwhelmed by one document at the top.
        
        pdfs_after_reranking = len(set(doc.metadata.get("source", "Unknown") for doc, _ in final_reranked))
        
        comp_debug = {
            "total_indexed_pdfs": total_indexed_pdfs,
            "pdfs_before_reranking": pdfs_before_reranking,
            "pdfs_after_reranking": pdfs_after_reranking
        }
        return final_reranked, comp_debug
    else:
        # Standard flow: slice to 8 candidates and rerank to top k
        fused_docs = [doc for doc, _ in fused[:8]]
        return rerank(query, fused_docs, top_k=k)


def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    """Merge two ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF score = Σ  1 / (k + rank)  for each list the document appears in.

    This balances semantic and keyword relevance without needing to
    normalize scores across the two very different retrieval methods.
    A document ranked highly by both methods gets a higher fused score.

    Args:
        faiss_results: List of (Document, score) from FAISS.
        bm25_results: List of (Document, score) from BM25.
        k: RRF constant (default 60). Higher values reduce the
           impact of high rankings from a single source.

    Returns:
        List of (Document, rrf_score) tuples sorted by fused score.
    """
    doc_scores = {}  # content_hash → accumulated RRF score
    doc_map = {}     # content_hash → Document object

    for rank, (doc, score) in enumerate(faiss_results):
        key = hash(doc.page_content)
        if "faiss_score" not in doc.metadata:
            doc.metadata["faiss_score"] = float(score)
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    for rank, (doc, score) in enumerate(bm25_results):
        key = hash(doc.page_content)
        if "bm25_score" not in doc.metadata:
            doc.metadata["bm25_score"] = float(score)
        doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc_map[key] = doc

    sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in sorted_results]

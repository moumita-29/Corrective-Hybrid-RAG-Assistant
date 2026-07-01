"""Confidence score calculation from retrieval scores.

Supports two score types:
- "l2": FAISS L2 distances (Phase 2, used as fallback)
- "reranker": Cross-encoder logits (Phase 3, used with hybrid search)
"""

import math


def compute_confidence(chunks):
    """Compute a unified confidence score combining multiple retrieval and grading signals.

    Combines:
    - Dense retrieval score (FAISS L2 distance -> cosine similarity)
    - BM25 score (normalized to ~0-1)
    - Cross Encoder score (logits -> probability)
    - Document Grader score (0-100 -> 0-1)

    Args:
        chunks: List of chunk dictionaries from refine_knowledge/retrieve pipeline.
                Expected to have keys: "doc" (with metadata scores), "score" (CrossEncoder), "grade" (with confidence).

    Returns:
        Tuple of (score, label, emoji).
    """
    if not chunks:
        return 0.0, "Low", "🔴"

    chunk_scores = []
    
    for chunk in chunks:
        doc = chunk["doc"]
        ce_score = chunk["score"]
        grade_conf = chunk["grade"].get("confidence", 0)

        # 1. FAISS L2 -> Normalized (0 to 1). Lower L2 is better. 
        # Using typical cosine similarity approx: max(0, 1 - L2^2 / 2)
        faiss_raw = doc.metadata.get("faiss_score", 1.0) # default to neutral distance if missing
        faiss_norm = max(0.0, 1.0 - (faiss_raw ** 2) / 2.0)
        
        # 2. BM25 -> Normalized (0 to 1). BM25 is unbounded, but typically 0-10 for short chunks.
        bm25_raw = doc.metadata.get("bm25_score", 0.0)
        bm25_norm = min(1.0, bm25_raw / 10.0) # Simple heuristic normalization
        
        # 3. Cross Encoder -> Sigmoid probability (0 to 1)
        ce_norm = 1.0 / (1.0 + math.exp(-ce_score))
        
        # 4. Document Grader -> (0 to 1)
        grade_norm = grade_conf / 100.0
        
        # Average the four normalized signals for this chunk
        combined = (faiss_norm + bm25_norm + ce_norm + grade_norm) / 4.0
        chunk_scores.append(combined)
        
    avg_confidence = (sum(chunk_scores) / len(chunk_scores)) * 100.0

    if avg_confidence >= 70:
        return round(avg_confidence, 1), "High", "🟢"
    elif avg_confidence >= 40:
        return round(avg_confidence, 1), "Medium", "🟡"
    else:
        return round(avg_confidence, 1), "Low", "🔴"

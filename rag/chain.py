"""RAG chain — retrieves context and generates answers with source citations.

Combines the hybrid retriever with the LLM to answer
questions grounded in the uploaded documents.
Phase 4: Added conversation memory and retrieval logging.
"""

import os
import time
import logging
from rag.retriever import hybrid_search
from rag.llm import get_llm
from rag.confidence import compute_confidence
from config import TOP_K

logger = logging.getLogger(__name__)

# Reused from the original project
RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip()

FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. When you don't know something, "
    "be honest about it. Provide clear, concise, and accurate responses."
)


# ---------- Retrieval ----------


def _run_retrieval_pass(search_query, vector_store, bm25_index, k):
    """Helper to run one pass of retrieval, reranking, and grading."""
    from rag.document_grader import refine_knowledge
    res = hybrid_search(search_query, vector_store, bm25_index, k=k)
    
    comp_debug = None
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        res, comp_debug = res
        
    if not res:
        return "Failure", [], [], comp_debug
        
    ref = refine_knowledge(search_query, res)
    valid = ref["correct_chunks"] + ref["ambiguous_chunks"]
    all_c = ref["correct_chunks"] + ref["ambiguous_chunks"] + ref["removed_chunks"]
    
    if comp_debug is not None:
        pdfs_after_grading = len(set(chunk["doc"].metadata.get("source", "Unknown") for chunk in valid))
        comp_debug["pdfs_removed_by_grading"] = comp_debug["pdfs_after_reranking"] - pdfs_after_grading
        
    return ref["status"], valid, all_c, comp_debug

def _format_debug_chunks(chunks):
    g_info = []
    for chunk in chunks:
        doc = chunk["doc"]
        g_info.append({
            "score": float(chunk["score"]),
            "source": os.path.basename(doc.metadata.get("source", "Unknown")).replace("temp_", ""),
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content[:300] + "...",
            "grade": chunk["grade"]
        })
    return g_info

def retrieve(query, vector_store, bm25_index=None, k=TOP_K):
    """Retrieve relevant documents using hybrid search + reranking and grading."""
    start_time = time.time()
    
    if vector_store is None:
        logger.info(f"Retrieval skipped: No vector store available for query '{query}'")
        return [], [], (0.0, "Low", "🔴"), {}

    status, final_valid, final_all_c, comp_debug = _run_retrieval_pass(query, vector_store, bm25_index, k)

    # --- Build Outputs ---
    context_parts, sources = [], []
    for chunk in final_valid:
        doc = chunk["doc"]
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", None)
        page_str = f", Page: {page_num + 1}" if page_num is not None else ""
        
        # Include metadata directly in the context part so the LLM can cite it
        context_parts.append(f"[Document: {source_name}{page_str}]\n{doc.page_content}")
        
        sources.append({
            "content": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": page_num + 1 if page_num is not None else "N/A",
            "grade": chunk["grade"]["label"]
        })

    if status == "Failure" or not final_valid:
        confidence = (0.0, "Low", "🔴")
    else:
        confidence = compute_confidence(final_valid)
    
    elapsed = time.time() - start_time
    logger.info(
        f"Retrieval complete in {elapsed:.2f}s | "
        f"Found {len(final_valid)} valid chunks | "
        f"Confidence: {confidence[0]}% ({confidence[1]}) | "
        f"Query: '{query}'"
    )
    
    debug_info = {
        "query": query,
        "time_taken_sec": elapsed,
        "num_chunks_retrieved": len(final_valid),
        "confidence_score": confidence[0],
        "chunks": _format_debug_chunks(final_all_c),
        "rewritten_query": None,
        "original_pass": None,
        "comp_debug": comp_debug
    }
    
    return context_parts, sources, confidence, debug_info


# ---------- Generation (with Memory) ----------


def _build_rag_prompt(query, context_parts):
    """Build the RAG prompt from query and context chunks."""
    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Provide a comprehensive answer based on the context above. "
        f"If the context contains source filenames or page numbers, cite them naturally in your response (e.g., 'According to document.pdf on page 4...'). "
        f"Do not hallucinate page numbers. If they are not provided, do not guess."
        f"If the context doesn't contain enough information, say so clearly."
    )


def _build_messages(system_prompt, query, context_parts=None, chat_history=None):
    """Construct message list including system prompt, history, and current query."""
    messages = [("system", system_prompt)]
    
    if chat_history:
        for turn in chat_history:
            messages.append(("human", turn["question"]))
            messages.append(("ai", turn["answer"]))
            
    if context_parts:
        prompt = _build_rag_prompt(query, context_parts)
    else:
        prompt = query
        
    messages.append(("human", prompt))
    return messages


def _stream_response(messages):
    """Helper to stream LLM response tokens."""
    try:
        llm = get_llm()
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        logger.error(f"LLM Error: {str(e)}")
        yield f"\n\n⚠️ **LLM Error**: {str(e)}. Please check your API key or network connection."

def generate_stream(query, context_parts, chat_history=None):
    """Stream LLM response tokens for a RAG query, using chat history."""
    messages = _build_messages(RAG_SYSTEM_PROMPT, query, context_parts, chat_history)
    yield from _stream_response(messages)


def fallback_stream(query, chat_history=None):
    """Stream a fallback response when no documents match, using chat history."""
    messages = _build_messages(FALLBACK_SYSTEM_PROMPT, query, None, chat_history)
    yield from _stream_response(messages)


def rewrite_query(query, chat_history=None):
    """Corrective RAG: Rewrite a poorly performing query to improve retrieval."""
    try:
        llm = get_llm()
        prompt = (
            "You are an expert at optimizing search queries for a document retrieval system.\n"
            f"The user asked: '{query}'\n"
            "This query resulted in a retrieval failure or low quality results.\n"
            "Rewrite the query to:\n"
            "- Preserve the original user intent.\n"
            "- Improve retrieval specificity.\n"
            "- Remove any ambiguity.\n"
            "Return ONLY the rewritten query text, with no preamble, quotes, or explanations."
        )
        messages = [("system", "You are a search query rewriting assistant.")]
        
        if chat_history:
            for turn in chat_history:
                messages.append(("human", turn["question"]))
                messages.append(("ai", turn["answer"]))
                
        messages.append(("human", prompt))
        response = llm.invoke(messages)
        return response.content.strip().strip("'\"")
    except Exception as e:
        logger.error(f"LLM Error during query rewrite: {str(e)}")
        # Graceful fallback: return the original query
        return query


def route_query(query, chat_history=None):
    """Determine if a query can be answered using only chat history.
    
    Returns True if it's a follow-up that requires no new retrieval.
    """
    if not chat_history:
        return False
        
    try:
        import json
        llm = get_llm()
        prompt = (
            "You are an expert intent classifier for a document retrieval system.\n"
            f"The user asked: '{query}'\n\n"
            "Evaluate if this query can be fully and accurately answered using ONLY the context provided in the conversation history.\n"
            "For example, 'tell me more', 'summarize that', or 'why?' are follow-up questions that often don't need new documents.\n"
            "If it can be answered using history, output a strict JSON object: {\"action\": \"HISTORY_ONLY\"}\n"
            "If it requires looking up new facts or documents, output a strict JSON object: {\"action\": \"NEEDS_RETRIEVAL\"}\n"
            "Output ONLY the JSON object."
        )
        messages = [("system", "You are an expert intent classifier. You always output valid JSON and nothing else.")]
        
        for turn in chat_history[-3:]: # Limit history context for routing to save tokens
            messages.append(("human", turn["question"]))
            messages.append(("ai", turn["answer"]))
            
        messages.append(("human", prompt))
        response = llm.invoke(messages)
        
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        result = json.loads(content.strip())
        return result.get("action") == "HISTORY_ONLY"
    except Exception as e:
        logger.error(f"LLM Error during intent routing: {str(e)}")
        # Graceful fallback: default to retrieval
        return False


# ---------- Non-streaming (kept for compatibility/testing) ----------


def ask(query, vector_store, bm25_index=None, k=TOP_K, chat_history=None):
    """Non-streaming RAG query."""
    context_parts, sources, confidence, debug_info = retrieve(query, vector_store, bm25_index, k)

    if not context_parts:
        return _fallback(query, chat_history), [], (0.0, "Low", "🔴"), {}

    llm = get_llm()
    messages = _build_messages(RAG_SYSTEM_PROMPT, query, context_parts, chat_history)
    response = llm.invoke(messages)
    return response.content, sources, confidence, debug_info


def _fallback(query, chat_history=None):
    """Answer using general knowledge when no documents match."""
    llm = get_llm()
    messages = _build_messages(FALLBACK_SYSTEM_PROMPT, query, None, chat_history)
    response = llm.invoke(messages)
    return response.content

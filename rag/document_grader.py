import json
import logging
from rag.llm import get_llm

logger = logging.getLogger(__name__)

def grade_document(query: str, document_content: str) -> dict:
    from rag.retriever import is_comparison_query
    
    llm = get_llm()
    
    comparison_guidance = ""
    if is_comparison_query(query):
        comparison_guidance = "\nNOTE: The user is asking to compare or summarize multiple documents. This is just ONE chunk from ONE document. Grade it as 'Correct' if it contains ANY information relevant to the themes or arguments requested. Do not penalize it for failing to answer the entire comparison query on its own."
        
    prompt = f"""You are a strict grading assistant evaluating document retrieval.
Your task is to evaluate if a retrieved document chunk contains information relevant to the user's query.
{comparison_guidance}

User Query: {query}
Document Chunk: {document_content}

Evaluate the document chunk based on its relevance to the query.
Return your evaluation as a strict JSON object with EXACTLY these three keys:
- "label": strictly one of ["Correct", "Incorrect", "Ambiguous"]
- "explanation": a concise 1-sentence explanation for your label
- "confidence": an integer between 0 and 100 representing your confidence in this grade

Output ONLY the JSON object. Do not include markdown blocks or any other text.
"""
    messages = [
        ("system", "You are an expert document relevance grader. You always output valid JSON and nothing else."),
        ("human", prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        # Attempt to parse the content as JSON. Clean up markdown backticks if present.
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        result = json.loads(content)
        
        # Validate expected label
        if result.get("label") not in ["Correct", "Incorrect", "Ambiguous"]:
            result["label"] = "Ambiguous"
            
        # Ensure confidence is an int
        if "confidence" in result:
            result["confidence"] = int(result["confidence"])
            
        return result
    except Exception as e:
        logger.error(f"Error grading document: {str(e)}")
        return {
            "label": "Ambiguous",
            "explanation": "Failed to parse or generate grade.",
            "confidence": 0
        }

def refine_knowledge(query: str, results: list) -> dict:
    """Refine retrieved documents by segregating them based on their grade.
    
    Args:
        query: User's query
        results: List of (Document, score) tuples
        
    Returns:
        dict: A structured response containing segregated chunks and status.
    """
    correct_chunks = []
    ambiguous_chunks = []
    removed_chunks = []
    
    for doc, score in results:
        grade = grade_document(query, doc.page_content)
        chunk_info = {
            "doc": doc,
            "score": score,
            "grade": grade
        }
        
        if grade["label"] == "Correct":
            correct_chunks.append(chunk_info)
        elif grade["label"] == "Ambiguous":
            ambiguous_chunks.append(chunk_info)
        else: # Incorrect
            removed_chunks.append(chunk_info)
            
    status = "Failure" if not correct_chunks and not ambiguous_chunks else "Success"
            
    return {
        "correct_chunks": correct_chunks,
        "ambiguous_chunks": ambiguous_chunks,
        "removed_chunks": removed_chunks,
        "status": status
    }

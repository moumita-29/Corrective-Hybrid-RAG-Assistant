import json
import logging
from rag.llm import get_llm

logger = logging.getLogger(__name__)

def verify_answer(query: str, context_parts: list, answer: str) -> str:
    """Verify generated answer for groundedness, hallucination, and citations.
    
    Returns:
        'PASS' or 'FAIL'
    """
    if not context_parts:
        return "PASS"  # Cannot verify groundedness if no context was used (e.g. fallback)
        
    llm = get_llm()
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are an expert answer verifier.
Evaluate the following generated answer against the provided context and original query.

Query: {query}
Context:
{context}

Generated Answer:
{answer}

Evaluate based on these criteria:
1. Groundedness: Is the answer fully supported by the provided context?
2. Hallucination Risk: Does the answer introduce outside facts or assumptions not present in the context?
3. Missing Citations (Implicit): Are the claims in the answer traceable back to specific parts of the context?

Based on your evaluation, output a strict JSON object with EXACTLY one key:
- "status": strictly one of ["PASS", "FAIL"]

If the answer hallucinates or contradicts the context significantly, output FAIL. Otherwise, PASS.
Do not output any other text or explanation, ONLY the JSON object.
"""
    messages = [
        ("system", "You are an expert answer verifier. You always output valid JSON and nothing else."),
        ("human", prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        content = content.strip()
        result = json.loads(content)
        
        status = result.get("status", "FAIL")
        if status not in ["PASS", "FAIL"]:
            status = "FAIL"
            
        return status
    except Exception as e:
        logger.error(f"Error verifying answer: {str(e)}")
        # If the verifier fails, default to PASS to not interrupt user experience indefinitely
        return "PASS"

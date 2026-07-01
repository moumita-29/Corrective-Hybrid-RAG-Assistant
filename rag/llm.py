"""LLM wrapper using Groq API.

Replaces Hugging Face Inference API with Groq for extremely fast inference.
"""

import os
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_groq import ChatGroq
from config import GROQ_LLM_MODEL

# Load environment variables from .env (for GROQ_API_KEY)
load_dotenv()

# Enable in-memory caching for all LLM calls globally
set_llm_cache(InMemoryCache())

_llm = None


def get_llm():
    """Return a singleton ChatGroq instance.

    Connects to the Groq API.
    """
    global _llm
    if _llm is None:
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is missing! Please add it to your .env file or Streamlit secrets.")

        _llm = ChatGroq(
            model_name=GROQ_LLM_MODEL,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=0.0
        )
    return _llm

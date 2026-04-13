"""
LLM Engine - Connects to Groq API for fast, free LLM inference.
Used by the LangChain Pandas Agent for natural language query processing.
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional at import time; environment variables still work.
    pass

# Model mặc định — có thể override qua .env
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))


def get_llm(model: str = None, temperature: float = 0):
    """
    Create and return an LLM client connected via Groq.

    Args:
        model: Groq model name. Default is llama-3.1-8b-instant.
               See available models at: https://console.groq.com/docs/models
        temperature: 0 = deterministic (best for code/query), 0.7 = creative

    Returns:
        ChatGroq instance ready to use
    """
    try:
        from langchain_groq import ChatGroq

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY. Set it in environment or .env file."
            )

        selected_model = model or DEFAULT_MODEL

        llm = ChatGroq(
            model=selected_model,
            api_key=groq_api_key,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        logger.info(f"[LLM Engine] Initialized with model: {selected_model} (Groq)")
        return llm

    except ImportError as e:
        logger.error(f"[LLM Engine] langchain-groq not installed: {e}")
        logger.error("Run: pip install langchain-groq")
        raise
    except Exception as e:
        logger.error(f"[LLM Engine] Failed to initialize LLM: {e}")
        raise


def test_llm_connection():
    """Test LLM connection — call this when debugging."""
    try:
        llm = get_llm()
        response = llm.invoke("Say 'OK' in one word.")
        logger.info(f"[LLM Engine] Connection test PASSED: {response.content}")
        return True
    except Exception as e:
        logger.error(f"[LLM Engine] Connection test FAILED: {e}")
        return False

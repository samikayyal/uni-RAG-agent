"""Lazy construction of the separately configured answer chat model."""

from __future__ import annotations

from ..config import Config
from .models import AnswerGenerationError


def build_answer_chat_model(config: Config) -> object:
    """Construct exactly ``UNI_RAG_ANSWER_LLM_*`` and never planner settings."""
    provider = config.answer_llm_provider
    model = config.answer_llm_model
    if provider is None or model is None:
        raise AnswerGenerationError(
            "answer generation requires UNI_RAG_ANSWER_LLM_PROVIDER and "
            "UNI_RAG_ANSWER_LLM_MODEL"
        )
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=0)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, temperature=0)
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            from ..gemini_failover import build_gemini_with_failover

            def _build_gemini(api_key: str | None) -> object:
                return ChatGoogleGenerativeAI(
                    model=model,
                    temperature=0,
                    api_key=api_key,
                    thinking_level="medium",
                )

            return build_gemini_with_failover(_build_gemini, config)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model, temperature=0)
    except ImportError as exc:
        raise AnswerGenerationError(
            f"LLM provider '{provider}' requires the optional 'llm' extra. "
            "Install it with: uv sync --extra llm"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - construction is a fatal boundary
        raise AnswerGenerationError(
            f"Could not construct answer LLM provider '{provider}': {exc}"
        ) from exc
    raise AnswerGenerationError(f"Unsupported answer LLM provider: {provider}")

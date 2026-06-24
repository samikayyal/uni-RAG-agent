"""Uni RAG Agent package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uni-rag-agent")
except PackageNotFoundError:
    __version__ = "0.1.0"

"""FastAPI application boundary for the local answering interface."""

from .api import AppServices, create_app

__all__ = ["AppServices", "create_app"]

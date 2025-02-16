# app/__init__.py
# This file can be left empty, or you can expose important objects for easier access.

from app.core.embeddings import EmbeddingModel
from app.core.registries import ModelRegistry
from app.main import app

__all__ = ["app", "EmbeddingModel", "ModelRegistry"]

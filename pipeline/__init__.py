"""Sketch2Shape AI pipeline package (lazy import keeps light modules usable without torch)."""
from . import config  # noqa: F401  (must run first: sets HF_HOME before any HF library is imported)

__all__ = ["Sketch2ShapePipeline"]


def __getattr__(name):
    if name == "Sketch2ShapePipeline":
        from .pipeline import Sketch2ShapePipeline
        return Sketch2ShapePipeline
    raise AttributeError(name)

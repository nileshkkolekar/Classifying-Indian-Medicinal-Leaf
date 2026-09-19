"""FastAPI prediction service.

The HTTP surface over :class:`medicinal_leaf.inference.predictor.LeafPredictor`:
one image at a time, or a ZIP of many. Import ``app`` for ASGI hosting.
"""

from medicinal_leaf.api.schemas import (
    BatchResult,
    BatchSummary,
    HealthResponse,
    PredictionResult,
    Thresholds,
    Verdict,
)

__all__ = [
    "BatchResult",
    "BatchSummary",
    "HealthResponse",
    "PredictionResult",
    "Thresholds",
    "Verdict",
]

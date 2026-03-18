"""
Trust Tone Evaluator - Prosody analysis for trust assessment in human-robot interaction.

This module provides tools for extracting prosodic features from speech and
predicting trust-related indicators such as confidence, hesitation, and stress.
"""

__version__ = "0.1.0"

from trust_tone_evaluator.output.trust_metrics import (
    FusionReadyOutput,
    ProsodyFeatures,
    TrustIndicators,
    TrustLevel,
    TrustMetrics,
)

__all__ = [
    "TrustLevel",
    "TrustIndicators",
    "ProsodyFeatures",
    "TrustMetrics",
    "FusionReadyOutput",
]

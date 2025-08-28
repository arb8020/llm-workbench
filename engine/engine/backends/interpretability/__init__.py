"""Interpretability backends for specialized inference patterns."""

from .amplified_sampling import AmplifiedSampler
from .activation_collection import ActivationCollector

__all__ = [
    "AmplifiedSampler",
    "ActivationCollector"
]